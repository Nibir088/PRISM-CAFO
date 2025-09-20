import os
import argparse
import json
import yaml
import numpy as np
import torch
from pytorch_lightning.strategies import DDPStrategy

from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from model.classifierModule_v2 import CAFOClassifier
from datas.cafodata_module import CAFODataModule

# --- perf knobs ---
torch.set_float32_matmul_precision("medium")   # TF32 on Ampere for FP32 matmuls
torch.backends.cudnn.benchmark = True          # autotune convs when shapes are stable


# ---------- EVALUATION ----------
def evaluate_model(model, dataloader, num_classes, use_amp=True):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels, all_logits = [], [], []

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda",
                                enabled=use_amp and device.type == "cuda",
                                dtype=amp_dtype):
            for batch in dataloader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                x = batch["rgb"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)

                logits = model(batch)['logits']                          # possibly bf16/fp16
                preds  = logits.argmax(dim=1)

                all_logits.append(logits.float().cpu())    # -> fp32 for NumPy
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

    # concat once
    logits_np = torch.cat(all_logits, dim=0).numpy()       # [N, C] float32
    preds_np  = torch.cat(all_preds).numpy()               # [N]
    labels_np = torch.cat(all_labels).numpy()              # [N]

    # metrics
    report = classification_report(labels_np, preds_np, output_dict=True)
    conf   = confusion_matrix(labels_np, preds_np)

    # stable softmax in NumPy for AP
    x = logits_np - logits_np.max(axis=1, keepdims=True)
    probs_np = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    y_true_bin = label_binarize(labels_np, classes=list(range(num_classes)))
    map_macro  = float(average_precision_score(y_true_bin, probs_np, average="macro"))

    return report, conf, map_macro


# ---------- MAIN ----------
def main(config_file):
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # === Data (defaults in your CAFODataModule now use the fast 'cls' collate) ===
    dm = CAFODataModule(
        train_jsonl=cfg["train_jsonl"],
        test_jsonl=cfg["test_jsonl"],
        val_jsonl=None,            # no val file → will split from train
        resize=(cfg['size'], cfg['size']),         # change to (224,224) later if you want more speed
        transform=None,            # use dataset/default transforms
        batch_size=cfg['batch_size'],
        num_workers=cfg['worker'],
        pin_memory=True,
        shuffle_train=True,
        split_seed=42,
        val_ratio=0.15,
        drop_last=False,
        collate_train_mode="full",  # default is "cls" (minimal)
        collate_eval_mode="full",
        # channels_last=False,
    )

    # === Model ===
    model = CAFOClassifier(
        num_classes=cfg["num_classes"],
        lr=cfg["lr"],
        model_name=cfg["model_name"],
        use_attn_mask = cfg['use_attn_mask'],
        remoteclip_ckpt_path = cfg['ckpt_path'],
        use_spatial_attn=cfg['use_spatial_attn'],
        use_attn_pooling=cfg['use_attn_pooling']
    )
    print(model)

    # === Logger and Callbacks ===
    logger = CSVLogger(save_dir=cfg["log_dir"], name=f"{cfg['data_type']}_{cfg['model_name']}")
    checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True)
    earlystop_cb = EarlyStopping(monitor="val_acc", patience=cfg["patience"], mode="max")

    # === Trainer (DDP only if >1 GPU) ===
    use_gpu = torch.cuda.is_available() and cfg["gpus"] > 0
    num_devices = cfg["gpus"] if use_gpu else 1
    # strategy = "ddp" if (use_gpu and cfg["gpus"] > 1) else "auto"
    
    if use_gpu and cfg["gpus"] > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        accelerator="gpu" if use_gpu else "cpu",
        devices=num_devices,
        precision=cfg["precision"],    # e.g., "bf16-mixed" or "16-mixed" or "32-true"
        log_every_n_steps=10,
        strategy=strategy,
    )

    # === Train ===
    trainer.fit(model, datamodule=dm)

    # Use best checkpoint (if available) for reporting
    model_for_eval = model
    if checkpoint_cb.best_model_path:
        # load best weights (Lightning restores hparams automatically)
        model_for_eval = CAFOClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
        model_for_eval.eval()

    # === Post-train evaluation only on global rank 0 ===
    if getattr(trainer, "is_global_zero", True):
        print("Evaluate on validation set")
        val_report, val_conf_matrix, val_map = evaluate_model(
            model_for_eval, dm.val_dataloader(), cfg["num_classes"]
        )

        print("Evaluate on test set")
        test_report, test_conf_matrix, test_map = evaluate_model(
            model_for_eval, dm.test_dataloader(), cfg["num_classes"]
        )

        # === Save all results ===
        results = {
            "model": cfg["model_name"],
            "best_val_acc": float(trainer.callback_metrics.get("val_acc", -1)),
            "best_val_f1": float(trainer.callback_metrics.get("val_f1", -1)),
            "log_dir": logger.log_dir,
            "checkpoint_path": checkpoint_cb.best_model_path,
            "val_classification_report": val_report,
            "val_confusion_matrix": val_conf_matrix.tolist(),
            "val_mAP": float(val_map),
            "test_classification_report": test_report,
            "test_confusion_matrix": test_conf_matrix.tolist(),
            "test_mAP": float(test_map),
        }

        output_path = f"results/{cfg['data_type']}_{cfg['model_name']}_attn_{cfg['use_attn_mask']}_spa_{cfg['use_spatial_attn']}_pool_{cfg['use_attn_pooling']}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Full training results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="config/config_fold1_v4.yaml")
    args = parser.parse_args()
    main(config_file=args.config_file)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score,MulticlassAveragePrecision
from huggingface_hub import hf_hub_download


import timm  # required for DINOv2, EfficientNet etc.
import open_clip  # for CLIP and RemoteCLIP


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def make_clip_preprocess(img_size: int):
    """CLIP-like preprocessing for a square size (must be multiple of 32 for ViT-B/32)."""
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def resize_openclip_pos_embed(visual: nn.Module, image_size: int, patch: int = 32):
    """
    Resize OpenCLIP vision transformer positional embeddings to a new image size.
    visual.positional_embedding: [1 + H0*W0, C] (CLS + patch grid flattened)
    """
    with torch.no_grad():
        pe = visual.positional_embedding           # [N+1, C]
        cls_pos = pe[:1]                           # [1, C]
        patch_pos = pe[1:]                         # [N, C]

        n = patch_pos.shape[0]
        orig = int(n ** 0.5)                       # e.g., 7 for 224/32
        C = patch_pos.shape[1]

        # [1, C, orig, orig] -> interpolate -> [1, C, new, new]
        patch_pos = patch_pos.view(1, orig, orig, C).permute(0, 3, 1, 2)
        new = image_size // patch                  # e.g., 704/32 = 22
        patch_pos = F.interpolate(patch_pos, size=(new, new), mode='bicubic', align_corners=False)

        # back to [new*new, C] and concat CLS
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new * new, C)
        visual.positional_embedding = nn.Parameter(torch.cat([cls_pos, patch_pos], dim=0))


class CAFOClassifier(pl.LightningModule):
    def __init__(self, num_classes=6, lr=1e-4, model_name='resnet18'):
        super().__init__()
        self.save_hyperparameters()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # elif model_name == 'vit_b_16':
        #     self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        #     self.backbone.heads = nn.Linear(self.backbone.heads.in_features, num_classes)
        elif model_name == 'vit_b_16':
            # self.backbone = models.vit_b_16(weights=models.vit_base_patch16_384.IMAGENET1K_V1)
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
            # Safely get in_features from the final Linear layer inside the heads
            if isinstance(self.backbone.heads, nn.Sequential):
                in_features = self.backbone.heads[-1].in_features
            else:
                in_features = self.backbone.heads.in_features

            self.backbone.heads = nn.Linear(in_features, num_classes)


        elif model_name == 'swin_b':
            self.backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

        elif model_name == 'dinov2_vit_b':
            self.backbone = timm.create_model("vit_base_patch16_224.dino", pretrained=True)

            # DINOv2 uses an identity head by default; replace it
            if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Identity):
                in_features = self.backbone.num_features  # use num_features for feature dimension
                self.backbone.head = nn.Linear(in_features, num_classes)
            else:
                raise AttributeError("Unexpected head structure in DINOv2 model")
        elif model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone.classifier[2] = nn.Linear(self.backbone.classifier[2].in_features, num_classes)
        elif model_name == 'clip':
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            # choose an image size that is a multiple of 32 (e.g., 704 or 736)
            clip_img_size = 704
            resize_openclip_pos_embed(model.visual, clip_img_size, patch=32)
            self.clip_image_encoder = model.visual
            self.backbone = nn.Sequential(
                self.clip_image_encoder,
                nn.Linear(model.visual.output_dim, num_classes)
            )
            # store desired CLIP preprocess somewhere you can pass into your DataModule
            self.clip_preprocess = make_clip_preprocess(clip_img_size)
            
        elif model_name == 'remoteclip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
            ckpt_path = "/project/biocomplexity/wyr6fx(Nibir)/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            message = model.load_state_dict(ckpt)

            
            self.clip_image_encoder = model.visual
            self.backbone = nn.Sequential(
                self.clip_image_encoder,
                nn.Linear(model.visual.output_dim, num_classes)
            )
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_map = MulticlassAveragePrecision(num_classes=num_classes, average="macro")


    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["rgb"], batch["label"].long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)

        # update metric states
        self.train_acc(preds, y)

        # log
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["rgb"], batch["label"].long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_map(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc",  self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1",   self.val_f1,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_map",  self.val_map, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

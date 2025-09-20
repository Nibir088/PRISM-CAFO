#!/usr/bin/env python3
import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
# from tqdm import tqdm
# ===== CONFIG =====
# CSV_PATH     = "/project/biocomplexity/gza5dr/CAFO_Test/csvs/predict_all_cafo_wNeg.csv"  # <- your CSV
CSV_PATH = "/project/biocomplexity/gza5dr/CAFO_Test/mask_exports/labels.csv"
MODEL_PATH   = "/project/biocomplexity/gza5dr/CAFO_Test/yolo_runs_new/yolov8s_cafo_832/weights/best.pt"  # <- your trained model
IMG_COL      = "image"
OUT_DIR    = "/project/biocomplexity/gza5dr/CAFO_Test/mask_exports"           # outputs here
OUT_CSV    = str(Path(OUT_DIR) / "yolo_predicts_114.csv")

IMGSZ      = 832
CONF_THRES = 0.3

# ===== helpers =====
def load_img_bgr(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")                 # drop extra bands if any (e.g., NIR)
    arr = np.asarray(im)
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / max(1, arr.max() - arr.min())).astype(np.uint8)
    return arr[..., ::-1]  # RGB->BGR for Ultralytics

def save_annotated(result, out_path: Path):
    ann_bgr = result.plot()                    # numpy BGR
    ann_rgb = ann_bgr[:, :, ::-1]
    Image.fromarray(ann_rgb).save(out_path)

# ===== load =====
df = pd.read_csv(CSV_PATH)
imgs = [p for p in df[IMG_COL].astype(str).unique() if Path(p).exists()]
Path(OUT_DIR, "images").mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)
name_map = {int(k): v for k, v in model.names.items()}

rows = []
for ipath in imgs:
    try:
        arr_bgr = load_img_bgr(ipath)
        results = model.predict(source=arr_bgr, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
        if not results:
            continue
        r = results[0]

        # save annotated image
        out_img = Path(OUT_DIR, "images", f"{Path(ipath).stem}.png")
        save_annotated(r, out_img)

        # collect detections
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            for (x1,y1,x2,y2), c, s in zip(xyxy, cls, conf):
                rows.append({
                    "image": ipath,
                    "x1": int(round(x1)), "y1": int(round(y1)),
                    "x2": int(round(x2)), "y2": int(round(y2)),
                    "label": name_map.get(int(c), str(int(c))),
                    "conf": float(s),
                    "annotated_path": str(out_img),
                })
        else:
            # still record that we processed the image (no detections)
            rows.append({
                "image": ipath, "x1": None, "y1": None, "x2": None, "y2": None,
                "label": "", "conf": None, "annotated_path": str(out_img),
            })
        print(f"[ok] {ipath}")
    except Exception as e:
        print(f"[warn] {ipath}: {e}")

pred_df = pd.DataFrame(rows)
tmp = str(Path(OUT_CSV).with_suffix(".tmp.csv"))
pred_df.to_csv(tmp, index=False)
os.replace(tmp, OUT_CSV)

print(f"\nImages processed: {len(imgs)}")
print(f"Predictions CSV : {OUT_CSV}")
print(f"Annotated imgs  : {Path(OUT_DIR,'images')}")

from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # or "yolov8n.pt" if you want lighter

model.train(
    data="/project/biocomplexity/gza5dr/CAFO_Test/yolo_cafo_2/cafo_yolo.yaml",
    imgsz=832,
    epochs=120,
    batch=0,          # 0 = auto-batch in Python API
    rect=True,
    workers=4,        # adjust if needed
    device=0,         # GPU 0; use 'cpu' if no GPU
    project="/project/biocomplexity/gza5dr/CAFO_Test/yolo_runs_new",
    name="yolov8s_cafo_832"
)

# from ultralytics import YOLO
# model = YOLO("yolov8n-seg.pt")  # or yolov8s-seg.pt
# model.train(
#     data="/project/biocomplexity/gza5dr/CAFO_Test/yolo_seg_dataset/data.yaml",
#     imgsz=832, epochs=100, batch=8, device=0, workers=8,
#     project="/project/biocomplexity/gza5dr/CAFO_Test/yolo_runs", name="v8n_seg_1024"
# )


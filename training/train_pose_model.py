from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n-pose.pt"
DATASET_CONFIG = Path(__file__).resolve().parent / "car_dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

if __name__ == "__main__":
    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATASET_CONFIG),
        epochs=150,
        imgsz=416,
        batch=16,
        workers=2,
        device=0 if torch.cuda.is_available() else "cpu",
        weight_decay=0.005,
        pose=15.0,
        box=7.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.7,
        perspective=0.0005,
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        scale=0.2,
        mosaic=0.0,
        save=True,
        project=str(RUNS_DIR),
        name="v3_model_nano",
    )

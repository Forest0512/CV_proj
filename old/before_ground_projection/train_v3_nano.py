from ultralytics import YOLO

if __name__ == '__main__':
    # Load the pretrained pose model
    model = YOLO('yolov8n-pose.pt')

    # Start training (optimized for keypoint regression)
    model.train(
        data='car_dataset.yaml',
        epochs=150,             # Slightly more epochs so pose and shadow features fully converge
        imgsz=416,              # Important! 416 greatly reduces compute; key to reaching 60+ FPS on a laptop
        batch=16,               # For small datasets, lowering batch to 8 gives more frequent updates and reduces overfitting
        workers=2,
        device=0,

        # --- Regularization and keypoint weights ---
        weight_decay=0.005,     # Stronger L2 regularization to avoid overfitting on few samples
        pose=15.0,              # Higher keypoint loss weight forces the model to lock onto front and tail
        box=7.5,

        # --- Shadow and lighting robustness (HSV augmentation) ---
        hsv_h=0.015,            # Slight hue variation
        hsv_s=0.7,              # Wide saturation variation
        hsv_v=0.7,              # Wider brightness range (default 0.4 -> 0.7), greatly improves detection in shadows

        # --- Geometric augmentation ---
        perspective=0.0005,     # Slight perspective warp to simulate camera tilt
        degrees=180,            # Rotation augmentation to cover the car's turning angles
        fliplr=0.5,             # Horizontal flip (requires flip_idx: [1, 0] in car_dataset.yaml)
        flipud=0.5,             # Vertical flip (50% probability)
        scale=0.2,              # Slight scaling
        mosaic=0.0,             # Mosaic disabled for small datasets to avoid crops breaking the keypoint structure

        save=True,
        project='toy_car_project',
        name='v3_model_nano'
    )
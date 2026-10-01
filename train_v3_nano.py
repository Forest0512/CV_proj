from ultralytics import YOLO

if __name__ == '__main__':
    # 載入預訓練 Pose 模型
    model = YOLO('yolov8n-pose.pt')

    # 開始訓練 (優化關鍵點回歸)
    model.train(
        data='car_dataset.yaml',
        epochs=150,             # 稍微增加 Epochs，讓 Pose 與陰影特徵充分收斂
        imgsz=416,              # 關鍵！改為 416 可大幅減少計算量，是筆記本達成 60+ FPS 的關鍵
        batch=16,               # 小數據集建議將 batch 降至 8，梯度更新更頻繁，防止過擬合
        workers=2,
        device=0,
        
        # --- 正則化與關鍵點權重 ---
        weight_decay=0.005,     # 增加 L2 正則化，避免對少量樣本過擬合
        pose=15.0,              # 提高 keypoint 損失權重，強制模型鎖定車頭與車尾
        box=7.5,
        
        # --- 陰影與光影抗性增強 (HSV Augmentation) ---
        hsv_h=0.015,            # 色調微調
        hsv_s=0.7,              # 高飽和度變動範圍
        hsv_v=0.7,              # 提高明度變動範圍（預設 0.4 -> 0.7），大幅提升陰影處辨識率
        
        # --- 幾何增強 ---
        perspective=0.0005,     # 輕微透視變形，模擬視角傾斜
        degrees=180,            # 旋轉增強，適應小車轉彎角度
        fliplr=0.5,             # 左右翻轉（需確認 car_dataset.yaml 已設定 flip_idx: [1, 0]）
        flipud=0.5,             # 上下翻轉 (50% 機率)
        scale=0.2,              # 輕微縮放
        mosaic=0.0,             # 小數據集建議關閉 Mosaic，避免剪裁破壞小車關鍵點結構   

        save=True,
        project='toy_car_project',
        name='v3_model_nano'         
    )
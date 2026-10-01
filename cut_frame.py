import cv2
import os

video_path = "shortover.mp4"  # 換成你的影片檔名
output_dir = "C:/Users/Aphrodite/Desktop/Master_Class/CV_proj/dataset/images/add"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 每 6 幀取 1 張圖 (60fps 情況下相當於每秒 10 張)
    if frame_count % 6 == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"完成！共抽出 {saved_count} 張圖片，存於 {output_dir}")

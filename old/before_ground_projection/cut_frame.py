import cv2
import os

video_path = "shortover.mp4"  # Replace with your video file name
output_dir = "C:/Users/Aphrodite/Desktop/Master_Class/CV_proj/dataset/images/add"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Keep 1 of every 6 frames (10 images per second at 60 fps)
    if frame_count % 6 == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Done! Extracted {saved_count} images, saved to {output_dir}")

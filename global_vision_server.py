import cv2
import math
import time
import socket
import argparse
import numpy as np
from ultralytics import YOLO

# 1. 解析 CLI 參數
parser = argparse.ArgumentParser(description="Global Vision System Server")
parser.add_argument('--port', type=int, default=5000, help='UDP target port')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='UDP target IP')
parser.add_argument('--cam', type=int, default=0, help='Camera index')
parser.add_argument('--model', type=str, default='best.pt', help='YOLO model path')
args = parser.parse_args()

# 2. Homography 矩陣標定 (像素 (u,v) -> 真實毫米 (x,y))
pts_image = np.array([
    [237, 180],
    [1024, 185],
    [1050, 700],
    [210, 695]
], dtype=np.float32)

pts_real = np.array([
    [0.0, 0.0],
    [1500.0, 0.0],
    [1500.0, 2500.0],
    [0.0, 2500.0]
], dtype=np.float32)

H_matrix, _ = cv2.findHomography(pts_image, pts_real)

def pixel_to_real(u, v):
    pt = np.array([[[u, v]]], dtype=np.float32)
    real_pt = cv2.perspectiveTransform(pt, H_matrix)
    return float(real_pt[0][0][0]), float(real_pt[0][0][1])

# 3. 初始化 Socket 與相機
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
start_timestamp = time.time_ns() // 1000

cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

model = YOLO(args.model)

prev_time = None
prev_x, prev_y = None, None
prev_theta = None
CAR_ID = "Yellow Racer #7"

print(f"Global Vision Server Started on {args.ip}:{args.port}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time_us = (time.time_ns() // 1000) - start_timestamp
    current_time_sec = time.time()

    results = model(frame, conf=0.3, verbose=False)
    car_detected = False

    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            
            if len(kpts) >= 2 and kpts[0][0] > 0 and kpts[1][0] > 0:
                car_detected = True
                head_u, head_v = kpts[0]
                tail_u, tail_v = kpts[1]

                center_u = int((head_u + tail_u) / 2.0)
                center_v = int((head_v + tail_v) / 2.0)

                real_x, real_y = pixel_to_real(center_u, center_v)
                theta = math.degrees(math.atan2(-(head_v - tail_v), head_u - tail_u))

                dx, dy, omega = 0.0, 0.0, 0.0
                if prev_time is not None:
                    dt = current_time_sec - prev_time
                    if dt > 0:
                        dx = (real_x - prev_x) / dt
                        dy = (real_y - prev_y) / dt
                        
                        d_theta = theta - prev_theta
                        if d_theta > 180: d_theta -= 360
                        elif d_theta < -180: d_theta += 360
                        omega = d_theta / dt

                prev_time = current_time_sec
                prev_x, prev_y = real_x, real_y
                prev_theta = theta

                msg = f'{current_time_us}:"{CAR_ID}",{real_x:.1f},{real_y:.1f},{theta:.1f},{dx:.1f},{dy:.1f},{omega:.1f},{center_u},{center_v}\n'
                udp_sock.sendto(msg.encode('utf-8'), (args.ip, args.port))

                cv2.circle(frame, (int(head_u), int(head_v)), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(tail_u), int(tail_v)), 4, (255, 0, 0), -1)
                cv2.arrowedLine(frame, (int(tail_u), int(tail_v)), (int(head_u), int(head_v)), (0, 255, 255), 2)
                cv2.putText(frame, f"X:{real_x:.0f} Y:{real_y:.0f} Ang:{theta:.1f}deg", 
                            (center_u + 10, center_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    if not car_detected:
        msg = f'{current_time_us}:"{CAR_ID}",-1000.0,-1000.0,-1000.0,0.0,0.0,0.0,-1,-1\n'
        udp_sock.sendto(msg.encode('utf-8'), (args.ip, args.port))

    cv2.imshow("Global Vision Server (60 FPS)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
udp_sock.close()
cv2.destroyAllWindows()
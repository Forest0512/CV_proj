import time
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    # 1. 載入訓練好的 3-Keypoint 最佳權重
    model_path = r"runs/pose/toy_car_project/v3_model_nano-5/weights/best.pt"
    model = YOLO(model_path)

    # 2. 設定小車影片路徑
    video_path = r"toycar2.mp4"
    #video_path = r"output.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 錯誤：無法開啟影片來源 '{video_path}'")
        return

    window_name = "Global Vision System - Realtime Tracking (Smoothed)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0

    dt = 1.0 / video_fps
    delay_ms = max(1, int(1000 / video_fps))

    # --- EMA 平滑係數 (alpha 越小越平滑，但反應會微延遲；0.25~0.35 效果最好) ---
    ALPHA_POS = 0.65  # 座標平滑係數
    ALPHA_DEG = 0.6  # 角度平滑係數

    # 保存上一幀平滑後的數據
    smooth_front = None
    smooth_tail = None
    smooth_theta = None

    prev_center = None
    last_known_box = None
    last_known_kpts = None
    missed_frames = 0
    MAX_MISSED_FRAMES = 12

    is_paused = False
    prev_frame_time = time.time()

    while cap.isOpened():
        start_time = time.time()

        curr_frame_time = time.time()
        time_diff = curr_frame_time - prev_frame_time
        realtime_fps = (1.0 / time_diff) if time_diff > 0 else 0.0
        prev_frame_time = curr_frame_time

        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                smooth_front = None
                smooth_tail = None
                smooth_theta = None
                prev_center = None
                last_known_box = None
                last_known_kpts = None
                missed_frames = 0
                ret, frame = cap.read()
                if not ret:
                    break

        display_frame = frame.copy()

        results = model(display_frame, conf=0.4, verbose=False)

        detected = False
        for result in results:
            if (
                result.boxes is not None
                and result.keypoints is not None
                and len(result.keypoints) > 0
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                kpts = result.keypoints.xy.cpu().numpy()

                if len(kpts) > 0 and len(kpts[0]) >= 3:
                    detected = True
                    missed_frames = 0
                    last_known_box = boxes[0]
                    last_known_kpts = kpts[0]

        if not detected:
            missed_frames += 1
            if missed_frames <= MAX_MISSED_FRAMES and last_known_box is not None:
                box_to_draw = last_known_box
                kpts_to_draw = last_known_kpts
            else:
                box_to_draw = None
                kpts_to_draw = None
        else:
            box_to_draw = last_known_box
            kpts_to_draw = last_known_kpts

        if box_to_draw is not None and kpts_to_draw is not None:
            x1, y1, x2, y2 = map(int, box_to_draw)
            box_color = (0, 255, 0) if detected else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

            head1_x, head1_y = kpts_to_draw[0]
            head2_x, head2_y = kpts_to_draw[1]
            tail_x, tail_y = kpts_to_draw[2]

            if head1_x > 0 and head2_x > 0 and tail_x > 0:
                # 1. 原始檢測計算
                raw_front_x = (head1_x + head2_x) / 2.0
                raw_front_y = (head1_y + head2_y) / 2.0
                raw_tail_x = tail_x
                raw_tail_y = tail_y

                # 2. 對【車頭中心點】與【車尾點】做 EMA 低通濾波平滑處理
                if smooth_front is None or not detected:
                    smooth_front = np.array([raw_front_x, raw_front_y])
                    smooth_tail = np.array([raw_tail_x, raw_tail_y])
                else:
                    smooth_front = ALPHA_POS * np.array(
                        [raw_front_x, raw_front_y]
                    ) + (1 - ALPHA_POS) * smooth_front
                    smooth_tail = ALPHA_POS * np.array(
                        [raw_tail_x, raw_tail_y]
                    ) + (1 - ALPHA_POS) * smooth_tail

                front_x, front_y = smooth_front
                tail_x_sm, tail_y_sm = smooth_tail

                # 3. 計算平滑後的車輛幾何中心點
                center_x = (front_x + tail_x_sm) / 2.0
                center_y = (front_y + tail_y_sm) / 2.0

                # 4. 計算角度與角度平滑 (防止角度跳動)
                dx = front_x - tail_x_sm
                dy = front_y - tail_y_sm
                raw_theta_deg = np.degrees(np.arctan2(-dy, dx))

                if smooth_theta is None or not detected:
                    smooth_theta = raw_theta_deg
                else:
                    # 處理角度跨越 ±180 度的問題
                    diff = (raw_theta_deg - smooth_theta + 180) % 360 - 180
                    smooth_theta = smooth_theta + ALPHA_DEG * diff
                    smooth_theta = (smooth_theta + 180) % 360 - 180

                theta_deg = smooth_theta

                # 5. 速度與角速度計算
                v, omega = 0.0, 0.0
                if prev_center is not None and not is_paused and detected:
                    dist = np.sqrt(
                        (center_x - prev_center[0]) ** 2
                        + (center_y - prev_center[1]) ** 2
                    )
                    v = dist / dt

                    d_theta = (
                        theta_deg - (smooth_theta - ALPHA_DEG * diff)
                    )  # 增量
                    omega = diff / dt

                if not is_paused and detected:
                    prev_center = (center_x, center_y)

                # 6. 繪製平滑後的點位與箭頭 (箭頭不會再狂抖)
                cv2.circle(
                    display_frame,
                    (int(head1_x), int(head1_y)),
                    3,
                    (255,255, 0),
                    -1,
                )
                cv2.circle(
                    display_frame,
                    (int(head2_x), int(head2_y)),
                    3,
                    (255,255, 0),
                    -1,
                )
                cv2.circle(
                    display_frame,
                    (int(tail_x_sm), int(tail_y_sm)),
                    5,
                    (255, 0, 0),
                    -1,
                )
                cv2.circle(
                    display_frame,
                    (int(center_x), int(center_y)),
                    5,
                    (0, 255, 255),
                    -1,
                )

                # 畫出穩定平滑的方向矢線
                cv2.arrowedLine(
                    display_frame,
                    (int(tail_x_sm), int(tail_y_sm)),
                    (int(front_x), int(front_y)),
                    ( 0, 0, 255),
                    6,
                    tipLength=0.25,
                )

                info_text = f"Pos:({int(center_x)},{int(center_y)}) | Angle:{theta_deg:.1f}deg | V:{v:.1f}px/s | W:{omega:.1f}deg/s"
                cv2.putText(
                    display_frame,
                    info_text,
                    (int(center_x) - 120, int(center_y) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        status_text = (
            "PAUSED" if is_paused else f"FPS: {realtime_fps:.1f} (Looping)"
        )
        status_color = (0, 0, 255) if is_paused else (0, 255, 0)
        cv2.putText(
            display_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        cv2.imshow(window_name, display_frame)

        elapsed_ms = int((time.time() - start_time) * 1000)
        actual_delay = max(1, delay_ms - elapsed_ms) if not is_paused else 0

        key = cv2.waitKey(actual_delay) & 0xFF

        if key == ord("q"):
            break
        elif key == 32:
            is_paused = not is_paused
        elif key == ord("d") and is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        elif key == ord("a") and is_paused:
            current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_idx - 2))
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
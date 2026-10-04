import time
import cv2
import numpy as np
from ultralytics import YOLO


class CarTracker:
    """用來獨立記錄每台車追蹤狀態的類別"""

    def __init__(self, name, box_color, arrow_color):
        self.name = name  # 車輛名稱 (car / car2)
        self.box_color = box_color  # 框顏色 (BGR)
        self.arrow_color = arrow_color  # 箭頭顏色 (BGR)

        self.smooth_front = None
        self.smooth_tail = None
        self.smooth_theta = None
        self.prev_center = None

        self.last_known_box = None
        self.last_known_kpts = None
        self.missed_frames = 0

    def reset(self):
        """影片重播時重置狀態"""
        self.smooth_front = None
        self.smooth_tail = None
        self.smooth_theta = None
        self.prev_center = None
        self.last_known_box = None
        self.last_known_kpts = None
        self.missed_frames = 0


def main():
    # 1. 載入訓練好的雙類別 3-Keypoint 最佳權重
    model_path = r"v4_model/weights/best.pt"
    model = YOLO(model_path)

    # 2. 設定影片路徑
    video_path = r"long_car2.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 錯誤：無法開啟影片來源 '{video_path}'")
        return

    window_name = "Global Vision System - Dual Car Realtime Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0

    dt = 1.0 / video_fps
    delay_ms = max(1, int(1000 / video_fps))

    # EMA 平滑參數
    ALPHA_POS = 0.8
    ALPHA_DEG = 0.8
    MAX_MISSED_FRAMES = 12

    # 設定兩種車輛的追蹤器 (Class 0: car, Class 1: car2)
    trackers = {
        0: CarTracker(
            name="car", box_color=(0, 255, 0), arrow_color=(0, 0, 255)
        ),  # 綠框、紅箭頭
        1: CarTracker(
            name="car2", box_color=(255, 165, 0), arrow_color=(255, 0, 255)
        ),  # 橙框、粉紫箭頭
    }

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
                for t in trackers.values():
                    t.reset()
                ret, frame = cap.read()
                if not ret:
                    break

        display_frame = frame.copy()

        # 模型推論
        results = model(display_frame, conf=0.4, verbose=False)

        # 標記這一幀各類別是否有被偵測到
        detected_classes = {0: False, 1: False}

        for result in results:
            if (
                result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)  # 取得類別代碼
                kpts = result.keypoints.xy.cpu().numpy()

                # 分配每個偵測到的物體給對應類別的追蹤器
                for idx, cls_id in enumerate(classes):
                    if cls_id in trackers and idx < len(kpts):
                        tracker = trackers[cls_id]
                        if len(kpts[idx]) >= 3:
                            detected_classes[cls_id] = True
                            tracker.missed_frames = 0
                            tracker.last_known_box = boxes[idx]
                            tracker.last_known_kpts = kpts[idx]

        # 針對每一台車做位置更新、平滑處理與繪製
        for cls_id, tracker in trackers.items():
            detected = detected_classes[cls_id]

            if not detected:
                tracker.missed_frames += 1
                if (
                    tracker.missed_frames <= MAX_MISSED_FRAMES
                    and tracker.last_known_box is not None
                ):
                    box_to_draw = tracker.last_known_box
                    kpts_to_draw = tracker.last_known_kpts
                else:
                    box_to_draw = None
                    kpts_to_draw = None
            else:
                box_to_draw = tracker.last_known_box
                kpts_to_draw = tracker.last_known_kpts

            if box_to_draw is not None and kpts_to_draw is not None:
                x1, y1, x2, y2 = map(int, box_to_draw)
                box_color = tracker.box_color if detected else (128, 128, 128)

                # 繪製 Bbox 與車輛類別名稱
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    display_frame,
                    f"{tracker.name}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

                head1_x, head1_y = kpts_to_draw[0]
                head2_x, head2_y = kpts_to_draw[1]
                tail_x, tail_y = kpts_to_draw[2]

                if head1_x > 0 and head2_x > 0 and tail_x > 0:
                    # 1. 計算原始座標
                    raw_front_x = (head1_x + head2_x) / 2.0
                    raw_front_y = (head1_y + head2_y) / 2.0
                    raw_tail_x = tail_x
                    raw_tail_y = tail_y

                    # 2. EMA 位置平滑
                    if tracker.smooth_front is None or not detected:
                        tracker.smooth_front = np.array(
                            [raw_front_x, raw_front_y]
                        )
                        tracker.smooth_tail = np.array([raw_tail_x, raw_tail_y])
                    else:
                        tracker.smooth_front = ALPHA_POS * np.array(
                            [raw_front_x, raw_front_y]
                        ) + (1 - ALPHA_POS) * tracker.smooth_front
                        tracker.smooth_tail = ALPHA_POS * np.array(
                            [raw_tail_x, raw_tail_y]
                        ) + (1 - ALPHA_POS) * tracker.smooth_tail

                    front_x, front_y = tracker.smooth_front
                    tail_x_sm, tail_y_sm = tracker.smooth_tail

                    # 3. 計算中心點
                    center_x = (front_x + tail_x_sm) / 2.0
                    center_y = (front_y + tail_y_sm) / 2.0

                    # 4. 角度計算與平滑
                    dx = front_x - tail_x_sm
                    dy = front_y - tail_y_sm
                    raw_theta_deg = np.degrees(np.arctan2(-dy, dx))

                    if tracker.smooth_theta is None or not detected:
                        tracker.smooth_theta = raw_theta_deg
                        diff = 0.0
                    else:
                        diff = (
                            raw_theta_deg - tracker.smooth_theta + 180
                        ) % 360 - 180
                        tracker.smooth_theta = (
                            tracker.smooth_theta + ALPHA_DEG * diff
                        )
                        tracker.smooth_theta = (
                            tracker.smooth_theta + 180
                        ) % 360 - 180

                    theta_deg = tracker.smooth_theta

                    # 5. 線速度與角速度計算
                    v, omega = 0.0, 0.0
                    if (
                        tracker.prev_center is not None
                        and not is_paused
                        and detected
                    ):
                        dist = np.sqrt(
                            (center_x - tracker.prev_center[0]) ** 2
                            + (center_y - tracker.prev_center[1]) ** 2
                        )
                        v = dist / dt
                        omega = diff / dt

                    if not is_paused and detected:
                        tracker.prev_center = (center_x, center_y)

                    # 6. 繪製關鍵點與方向箭頭
                    cv2.circle(
                        display_frame,
                        (int(head1_x), int(head1_y)),
                        3,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.circle(
                        display_frame,
                        (int(head2_x), int(head2_y)),
                        3,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.circle(
                        display_frame,
                        (int(tail_x_sm), int(tail_y_sm)),
                        3,
                        (255, 0, 0),
                        -1,
                    )
                    cv2.circle(
                        display_frame,
                        (int(center_x), int(center_y)),
                        3,
                        (0, 255, 255),
                        -1,
                    )

                    cv2.arrowedLine(
                        display_frame,
                        (int(tail_x_sm), int(tail_y_sm)),
                        (int(front_x), int(front_y)),
                        tracker.arrow_color,
                        3,
                        tipLength=0.25,
                    )

                    # 顯示資訊文字（錯開兩台車的文字顯示位置）
                    text_y_offset = -20 if cls_id == 0 else 20
                    info_text = f"[{tracker.name}] Pos:({int(center_x)},{int(center_y)}) | Angle:{theta_deg:.1f}deg | V:{v:.1f}px/s"
                    cv2.putText(
                        display_frame,
                        info_text,
                        (int(center_x) - 120, int(center_y) + text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        tracker.box_color,
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
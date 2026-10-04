import argparse
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from toy_car_vision.car_tracking import CarState, MultiCarTracker, TrackStatus, detections_from_yolo_result
from toy_car_vision.ground_plane import CalibrationError, GroundPlaneCalibration
from toy_car_vision.paths import DEFAULT_CALIBRATION_PATH, DEFAULT_MODEL_PATH, resolve_input_path
from toy_car_vision.udp_protocol import format_car_message
from toy_car_vision.visualization import draw_calibration, draw_car_state, draw_status, draw_top_view

DEFAULT_SOURCE_FPS = 30.0
WINDOW_NAME = "Global Vision Server"
KEY_QUIT = ord("q")
KEY_PAUSE = ord(" ")
KEY_STEP = ord("d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global vision server for toy race cars")
    parser.add_argument("--source", default="0", help="camera index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO pose model weights")
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="ground plane calibration JSON (see toy-car-calibrate)",
    )
    parser.add_argument(
        "--uncalibrated", action="store_true", help="run without calibration; world values are sent as nan"
    )
    parser.add_argument("--ip", default="127.0.0.1", help="UDP target IP")
    parser.add_argument("--port", type=int, default=5000, help="UDP target port")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference size (model trained with 416)")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or 0 (default: auto)")
    parser.add_argument(
        "--car-names", default=None, help='override names per class id, e.g. "0=Yellow Racer,1=Blue Racer"'
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="EMA factor for world position/heading in (0,1]; 1.0 disables smoothing",
    )
    parser.add_argument(
        "--max-missed-frames", type=int, default=12, help="frames without detection before velocity history is reset"
    )
    parser.add_argument(
        "--min-keypoint-conf", type=float, default=0.5, help="minimum keypoint confidence when the model provides one"
    )
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--fps", type=int, default=60, help="requested camera frame rate")
    parser.add_argument("--loop", action="store_true", help="restart video files at the end")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--save-video", default=None, help="write the annotated video to this file")
    return parser.parse_args(argv)


def parse_car_names(model_names: dict, override: str | None) -> dict[int, str]:
    names = {int(class_id): str(name) for class_id, name in model_names.items()}
    name_counts = {name: list(names.values()).count(name) for name in names.values()}
    names = {class_id: f"{name}_{class_id}" if name_counts[name] > 1 else name for class_id, name in names.items()}
    if override:
        for entry in override.split(","):
            class_id, _, name = entry.partition("=")
            if not name:
                raise ValueError(f"invalid --car-names entry: {entry!r}")
            names[int(class_id)] = name.strip()
    return names


def open_source(args: argparse.Namespace) -> tuple[cv2.VideoCapture, bool]:
    if args.source.isdigit():
        capture = cv2.VideoCapture(int(args.source))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        return capture, False
    source_path = resolve_input_path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"video file not found: {args.source}")
    return cv2.VideoCapture(str(source_path)), True


def load_calibration(args: argparse.Namespace) -> GroundPlaneCalibration | None:
    if args.uncalibrated:
        print("WARNING: running uncalibrated, no world coordinates will be computed", file=sys.stderr)
        return None
    calibration = GroundPlaneCalibration.load(args.calibration)
    error = calibration.reprojection_error
    print(
        f"Calibration '{args.calibration}': {int(calibration.inlier_mask.sum())}/{len(calibration.inlier_mask)} "
        f"inlier points, reprojection error mean {error.mean_world_mm:.2f} mm, max {error.max_world_mm:.2f} mm "
        f"(mean {error.mean_image_px:.2f} px, max {error.max_image_px:.2f} px)"
    )
    return calibration


def load_model(model_path):
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model weights not found: {model_path}")
    model = YOLO(str(path))
    if model.task != "pose":
        raise ValueError(f"model {model_path} is a '{model.task}' model, a pose model is required")
    return model


def resolve_device(requested_device: str | None):
    if requested_device is not None:
        return requested_device
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def describe_state(state: CarState) -> str:
    prefix = f"t={state.timestamp_us / 1e6:9.3f} s  {state.name:<12}"
    if state.status == TrackStatus.TRACKED:
        center_u, center_v = state.image_points.center_uv
        return (
            f"{prefix} X={state.center_mm[0]:8.1f} mm  Y={state.center_mm[1]:8.1f} mm  "
            f"theta={state.theta_deg:7.1f} deg  "
            f"v=({state.velocity_mm_s[0]:8.1f},{state.velocity_mm_s[1]:8.1f}) mm/s  "
            f"omega={state.angular_velocity_deg_s:8.1f} deg/s  image=({center_u:4.0f},{center_v:4.0f})"
        )
    if state.status == TrackStatus.UNCALIBRATED:
        center_u, center_v = state.image_points.center_uv
        return f"{prefix} detected at image (u={center_u:4.0f}, v={center_v:4.0f})  world: uncalibrated"
    return f"{prefix} {state.status.value.replace('_', ' ')}"


def render_overlay(frame, calibration, tracker, states, fps_estimate, frame_index, paused):
    annotated = frame.copy()
    draw_calibration(annotated, calibration)
    for state in states:
        draw_car_state(annotated, state, calibration, tracker.tracks[state.track_id])
    draw_top_view(annotated, calibration, tracker.tracks, states)
    draw_status(annotated, calibration, fps_estimate, frame_index, paused)
    return annotated


def publish_states(states, udp_socket, target):
    for state in states:
        udp_socket.sendto(format_car_message(state).encode("utf-8"), target)
        print(describe_state(state), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        calibration = load_calibration(args)
    except CalibrationError as error:
        print(f"ERROR: invalid ground plane calibration: {error}", file=sys.stderr)
        print(
            "Create one with toy-car-calibrate / toy-car-calibrate-tiles or start with --uncalibrated.", file=sys.stderr
        )
        return 2

    model = load_model(args.model)
    device = resolve_device(args.device)
    car_names = parse_car_names(model.names, args.car_names)
    tracker = MultiCarTracker(
        car_names,
        calibration,
        smoothing_alpha=args.smoothing,
        max_missed_frames=args.max_missed_frames,
        min_keypoint_confidence=args.min_keypoint_conf,
    )

    capture, is_video_file = open_source(args)
    if not capture.isOpened():
        print(f"ERROR: cannot open video source {args.source!r}", file=sys.stderr)
        return 1
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if calibration is not None:
        try:
            calibration.check_image_size(frame_width, frame_height)
        except CalibrationError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = DEFAULT_SOURCE_FPS

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_writer = None
    if args.save_video:
        video_writer = cv2.VideoWriter(
            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (frame_width, frame_height)
        )

    print(
        f"Global Vision Server: source={args.source} ({frame_width}x{frame_height} @ {source_fps:.1f} fps), "
        f"device={device}, cars={car_names}, UDP -> {args.ip}:{args.port}"
    )
    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start_time = time.perf_counter()
    frame_index = 0
    processed_frames = 0
    loop_offset_us = 0
    paused = False
    step_once = False
    frame = None
    fps_estimate = 0.0
    inference_times_ms = []

    try:
        while True:
            if not paused or step_once or frame is None:
                step_once = False
                ok, frame = capture.read()
                if not ok:
                    if is_video_file and args.loop:
                        loop_offset_us += int(frame_index * 1e6 / source_fps)
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracker.reset()
                        frame_index = 0
                        ok, frame = capture.read()
                    if not ok:
                        break
                # Video files get timestamps from the frame index (reproducible, independent of processing speed);
                # loop_offset_us keeps them increasing when --loop restarts the video. Cameras use the wall clock.
                if is_video_file:
                    timestamp_us = loop_offset_us + int(round(frame_index * 1e6 / source_fps))
                else:
                    timestamp_us = int((time.perf_counter() - start_time) * 1e6)

                frame_start = time.perf_counter()
                results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)
                inference_times_ms.append((time.perf_counter() - frame_start) * 1000.0)
                detections = detections_from_yolo_result(results[0])
                states = tracker.update(frame_index, timestamp_us, detections)

                publish_states(states, udp_socket, (args.ip, args.port))

                frame_time_s = time.perf_counter() - frame_start
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / frame_time_s) if fps_estimate else 1.0 / frame_time_s
                annotated = render_overlay(frame, calibration, tracker, states, fps_estimate, frame_index, paused)
                if video_writer:
                    video_writer.write(annotated)

                frame_index += 1
                processed_frames += 1
                if args.max_frames is not None and processed_frames >= args.max_frames:
                    break

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key == KEY_QUIT:
                    break
                if key == KEY_PAUSE:
                    paused = not paused
                elif key == KEY_STEP and paused:
                    step_once = True
    finally:
        capture.release()
        udp_socket.close()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if processed_frames == 0:
        print(f"ERROR: no frame could be read from video source {args.source!r}", file=sys.stderr)
        return 1
    if inference_times_ms:
        timings = np.array(inference_times_ms[1:] or inference_times_ms)
        print(
            f"Processed {processed_frames} frames, inference mean {timings.mean():.1f} ms, "
            f"p95 {np.percentile(timings, 95):.1f} ms per frame"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

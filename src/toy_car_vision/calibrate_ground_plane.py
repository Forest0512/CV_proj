import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from toy_car_vision.ground_plane import CalibrationError, GroundPlaneCalibration
from toy_car_vision.paths import DEFAULT_CALIBRATION_PATH, resolve_input_path

WINDOW_NAME = "Ground plane calibration"
KEY_ENTER = (13, 10)
KEY_ESCAPE = 27
KEY_UNDO = ord("u")


def parse_point_list(text: str) -> np.ndarray:
    points = []
    for entry in text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        values = [float(value) for value in entry.split(",")]
        if len(values) != 2:
            raise ValueError(f"point {entry!r} must have exactly two values")
        points.append(values)
    return np.array(points, dtype=np.float64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the image-to-ground homography from measured floor reference points"
    )
    parser.add_argument("--source", required=True, help="camera index or video/image file")
    parser.add_argument("--frame", type=int, default=0, help="frame index to use from a video file")
    parser.add_argument(
        "--world-points", required=True, help='measured floor coordinates in mm, in click order: "X1,Y1;X2,Y2;..."'
    )
    parser.add_argument("--ransac", action="store_true", help="use RANSAC (needs more than 4 points)")
    parser.add_argument("--output", default=str(DEFAULT_CALIBRATION_PATH))
    return parser.parse_args(argv)


def grab_frame(source: str, frame_index: int) -> np.ndarray:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        source = str(resolve_input_path(source))
        image = cv2.imread(source)
        if image is not None:
            return image
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise FileNotFoundError(f"cannot read a frame from {source!r}")
    return frame


def click_image_points(frame: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    clicked = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(world_points):
            clicked.append([float(x), float(y)])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    while True:
        canvas = frame.copy()
        for index, point in enumerate(clicked):
            cv2.drawMarker(canvas, (int(point[0]), int(point[1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.putText(
                canvas,
                str(index),
                (int(point[0]) + 5, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        if len(clicked) < len(world_points):
            target = world_points[len(clicked)]
            prompt = f"click point {len(clicked)} = ({target[0]:.0f}, {target[1]:.0f}) mm   [u]ndo  [esc] abort"
        else:
            prompt = "all points set: [enter] compute   [u]ndo   [esc] abort"
        cv2.putText(canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == KEY_ESCAPE:
            cv2.destroyWindow(WINDOW_NAME)
            raise KeyboardInterrupt("calibration aborted")
        if key == KEY_UNDO and clicked:
            clicked.pop()
        if key in KEY_ENTER and len(clicked) == len(world_points):
            cv2.destroyWindow(WINDOW_NAME)
            return np.array(clicked, dtype=np.float64)


def print_report(calibration: GroundPlaneCalibration) -> None:
    world_from_image, _ = calibration.project_to_world(calibration.image_points_px)
    print(f"{'#':>2} {'u':>8} {'v':>8} {'X_meas':>9} {'Y_meas':>9} {'X_proj':>9} {'Y_proj':>9} {'err_mm':>7} inlier")
    for index, (image_point, world_point, projected, inlier) in enumerate(
        zip(calibration.image_points_px, calibration.world_points_mm, world_from_image, calibration.inlier_mask)
    ):
        error = float(np.linalg.norm(projected - world_point))
        print(
            f"{index:>2} {image_point[0]:8.1f} {image_point[1]:8.1f} {world_point[0]:9.1f} {world_point[1]:9.1f} "
            f"{projected[0]:9.1f} {projected[1]:9.1f} {error:7.2f} {bool(inlier)}"
        )
    error = calibration.reprojection_error
    print(
        f"reprojection error (inliers): mean {error.mean_world_mm:.2f} mm, max {error.max_world_mm:.2f} mm, "
        f"mean {error.mean_image_px:.2f} px, max {error.max_image_px:.2f} px"
    )
    if len(calibration.image_points_px) == 4 and not calibration.use_ransac:
        print(
            "NOTE: with exactly 4 points the homography is fit exactly, so the reprojection error is ~0 "
            "and says nothing about accuracy. Use more points or validate with extra check points."
        )


def show_overlay(frame: np.ndarray, calibration: GroundPlaneCalibration) -> None:
    canvas = frame.copy()
    projected, valid = calibration.project_to_image(calibration.world_points_mm)
    for measured, reprojected, is_valid in zip(calibration.image_points_px, projected, valid):
        cv2.drawMarker(canvas, (int(measured[0]), int(measured[1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
        if is_valid:
            cv2.circle(canvas, (int(round(reprojected[0])), int(round(reprojected[1]))), 6, (0, 255, 0), 1)
    world_min = calibration.world_points_mm.min(axis=0)
    world_max = calibration.world_points_mm.max(axis=0)
    grid_step = 10 ** np.floor(np.log10(max(float((world_max - world_min).max()), 1.0) / 2))
    for x in np.arange(np.floor(world_min[0] / grid_step) * grid_step, world_max[0] + grid_step, grid_step):
        line, line_valid = calibration.project_to_image(np.array([[x, world_min[1]], [x, world_max[1]]]))
        if line_valid.all():
            cv2.line(canvas, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (255, 200, 0), 1)
    for y in np.arange(np.floor(world_min[1] / grid_step) * grid_step, world_max[1] + grid_step, grid_step):
        line, line_valid = calibration.project_to_image(np.array([[world_min[0], y], [world_max[0], y]]))
        if line_valid.all():
            cv2.line(canvas, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (255, 200, 0), 1)
    cv2.putText(
        canvas,
        f"projected world grid, {grid_step:.0f} mm spacing - press any key",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2,
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    world_points = parse_point_list(args.world_points)
    frame = grab_frame(args.source, args.frame)
    height, width = frame.shape[:2]

    try:
        image_points = click_image_points(frame, world_points)
    except KeyboardInterrupt as error:
        print(error, file=sys.stderr)
        return 1

    try:
        calibration = GroundPlaneCalibration.from_correspondences(
            image_points, world_points, image_size=(width, height), use_ransac=args.ransac
        )
    except CalibrationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print_report(calibration)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(output_path)
    print(f"saved calibration to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import cv2
import numpy as np

from toy_car_vision.car_tracking import CarState, CarTrack, TrackStatus, bbox_bottom_center
from toy_car_vision.ground_plane import GroundPlaneCalibration

FONT = cv2.FONT_HERSHEY_SIMPLEX
LOST_COLOR = (128, 128, 128)
CALIBRATION_POINT_COLOR = (0, 0, 255)
CALIBRATED_AREA_COLOR = (0, 200, 255)
STATUS_OK_COLOR = (0, 200, 0)
STATUS_WARNING_COLOR = (0, 0, 255)
TRACK_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]
TOP_VIEW_SIZE_PX = 260
TOP_VIEW_MARGIN_PX = 12


def track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def _point(uv) -> tuple[int, int]:
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def _put_text(image: np.ndarray, text: str, origin, color, scale: float = 0.5, thickness: int = 1) -> None:
    cv2.putText(image, text, origin, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_calibration(image: np.ndarray, calibration: GroundPlaneCalibration | None) -> None:
    if calibration is None:
        return
    area_px, valid = calibration.project_to_image(calibration.calibrated_area_mm)
    if valid.all():
        cv2.polylines(image, [area_px.astype(np.int32).reshape(-1, 1, 2)], True, CALIBRATED_AREA_COLOR, 1, cv2.LINE_AA)
    for image_point, world_point, inlier in zip(
        calibration.image_points_px, calibration.world_points_mm, calibration.inlier_mask
    ):
        color = CALIBRATION_POINT_COLOR if inlier else LOST_COLOR
        cv2.drawMarker(image, _point(image_point), color, cv2.MARKER_CROSS, 12, 1)
        _put_text(image, f"({world_point[0]:.0f},{world_point[1]:.0f})", _point(image_point + 6), color, 0.35)


def draw_car_state(
    image: np.ndarray, state: CarState, calibration: GroundPlaneCalibration | None, track: CarTrack
) -> None:
    color = track_color(state.track_id)
    detection = state.detection
    if detection is None:
        return
    box_color = color if state.status == TrackStatus.TRACKED else LOST_COLOR
    x1, y1, x2, y2 = (int(value) for value in detection.box_xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    _put_text(
        image,
        f"id={state.track_id} {state.name} {detection.confidence:.2f}",
        (x1, max(y1 - 8, 14)),
        box_color,
    )
    cv2.drawMarker(image, _point(bbox_bottom_center(detection.box_xyxy)), LOST_COLOR, cv2.MARKER_TILTED_CROSS, 8, 1)

    if state.image_points is not None:
        for keypoint in detection.keypoints_xy[:3]:
            cv2.circle(image, _point(keypoint), 3, (255, 255, 0), -1)
        cv2.arrowedLine(
            image,
            _point(state.image_points.tail_uv),
            _point(state.image_points.front_uv),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.3,
        )

    if state.status == TrackStatus.TRACKED:
        ground_point_px, valid = calibration.project_to_image(state.center_mm)
        if valid[0]:
            cv2.circle(image, _point(ground_point_px[0]), 5, (0, 255, 255), -1)
        if len(track.trajectory_mm) > 1:
            trajectory_px, trajectory_valid = calibration.project_to_image(np.array(track.trajectory_mm))
            trajectory_px = trajectory_px[trajectory_valid]
            cv2.polylines(image, [trajectory_px.astype(np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        area_note = "" if state.raw_pose.inside_calibrated_area else " OUTSIDE CALIB AREA"
        lines = [
            f"X={state.center_mm[0]:.0f} mm Y={state.center_mm[1]:.0f} mm{area_note}",
            f"theta={state.theta_deg:.1f} deg  v=({state.velocity_mm_s[0]:.0f},{state.velocity_mm_s[1]:.0f}) mm/s",
        ]
    else:
        lines = [state.status.value]
    for line_index, line in enumerate(lines):
        _put_text(image, line, (x1, y2 + 16 + 16 * line_index), box_color)


def draw_top_view(
    image: np.ndarray,
    calibration: GroundPlaneCalibration | None,
    tracks: dict[int, CarTrack],
    states: list[CarState],
) -> None:
    if calibration is None:
        return
    world_points = calibration.world_points_mm
    trajectories = [np.array(track.trajectory_mm) for track in tracks.values() if len(track.trajectory_mm)]
    all_points = np.vstack([world_points, *trajectories]) if trajectories else world_points
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    scale = (TOP_VIEW_SIZE_PX - 2 * TOP_VIEW_MARGIN_PX) / float(extent.max())
    height, width = image.shape[:2]
    view_width = int(extent[0] * scale) + 2 * TOP_VIEW_MARGIN_PX
    view_height = int(extent[1] * scale) + 2 * TOP_VIEW_MARGIN_PX
    if view_width >= width or view_height >= height:
        return
    origin_x = width - view_width - 10
    origin_y = height - view_height - 10

    def to_view(world_xy):
        offset = (np.asarray(world_xy) - minimum) * scale
        return (
            int(origin_x + TOP_VIEW_MARGIN_PX + offset[0]),
            int(origin_y + view_height - TOP_VIEW_MARGIN_PX - offset[1]),
        )

    overlay = image.copy()
    cv2.rectangle(overlay, (origin_x, origin_y), (origin_x + view_width, origin_y + view_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    area = np.array([to_view(point) for point in calibration.calibrated_area_mm], dtype=np.int32)
    cv2.polylines(image, [area.reshape(-1, 1, 2)], True, CALIBRATED_AREA_COLOR, 1, cv2.LINE_AA)
    for world_point in world_points:
        cv2.drawMarker(image, to_view(world_point), CALIBRATION_POINT_COLOR, cv2.MARKER_CROSS, 8, 1)
    _put_text(image, "top view (X right, Y up)", (origin_x + 4, origin_y + 14), (220, 220, 220), 0.4)
    for track in tracks.values():
        if len(track.trajectory_mm) > 1:
            points = np.array([to_view(point) for point in track.trajectory_mm], dtype=np.int32)
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False, track_color(track.track_id), 1, cv2.LINE_AA)
    for state in states:
        if state.status == TrackStatus.TRACKED:
            center = to_view(state.center_mm)
            heading = np.radians(state.theta_deg)
            tip = (int(center[0] + 15 * np.cos(heading)), int(center[1] - 15 * np.sin(heading)))
            cv2.circle(image, center, 4, track_color(state.track_id), -1)
            cv2.arrowedLine(image, center, tip, track_color(state.track_id), 1, cv2.LINE_AA, tipLength=0.4)


def draw_status(
    image: np.ndarray, calibration: GroundPlaneCalibration | None, fps: float, frame_index: int, paused: bool
):
    if calibration is None:
        calibration_text = "UNCALIBRATED - no world coordinates"
        calibration_color = STATUS_WARNING_COLOR
    else:
        error = calibration.reprojection_error
        calibration_text = (
            f"CALIBRATED: {int(calibration.inlier_mask.sum())} pts, "
            f"reproj mean {error.mean_world_mm:.1f} mm / max {error.max_world_mm:.1f} mm"
        )
        calibration_color = STATUS_OK_COLOR
    run_text = "PAUSED" if paused else f"FPS {fps:.1f}"
    _put_text(image, f"frame {frame_index}  {run_text}", (10, 22), STATUS_OK_COLOR, 0.6, 2)
    _put_text(image, calibration_text, (10, 44), calibration_color, 0.5, 1)

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from toy_car_vision.ground_plane import GroundPlaneCalibration

HEAD_LEFT_INDEX = 0
HEAD_RIGHT_INDEX = 1
TAIL_INDEX = 2
CAR_KEYPOINT_COUNT = 3
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.5
DEFAULT_MAX_MISSED_FRAMES = 12
DEFAULT_TRAJECTORY_LENGTH = 300


class TrackStatus(str, Enum):
    TRACKED = "tracked"
    NOT_DETECTED = "not_detected"
    KEYPOINTS_INVALID = "keypoints_invalid"
    PROJECTION_INVALID = "projection_invalid"
    UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class CarDetection:
    class_id: int
    confidence: float
    box_xyxy: np.ndarray
    keypoints_xy: np.ndarray
    keypoint_confidences: np.ndarray | None = None


@dataclass(frozen=True)
class ImageCarPoints:
    front_uv: np.ndarray
    tail_uv: np.ndarray

    @property
    def center_uv(self) -> np.ndarray:
        return (self.front_uv + self.tail_uv) / 2.0


@dataclass(frozen=True)
class WorldCarPose:
    front_mm: np.ndarray
    tail_mm: np.ndarray
    inside_calibrated_area: bool

    @property
    def center_mm(self) -> np.ndarray:
        return (self.front_mm + self.tail_mm) / 2.0

    @property
    def theta_deg(self) -> float:
        heading = self.front_mm - self.tail_mm
        return float(np.degrees(np.arctan2(heading[1], heading[0])))


def detections_from_yolo_result(result) -> list[CarDetection]:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    keypoint_confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    return [
        CarDetection(
            class_id=int(classes[index]),
            confidence=float(confidences[index]),
            box_xyxy=boxes[index],
            keypoints_xy=keypoints[index],
            keypoint_confidences=None if keypoint_confidences is None else keypoint_confidences[index],
        )
        for index in range(min(len(boxes), len(keypoints)))
    ]


def best_detection_per_class(detections: list[CarDetection]) -> dict[int, CarDetection]:
    best: dict[int, CarDetection] = {}
    for detection in detections:
        current = best.get(detection.class_id)
        if current is None or detection.confidence > current.confidence:
            best[detection.class_id] = detection
    return best


def car_points_from_keypoints(
    detection: CarDetection, min_keypoint_confidence: float = DEFAULT_MIN_KEYPOINT_CONFIDENCE
) -> ImageCarPoints | None:
    keypoints = np.asarray(detection.keypoints_xy, dtype=np.float64)
    if keypoints.shape[0] < CAR_KEYPOINT_COUNT or keypoints.shape[1] != 2:
        return None
    keypoints = keypoints[:CAR_KEYPOINT_COUNT]
    # Ultralytics reports keypoints it could not locate as (0, 0).
    if not np.all(np.isfinite(keypoints)) or np.any(np.all(keypoints <= 0.0, axis=1)):
        return None
    if detection.keypoint_confidences is not None:
        confidences = np.asarray(detection.keypoint_confidences)[:CAR_KEYPOINT_COUNT]
        if np.any(confidences < min_keypoint_confidence):
            return None
    front = (keypoints[HEAD_LEFT_INDEX] + keypoints[HEAD_RIGHT_INDEX]) / 2.0
    tail = keypoints[TAIL_INDEX]
    if np.allclose(front, tail):
        return None
    return ImageCarPoints(front_uv=front, tail_uv=tail)


def bbox_bottom_center(box_xyxy) -> np.ndarray:
    x1, _, x2, y2 = (float(value) for value in box_xyxy)
    return np.array([(x1 + x2) / 2.0, y2])


def world_pose_from_image_points(
    image_points: ImageCarPoints, calibration: GroundPlaneCalibration
) -> WorldCarPose | None:
    # Front and tail are projected separately and averaged on the floor: because of perspective,
    # the projection of the image midpoint is not the midpoint of the car on the floor.
    world_points, valid = calibration.project_to_world(np.vstack([image_points.front_uv, image_points.tail_uv]))
    if not valid.all():
        return None
    front_mm, tail_mm = world_points
    center_mm = (front_mm + tail_mm) / 2.0
    return WorldCarPose(
        front_mm=front_mm,
        tail_mm=tail_mm,
        inside_calibrated_area=calibration.is_inside_calibrated_area(center_mm),
    )


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class CarState:
    track_id: int
    name: str
    timestamp_us: int
    frame_index: int
    status: TrackStatus
    detection: CarDetection | None = None
    image_points: ImageCarPoints | None = None
    raw_pose: WorldCarPose | None = None
    center_mm: np.ndarray | None = None
    theta_deg: float | None = None
    velocity_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angular_velocity_deg_s: float = 0.0

    @property
    def has_world_position(self) -> bool:
        return self.status == TrackStatus.TRACKED


class CarTrack:
    def __init__(
        self,
        track_id: int,
        name: str,
        smoothing_alpha: float = 1.0,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH,
    ):
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.track_id = track_id
        self.name = name
        self.smoothing_alpha = smoothing_alpha
        self.max_missed_frames = max_missed_frames
        self.trajectory_mm: deque[np.ndarray] = deque(maxlen=trajectory_length)
        self.reset()

    def reset(self) -> None:
        self.center_mm: np.ndarray | None = None
        self.theta_deg: float | None = None
        self.last_valid_timestamp_us: int | None = None
        self.missed_frames = 0
        self.trajectory_mm.clear()

    def update(
        self,
        frame_index: int,
        timestamp_us: int,
        detection: CarDetection | None,
        image_points: ImageCarPoints | None,
        raw_pose: WorldCarPose | None,
        calibrated: bool = True,
    ) -> CarState:
        if detection is None:
            status = TrackStatus.NOT_DETECTED
        elif image_points is None:
            status = TrackStatus.KEYPOINTS_INVALID
        elif not calibrated:
            status = TrackStatus.UNCALIBRATED
        elif raw_pose is None:
            status = TrackStatus.PROJECTION_INVALID
        else:
            status = TrackStatus.TRACKED

        if status != TrackStatus.TRACKED:
            self.missed_frames += 1
            # After a long gap the old pose is too stale for finite differences; start velocities from zero.
            if self.missed_frames > self.max_missed_frames:
                self.center_mm = None
                self.theta_deg = None
                self.last_valid_timestamp_us = None
            return CarState(
                track_id=self.track_id,
                name=self.name,
                timestamp_us=timestamp_us,
                frame_index=frame_index,
                status=status,
                detection=detection,
                image_points=image_points,
            )

        raw_center = raw_pose.center_mm
        raw_theta = raw_pose.theta_deg
        velocity = np.zeros(2)
        angular_velocity = 0.0

        if self.center_mm is None:
            new_center = raw_center
            new_theta = raw_theta
        else:
            alpha = self.smoothing_alpha
            new_center = alpha * raw_center + (1.0 - alpha) * self.center_mm
            # Smooth along the shortest angular difference so that e.g. 179 deg -> -179 deg is a 2 deg step.
            new_theta = wrap_angle_deg(self.theta_deg + alpha * wrap_angle_deg(raw_theta - self.theta_deg))
            elapsed_s = (timestamp_us - self.last_valid_timestamp_us) / 1e6
            if elapsed_s > 0.0:
                velocity = (new_center - self.center_mm) / elapsed_s
                angular_velocity = wrap_angle_deg(new_theta - self.theta_deg) / elapsed_s

        self.center_mm = new_center
        self.theta_deg = new_theta
        self.last_valid_timestamp_us = timestamp_us
        self.missed_frames = 0
        self.trajectory_mm.append(new_center.copy())
        return CarState(
            track_id=self.track_id,
            name=self.name,
            timestamp_us=timestamp_us,
            frame_index=frame_index,
            status=status,
            detection=detection,
            image_points=image_points,
            raw_pose=raw_pose,
            center_mm=new_center,
            theta_deg=new_theta,
            velocity_mm_s=velocity,
            angular_velocity_deg_s=angular_velocity,
        )


class MultiCarTracker:
    def __init__(
        self,
        car_names: dict[int, str],
        calibration: GroundPlaneCalibration | None,
        smoothing_alpha: float = 1.0,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        min_keypoint_confidence: float = DEFAULT_MIN_KEYPOINT_CONFIDENCE,
    ):
        self.calibration = calibration
        self.min_keypoint_confidence = min_keypoint_confidence
        self.tracks = {
            class_id: CarTrack(class_id, name, smoothing_alpha, max_missed_frames)
            for class_id, name in car_names.items()
        }

    def reset(self) -> None:
        for track in self.tracks.values():
            track.reset()

    def update(self, frame_index: int, timestamp_us: int, detections: list[CarDetection]) -> list[CarState]:
        best_detections = best_detection_per_class(detections)
        states = []
        for class_id, track in self.tracks.items():
            detection = best_detections.get(class_id)
            image_points = (
                car_points_from_keypoints(detection, self.min_keypoint_confidence) if detection is not None else None
            )
            raw_pose = (
                world_pose_from_image_points(image_points, self.calibration)
                if image_points is not None and self.calibration is not None
                else None
            )
            states.append(
                track.update(
                    frame_index,
                    timestamp_us,
                    detection,
                    image_points,
                    raw_pose,
                    calibrated=self.calibration is not None,
                )
            )
        return states

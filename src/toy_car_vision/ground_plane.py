import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_CORRESPONDENCES = 4
MIN_POLYGON_AREA_PX = 100.0
MIN_POLYGON_AREA_MM = 100.0
MAX_HOMOGRAPHY_CONDITION_NUMBER = 1e12
HOMOGENEOUS_SCALE_EPSILON = 1e-9
DEFAULT_RANSAC_THRESHOLD_MM = 20.0


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class ReprojectionError:
    mean_world_mm: float
    max_world_mm: float
    mean_image_px: float
    max_image_px: float


@dataclass(frozen=True)
class GroundPlaneCalibration:
    image_points_px: np.ndarray
    world_points_mm: np.ndarray
    image_to_world: np.ndarray
    world_to_image: np.ndarray
    inlier_mask: np.ndarray
    reprojection_error: ReprojectionError
    valid_scale_sign: float
    calibrated_area_mm: np.ndarray
    image_size: tuple[int, int] | None
    use_ransac: bool

    @classmethod
    def from_correspondences(
        cls,
        image_points_px,
        world_points_mm,
        image_size=None,
        use_ransac: bool = False,
        ransac_threshold_mm: float = DEFAULT_RANSAC_THRESHOLD_MM,
    ) -> "GroundPlaneCalibration":
        image_points, world_points = _validate_correspondences(image_points_px, world_points_mm)
        _check_not_degenerate(image_points, world_points)

        homography, inlier_mask = _estimate_homography(image_points, world_points, use_ransac, ransac_threshold_mm)
        inverse_homography = _normalized_inverse(homography)
        valid_scale_sign = _floor_scale_sign(homography, image_points, inlier_mask)

        return cls(
            image_points_px=image_points,
            world_points_mm=world_points,
            image_to_world=homography,
            world_to_image=inverse_homography,
            inlier_mask=inlier_mask,
            reprojection_error=_reprojection_error(
                homography, inverse_homography, image_points, world_points, inlier_mask
            ),
            valid_scale_sign=valid_scale_sign,
            calibrated_area_mm=_calibrated_area(world_points, inlier_mask),
            image_size=tuple(int(value) for value in image_size) if image_size else None,
            use_ransac=bool(use_ransac),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GroundPlaneCalibration":
        config = _read_config(path)
        _validate_config(config)
        return cls.from_correspondences(
            image_points_px=config["image_points_px"],
            world_points_mm=config["world_points_mm"],
            image_size=config.get("image_size"),
            use_ransac=bool(config.get("use_ransac", False)),
            ransac_threshold_mm=float(config.get("ransac_threshold_mm", DEFAULT_RANSAC_THRESHOLD_MM)),
        )

    def to_config(self) -> dict:
        return {
            "units": "mm",
            "image_size": list(self.image_size) if self.image_size else None,
            "image_points_px": self.image_points_px.tolist(),
            "world_points_mm": self.world_points_mm.tolist(),
            "use_ransac": self.use_ransac,
            "image_to_world_homography": self.image_to_world.tolist(),
            "reprojection_error": {
                "mean_world_mm": self.reprojection_error.mean_world_mm,
                "max_world_mm": self.reprojection_error.max_world_mm,
                "mean_image_px": self.reprojection_error.mean_image_px,
                "max_image_px": self.reprojection_error.max_image_px,
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_config(), indent=2), encoding="utf-8")

    def check_image_size(self, width: int, height: int) -> None:
        if self.image_size is not None and tuple(self.image_size) != (int(width), int(height)):
            raise CalibrationError(
                f"calibration was made for image size {self.image_size}, "
                f"but the video source delivers {(int(width), int(height))}"
            )

    def project_to_world(self, image_points_px) -> tuple[np.ndarray, np.ndarray]:
        image_points = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
        world_points = np.full_like(image_points, np.nan)
        finite = np.all(np.isfinite(image_points), axis=1)
        if not finite.any():
            return world_points, finite
        projected, scales = _apply_homography(self.image_to_world, image_points[finite])
        # Pixels with W of the wrong sign (or W ~ 0) lie on or above the horizon and have no floor point.
        in_front_of_horizon = (np.abs(scales) > HOMOGENEOUS_SCALE_EPSILON) & (np.sign(scales) == self.valid_scale_sign)
        valid = np.zeros(len(image_points), dtype=bool)
        valid[finite] = in_front_of_horizon
        world_points[valid] = projected[in_front_of_horizon] / scales[in_front_of_horizon, None]
        return world_points, valid

    def project_to_image(self, world_points_mm) -> tuple[np.ndarray, np.ndarray]:
        world_points = np.asarray(world_points_mm, dtype=np.float64).reshape(-1, 2)
        projected, scales = _apply_homography(self.world_to_image, world_points)
        valid = np.all(np.isfinite(world_points), axis=1) & (np.abs(scales) > HOMOGENEOUS_SCALE_EPSILON)
        image_points = np.full_like(world_points, np.nan)
        image_points[valid] = projected[valid] / scales[valid, None]
        return image_points, valid

    def is_inside_calibrated_area(self, world_point_mm) -> bool:
        x, y = (float(value) for value in world_point_mm)
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        signed_distance = cv2.pointPolygonTest(
            self.calibrated_area_mm.astype(np.float32).reshape(-1, 1, 2), (x, y), True
        )
        return signed_distance >= 0


def _read_config(path: str | Path) -> dict:
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CalibrationError(f"calibration file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"calibration file is not valid JSON: {path}: {error}") from error
    if not isinstance(config, dict):
        raise CalibrationError("calibration file must contain a JSON object")
    return config


def _validate_config(config: dict) -> None:
    missing_keys = {"image_points_px", "world_points_mm"} - config.keys()
    if missing_keys:
        raise CalibrationError(f"calibration file is missing keys: {sorted(missing_keys)}")
    empty_keys = [key for key in ("image_points_px", "world_points_mm") if not config[key]]
    if empty_keys:
        raise CalibrationError(f"{', '.join(empty_keys)} not filled in: enter the measured reference points first")
    if config.get("units", "mm") != "mm":
        raise CalibrationError("only 'mm' is supported as world unit")


def _validate_correspondences(image_points_px, world_points_mm) -> tuple[np.ndarray, np.ndarray]:
    image_points = _as_point_array(image_points_px, "image_points_px")
    world_points = _as_point_array(world_points_mm, "world_points_mm")
    if len(image_points) != len(world_points):
        raise CalibrationError(f"{len(image_points)} image points but {len(world_points)} world points")
    if len(image_points) < MIN_CORRESPONDENCES:
        raise CalibrationError(
            f"at least {MIN_CORRESPONDENCES} point correspondences are required, got {len(image_points)}"
        )
    return image_points, world_points


def _check_not_degenerate(image_points: np.ndarray, world_points: np.ndarray) -> None:
    if _convex_hull_area(image_points) < MIN_POLYGON_AREA_PX:
        raise CalibrationError("image points are (nearly) collinear or coincident")
    if _convex_hull_area(world_points) < MIN_POLYGON_AREA_MM:
        raise CalibrationError("world points are (nearly) collinear or coincident")


def _estimate_homography(
    image_points: np.ndarray, world_points: np.ndarray, use_ransac: bool, ransac_threshold_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    method = cv2.RANSAC if use_ransac and len(image_points) > MIN_CORRESPONDENCES else 0
    homography, mask = cv2.findHomography(image_points, world_points, method, ransac_threshold_mm)
    if homography is None or not np.all(np.isfinite(homography)):
        raise CalibrationError("cv2.findHomography could not estimate a homography")
    if np.linalg.cond(homography) > MAX_HOMOGRAPHY_CONDITION_NUMBER:
        raise CalibrationError("estimated homography is singular or ill-conditioned")
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(image_points), dtype=bool)
    if inlier_mask.sum() < MIN_CORRESPONDENCES:
        raise CalibrationError("fewer than 4 inlier correspondences after RANSAC")
    return homography / homography[2, 2], inlier_mask


def _normalized_inverse(homography: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(homography)
    return inverse / inverse[2, 2]


def _floor_scale_sign(homography: np.ndarray, image_points: np.ndarray, inlier_mask: np.ndarray) -> float:
    _, scales = _apply_homography(homography, image_points)
    if np.any(np.abs(scales) < HOMOGENEOUS_SCALE_EPSILON):
        raise CalibrationError("a calibration point maps to the line at infinity")
    # The homogeneous scale W changes sign at the horizon of the floor plane. All real floor
    # points share one sign; it is stored so that pixels above the horizon can be rejected later.
    inlier_scale_signs = np.sign(scales[inlier_mask])
    if not np.all(inlier_scale_signs == inlier_scale_signs[0]):
        raise CalibrationError("calibration points lie on both sides of the horizon; check the point order")
    return float(inlier_scale_signs[0])


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected, scales = _apply_homography(homography, points)
    return projected / scales[:, None]


def _reprojection_error(
    homography: np.ndarray,
    inverse_homography: np.ndarray,
    image_points: np.ndarray,
    world_points: np.ndarray,
    inlier_mask: np.ndarray,
) -> ReprojectionError:
    world_errors = np.linalg.norm(_project(homography, image_points) - world_points, axis=1)[inlier_mask]
    image_errors = np.linalg.norm(_project(inverse_homography, world_points) - image_points, axis=1)[inlier_mask]
    return ReprojectionError(
        mean_world_mm=float(world_errors.mean()),
        max_world_mm=float(world_errors.max()),
        mean_image_px=float(image_errors.mean()),
        max_image_px=float(image_errors.max()),
    )


def _calibrated_area(world_points: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
    return cv2.convexHull(world_points[inlier_mask].astype(np.float32)).reshape(-1, 2)


def _as_point_array(points, name: str) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must be a list of [x, y] pairs") from error
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} contains NaN or infinite values")
    return array


def _convex_hull_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous_points @ homography.T
    return transformed[:, :2], transformed[:, 2]

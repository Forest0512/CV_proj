import math
from dataclasses import dataclass

from toy_car_vision.car_tracking import CarState, TrackStatus

NO_CAR_VALUE = -1000.0
NO_CAR_PIXEL = -1
FIELD_COUNT = 9


@dataclass(frozen=True)
class CarMessage:
    timestamp_us: int
    name: str
    x_mm: float
    y_mm: float
    theta_deg: float
    vx_mm_s: float
    vy_mm_s: float
    omega_deg_s: float
    u_px: int
    v_px: int

    @property
    def car_detected(self) -> bool:
        return self.x_mm != NO_CAR_VALUE

    @property
    def has_world_position(self) -> bool:
        return self.car_detected and not math.isnan(self.x_mm)


def format_car_message(state: CarState) -> str:
    if state.status == TrackStatus.TRACKED:
        center_x, center_y = state.center_mm
        velocity_x, velocity_y = state.velocity_mm_s
        center_u, center_v = (int(round(value)) for value in state.image_points.center_uv)
        return (
            f'{state.timestamp_us}:"{state.name}",'
            f"{center_x:.1f},{center_y:.1f},{state.theta_deg:.1f},"
            f"{velocity_x:.1f},{velocity_y:.1f},{state.angular_velocity_deg_s:.1f},"
            f"{center_u},{center_v}\n"
        )
    if state.status == TrackStatus.UNCALIBRATED:
        center_u, center_v = (int(round(value)) for value in state.image_points.center_uv)
        return f'{state.timestamp_us}:"{state.name}",nan,nan,nan,nan,nan,nan,{center_u},{center_v}\n'
    return (
        f'{state.timestamp_us}:"{state.name}",'
        f"{NO_CAR_VALUE:.1f},{NO_CAR_VALUE:.1f},{NO_CAR_VALUE:.1f},"
        f"0.0,0.0,0.0,{NO_CAR_PIXEL},{NO_CAR_PIXEL}\n"
    )


def parse_car_message(message: str) -> CarMessage | None:
    timestamp, separator, payload = message.rstrip("\n").partition(":")
    fields = payload.split(",")
    if not separator or len(fields) != FIELD_COUNT:
        return None
    try:
        return CarMessage(
            timestamp_us=int(timestamp),
            name=fields[0].strip('"'),
            x_mm=float(fields[1]),
            y_mm=float(fields[2]),
            theta_deg=float(fields[3]),
            vx_mm_s=float(fields[4]),
            vy_mm_s=float(fields[5]),
            omega_deg_s=float(fields[6]),
            u_px=int(fields[7]),
            v_px=int(fields[8]),
        )
    except ValueError:
        return None

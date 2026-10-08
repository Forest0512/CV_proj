from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"

DEFAULT_MODEL_PATH = MODELS_DIR / "v6_model" / "weights" / "best.pt"
DEFAULT_CALIBRATION_PATH = CALIBRATION_DIR / "lab_floor.json"


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate

"""Central configuration for PCB Defect Inspector.

All thresholds, paths and constants live here.
Weights path is configurable via MODEL_PATH (default models/best.pt).
"""

import os
from pathlib import Path

# --- Base directories ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent
APP_DIR: Path = Path(__file__).resolve().parent

# Upload storage (spec: uploads/ folder for original + annotated)
UPLOAD_DIR: Path = BASE_DIR / "uploads"
ORIGINAL_DIR: Path = UPLOAD_DIR / "original"
ANNOTATED_DIR: Path = UPLOAD_DIR / "annotated"
REPORT_DIR: Path = UPLOAD_DIR / "reports"

# SQLite database (users + inspections)
DB_PATH: Path = BASE_DIR / "pcb_inspector.db"

# --- Model weights (configurable) ---
# Default per spec: models/best.pt
# Falls back to legacy models/pcb_yolo26.pt if best.pt is missing.
MODEL_PATH: Path = BASE_DIR / "models" / "best.pt"
LEGACY_MODEL_PATH: Path = BASE_DIR / "models" / "pcb_yolo26.pt"
FALLBACK_MODEL: str = "yolo26n.pt"

# --- Defect classes (fixed order) ---
DEFECT_CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# --- Upload validation (spec: jpg/jpeg/png only, 20 MB max) ---
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB: float = 20.0
MAX_FILE_SIZE_BYTES: int = int(MAX_FILE_SIZE_MB * 1024 * 1024)

# --- Image quality check (OpenCV) ---
BLUR_THRESHOLD: float = 100.0  # Laplacian variance below this = blurry
BRIGHTNESS_MIN: float = 40.0  # mean gray below this = too dark
BRIGHTNESS_MAX: float = 215.0  # mean gray above this = too bright
MIN_WIDTH: int = 256
MIN_HEIGHT: int = 256

# --- PCB verification (own module, heuristic, OpenCV-only) ---
PCB_GREEN_HSV_LOWER: tuple[int, int, int] = (35, 40, 40)
PCB_GREEN_HSV_UPPER: tuple[int, int, int] = (90, 255, 255)
PCB_MIN_GREEN_RATIO: float = 0.05
PCB_MIN_EDGE_DENSITY: float = 0.01  # Canny edge pixel ratio
PCB_MIN_DOMINANT_RATIO: float = 0.35  # dominant colour share of frame
PCB_MIN_DOMINANT_SATURATION: int = 40  # must be chromatic, not grey
PCB_DOMINANT_BINS: int = 16  # per-channel quantisation

# --- Detection ---
CONF_THRESHOLD: float = 0.25  # default slider value
CONF_MIN: float = 0.05
CONF_MAX: float = 0.95
IOU_THRESHOLD: float = 0.45  # NMS IoU
DEVICE: str = "cpu"
IMAGE_SIZE: int = 640

# --- SAHI mode: 4 slices (2x2 grid) with small overlap ---
SAHI_NUM_SLICES: int = 4
SAHI_SLICE_ROWS: int = 2
SAHI_SLICE_COLS: int = 2
SAHI_OVERLAP_RATIO: float = 0.1  # 10% overlap between slices
SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
SAHI_OVERLAP_WIDTH_RATIO: float = 0.2
SAHI_POSTPROCESS_TYPE: str = "GREEDYNMM"
SAHI_POSTPROCESS_MATCH_METRIC: str = "IOS"
SAHI_POSTPROCESS_MATCH_THRESHOLD: float = 0.5

# --- Annotation colours (BGR) per class ---
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "missing_hole": (0, 0, 255),      # red
    "mouse_bite": (0, 165, 255),      # orange
    "open_circuit": (0, 255, 255),    # yellow
    "short": (255, 0, 0),             # blue
    "spur": (255, 0, 255),            # magenta
    "spurious_copper": (0, 255, 0),   # green
}

# --- Auth / sessions ---
# Set PCB_SECRET_KEY in the environment for anything other than local dev.
SECRET_KEY: str = os.getenv("PCB_SECRET_KEY", "pcb-inspector-dev-secret-change-me")
SESSION_COOKIE: str = "pcb_session"
SESSION_MAX_AGE: int = 60 * 60 * 24 * 7  # 7 days

# Ensure dirs exist on import (Windows/Linux safe via pathlib)
for _d in (ORIGINAL_DIR, ANNOTATED_DIR, REPORT_DIR, BASE_DIR / "models"):
    _d.mkdir(parents=True, exist_ok=True)

"""PCB verification using OpenCV (own module, easy to improve later).

Two heuristic signals (no ML dependency):
1. Green solder-mask ratio (primary path for classic green boards).
2. Dominant surface-colour + trace-edge structure (fallback for
   red/blue/black/white solder-mask boards).

This is a guard rail, not a classifier: it passes PCB-like images and
rejects obvious non-PCB photos. Swap/improve internals without touching
callers — signature stays (is_pcb -> bool, message, metrics).
"""

from pathlib import Path

import cv2
import numpy as np

from app.config import (
    PCB_DOMINANT_BINS,
    PCB_GREEN_HSV_LOWER,
    PCB_GREEN_HSV_UPPER,
    PCB_MIN_DOMINANT_RATIO,
    PCB_MIN_DOMINANT_SATURATION,
    PCB_MIN_EDGE_DENSITY,
    PCB_MIN_GREEN_RATIO,
)


def green_ratio(hsv: np.ndarray) -> float:
    """Fraction of pixels in the PCB-green HSV range, in [0, 1]."""
    lower = np.array(PCB_GREEN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(PCB_GREEN_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / mask.size)


def edge_density(gray: np.ndarray) -> float:
    """Canny edge pixel density in [0, 1] (trace/structure signal)."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def dominant_color(hsv: np.ndarray,
                   bins: int = PCB_DOMINANT_BINS) -> tuple[float, int]:
    """Most common quantised colour: (share of frame, saturation).

    PCB photos are dominated by one board surface colour, while random
    photos are usually dominated by neutral/grey background.
    """
    quant = (hsv.astype(np.int32) * bins // 256)
    flat = quant[:, :, 0] * bins * bins + quant[:, :, 1] * bins + quant[:, :, 2]
    counts = np.bincount(flat.ravel(), minlength=bins ** 3)
    idx = int(counts.argmax())
    share = float(counts[idx] / flat.size)
    saturation = int(((idx // bins) % bins) * 256 / bins + 128 / bins)
    return share, saturation


def is_pcb(image_path: Path) -> tuple[bool, str, dict]:
    """Decide whether an image looks like a PCB.

    Returns:
        (is_pcb, message, metrics dict).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not read image for PCB verification.", {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gr = green_ratio(hsv)
    ed = edge_density(gray)
    dom_ratio, dom_sat = dominant_color(hsv)
    metrics: dict = {
        "green_ratio": gr,
        "edge_density": ed,
        "dominant_ratio": dom_ratio,
        "dominant_saturation": dom_sat,
    }
    if gr >= PCB_MIN_GREEN_RATIO:
        return True, "PCB verification passed.", metrics
    if (dom_ratio >= PCB_MIN_DOMINANT_RATIO
            and dom_sat >= PCB_MIN_DOMINANT_SATURATION
            and ed >= PCB_MIN_EDGE_DENSITY):
        return True, "PCB verification passed.", metrics
    return (
        False,
        "This does not look like a PCB image. "
        "Please upload a valid PCB image.",
        metrics,
    )

"""Heuristic PCB check using OpenCV (no ML dependency).

Two signals are used:
1. Green solder-mask ratio (primary path for the usual green boards).
2. Dominant surface-colour + trace-edge structure (fallback for non-green boards).

Both signals are computed with plain OpenCV/NumPy so the check stays fast and
dependency-light. It is a guard rail, not a classifier: it is tuned on PCB-like
images and is expected to reject obvious non-PCB photos.
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
    """Compute fraction of pixels in the PCB-green HSV range.

    Args:
        hsv: Image in HSV color space.

    Returns:
        Ratio of green-ish pixels in [0, 1].
    """
    lower = np.array(PCB_GREEN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(PCB_GREEN_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / mask.size)


def edge_density(gray: np.ndarray) -> float:
    """Compute Canny edge pixel density.

    Args:
        gray: Grayscale image array.

    Returns:
        Ratio of edge pixels in [0, 1].
    """
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def dominant_color(hsv: np.ndarray, bins: int = PCB_DOMINANT_BINS) -> tuple[float, int]:
    """Find the most common quantised colour and how much of the frame it covers.

    A PCB photograph is dominated by one board surface colour (solder mask or
    substrate), whereas photos of unrelated scenes are usually dominated by a
    neutral/grey background.

    Args:
        hsv: Image in HSV color space.
        bins: Quantisation levels per channel.

    Returns:
        Tuple of (share of frame, saturation of the dominant bin).
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

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (is_pcb, message, metrics dict with green_ratio,
        dominant_ratio, dominant_saturation and edge_density).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not read image for PCB check.", {}
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
        return True, "PCB check passed.", metrics
    if (
        dom_ratio >= PCB_MIN_DOMINANT_RATIO
        and dom_sat >= PCB_MIN_DOMINANT_SATURATION
        and ed >= PCB_MIN_EDGE_DENSITY
    ):
        return True, "PCB check passed (dominant surface colour fallback).", metrics
    return (
        False,
        "This does not look like a PCB image. Please upload a PCB photo.",
        metrics,
    )
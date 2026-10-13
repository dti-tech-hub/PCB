"""Image quality checks using OpenCV (runs BEFORE detection).

Checks, in order:
1. minimum resolution
2. blur (Laplacian variance)
3. brightness (too dark / too bright)

If any check fails, the pipeline must STOP with a clear message.
"""

from pathlib import Path

import cv2
import numpy as np

from app.config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    MIN_HEIGHT,
    MIN_WIDTH,
)


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur score = variance of Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    """Mean grayscale value in [0, 255]."""
    return float(np.mean(gray))


def check_image_quality(image_path: Path) -> tuple[bool, str, dict]:
    """Validate resolution, blur and brightness.

    Returns:
        (passed, message, metrics dict with blur/brightness/size).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not read image. File may be corrupted.", {}
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return (
            False,
            f"Image resolution is too small ({w}x{h}). "
            f"Minimum is {MIN_WIDTH}x{MIN_HEIGHT}. "
            "Please upload a higher-resolution image.",
            {"width": w, "height": h},
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = laplacian_variance(gray)
    bright = mean_brightness(gray)
    metrics: dict = {"blur": blur, "brightness": bright,
                     "width": w, "height": h}
    if blur < BLUR_THRESHOLD:
        return (
            False,
            f"Image quality is poor (blurry). Please upload a clearer image. "
            f"(sharpness {blur:.1f} < {BLUR_THRESHOLD})",
            metrics,
        )
    if bright < BRIGHTNESS_MIN:
        return (
            False,
            f"Image quality is poor (too dark). Please upload a brighter image. "
            f"(brightness {bright:.1f} < {BRIGHTNESS_MIN})",
            metrics,
        )
    if bright > BRIGHTNESS_MAX:
        return (
            False,
            f"Image quality is poor (too bright). Please upload a darker image. "
            f"(brightness {bright:.1f} > {BRIGHTNESS_MAX})",
            metrics,
        )
    return True, "Quality check passed.", metrics

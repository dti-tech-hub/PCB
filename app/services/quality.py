"""Image quality checks using OpenCV."""

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
    """Compute blur score (variance of Laplacian) for a grayscale image.

    Args:
        gray: Grayscale image array.

    Returns:
        Blur score; higher means sharper.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    """Compute mean brightness of a grayscale image.

    Args:
        gray: Grayscale image array.

    Returns:
        Mean pixel value in [0, 255].
    """
    return float(np.mean(gray))


def check_image_quality(image_path: Path) -> tuple[bool, str, dict]:
    """Validate blur, brightness and minimum resolution.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (passed, message, metrics dict with blur/brightness/size).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not read image. File may be corrupted.", {}
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return (
            False,
            f"Image too small ({w}x{h}). Minimum is {MIN_WIDTH}x{MIN_HEIGHT}.",
            {"width": w, "height": h},
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = laplacian_variance(gray)
    bright = mean_brightness(gray)
    metrics: dict = {"blur": blur, "brightness": bright, "width": w, "height": h}
    if blur < BLUR_THRESHOLD:
        return False, f"Image is too blurry (score {blur:.1f} < {BLUR_THRESHOLD}).", metrics
    if bright < BRIGHTNESS_MIN:
        return False, f"Image is too dark (brightness {bright:.1f} < {BRIGHTNESS_MIN}).", metrics
    if bright > BRIGHTNESS_MAX:
        return False, f"Image is too bright (brightness {bright:.1f} > {BRIGHTNESS_MAX}).", metrics
    return True, "Quality check passed.", metrics

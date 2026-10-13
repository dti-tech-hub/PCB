"""Generate PCB sample images for manual testing of the web app.

Run: python scripts/generate_samples.py
Writes to data/uploads/samples/.
"""

import random
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "uploads" / "samples"


def small_sharp_pcb() -> np.ndarray:
    """Build a 600x600 green PCB-like image (YOLO path, expects Pass).

    Returns:
        BGR image array.
    """
    rng = np.random.default_rng(0)
    img = np.full((600, 600, 3), (0, 128, 0), dtype=np.uint8)
    for i in range(0, 600, 50):
        cv2.line(img, (0, i), (600, i), (200, 200, 200), 2)
        cv2.line(img, (i, 0), (i, 600), (200, 200, 200), 2)
    for _ in range(150):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 600))
        cv2.circle(img, (x, y), 3, (0, 0, 0), -1)
    return img


def large_textured_pcb() -> np.ndarray:
    """Build a 1600x1600 textured PCB image (SAHI 4-slice path, expects Pass).

    Returns:
        BGR image array.
    """
    rng = np.random.default_rng(1)
    img = np.full((1600, 1600, 3), (0, 130, 0), dtype=np.uint8)
    for i in range(0, 1600, 80):
        cv2.line(img, (0, i), (1600, i), (220, 220, 220), 2)
        cv2.line(img, (i, 0), (i, 1600), (220, 220, 220), 2)
    for _ in range(600):
        x, y = int(rng.integers(0, 1600)), int(rng.integers(0, 1600))
        cv2.circle(img, (x, y), 4, (0, 0, 0), -1)
    return img


def non_pcb_photo() -> np.ndarray:
    """Build a non-PCB image that must be rejected at step 3.

    Returns:
        BGR image array.
    """
    rng = np.random.default_rng(2)
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    for _ in range(80):
        x, y = int(rng.integers(0, 400)), int(rng.integers(0, 400))
        cv2.circle(img, (x, y), int(rng.integers(10, 30)), (150, 120, 200), -1)
    return img


def blurry_image() -> np.ndarray:
    """Build a blurry image that must be rejected at step 2.

    Returns:
        BGR image array.
    """
    img = np.full((600, 600, 3), 128, dtype=np.uint8)
    return cv2.GaussianBlur(img, (51, 51), 0)


def main() -> None:
    """Write all sample images to data/uploads/samples."""
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {
        "1_small_pcb_600.png": small_sharp_pcb(),
        "2_large_pcb_1600.png": large_textured_pcb(),
        "3_non_pcb.png": non_pcb_photo(),
        "4_blurry.png": blurry_image(),
    }
    for name, img in samples.items():
        cv2.imwrite(str(OUT / name), img)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()

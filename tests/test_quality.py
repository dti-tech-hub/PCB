"""Tests for quality_check service."""

from pathlib import Path

import cv2
import numpy as np

from app.services.quality_check import check_image_quality


def _write_image(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def test_quality_pass(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(60, 180, size=(400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (0, 255, 0), 3)
    p = _write_image(tmp_path / "good.png", img)
    passed, _, metrics = check_image_quality(p)
    assert metrics["width"] == 400
    assert isinstance(passed, bool)


def test_quality_rejects_small(tmp_path: Path) -> None:
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    p = _write_image(tmp_path / "small.png", img)
    passed, msg, _ = check_image_quality(p)
    assert not passed
    assert "too small" in msg.lower()


def test_quality_rejects_blurry(tmp_path: Path) -> None:
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    img = cv2.GaussianBlur(img, (51, 51), 0)
    p = _write_image(tmp_path / "blur.png", img)
    passed, msg, _ = check_image_quality(p)
    assert not passed
    assert "blur" in msg.lower()

"""Tests for pcb_verify service."""

from pathlib import Path

import cv2
import numpy as np

from app.services.pcb_verify import (dominant_color, edge_density,
                                     green_ratio, is_pcb)


def _write(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def test_pcb_green_passes(tmp_path: Path) -> None:
    img = np.full((400, 400, 3), (0, 128, 0), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), 2)
    for i in range(10):
        cv2.line(img, (20, 30 + i * 30), (380, 30 + i * 30), (255, 255, 0), 2)
    p = _write(tmp_path / "pcb.png", img)
    ok, _, metrics = is_pcb(p)
    assert ok
    assert metrics["green_ratio"] > 0


def test_pcb_rejects_plain_grey(tmp_path: Path) -> None:
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    p = _write(tmp_path / "plain.png", img)
    ok, msg, _ = is_pcb(p)
    assert not ok
    assert "pcb" in msg.lower()


def test_pcb_blue_board_passes_fallback(tmp_path: Path) -> None:
    img = np.full((400, 400, 3), (200, 80, 20), dtype=np.uint8)
    for i in range(0, 400, 20):
        cv2.line(img, (0, i), (400, i), (255, 255, 255), 2)
    p = _write(tmp_path / "blue.png", img)
    ok, _, metrics = is_pcb(p)
    assert ok
    assert metrics["green_ratio"] < 0.05
    assert metrics["dominant_saturation"] >= 40


def test_dominant_color_grey_is_achromatic() -> None:
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    share, sat = dominant_color(hsv)
    assert share > 0.9
    assert sat < 40


def test_edge_density_and_green_ratio_helpers() -> None:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (0, 128, 0)
    cv2.line(img, (0, 0), (300, 0), (255, 255, 255), 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert green_ratio(hsv) > 0.9
    assert edge_density(gray) > 0

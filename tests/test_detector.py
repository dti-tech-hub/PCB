"""Tests for detector helpers (no weights required)."""

from pathlib import Path

import cv2
import numpy as np

from app.config import DEFECT_CLASSES
from app.services.detector import (category_mapping, count_per_class,
                                   decide_pass_fail, draw_boxes,
                                   is_large_image, nms_merge,
                                   set_conf_threshold)


def test_category_mapping_order() -> None:
    m = category_mapping()
    assert [m[str(i)] for i in range(6)] == DEFECT_CLASSES


def test_is_large_image() -> None:
    assert is_large_image(1281, 100)
    assert not is_large_image(640, 640)


def test_decide_pass_fail() -> None:
    assert decide_pass_fail([]) == "PASS"
    assert decide_pass_fail([{"class_name": "short",
                              "confidence": 0.9}]) == "FAIL"


def test_set_conf_threshold_clamps() -> None:
    assert set_conf_threshold(0.25) == 0.25
    assert set_conf_threshold(99) <= 0.95
    assert set_conf_threshold(-1) >= 0.05
    assert set_conf_threshold("bad") == 0.25


def test_nms_merge_suppresses_overlap() -> None:
    a = {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 3,
         "class_name": "short", "confidence": 0.9}
    b = {"x1": 5, "y1": 5, "x2": 105, "y2": 105, "class_id": 3,
         "class_name": "short", "confidence": 0.5}
    assert len(nms_merge([a, b])) == 1


def test_count_per_class() -> None:
    dets = [{"class_name": "short"}, {"class_name": "short"},
            {"class_name": "spur"}]
    counts = count_per_class(dets)
    assert counts["short"] == 2 and counts["spur"] == 1
    assert counts["missing_hole"] == 0


def test_draw_boxes(tmp_path: Path) -> None:
    img = np.full((300, 300, 3), (0, 128, 0), dtype=np.uint8)
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), img)
    out = tmp_path / "ann.png"
    dets = [{"x1": 10, "y1": 10, "x2": 100, "y2": 100, "class_id": 3,
             "class_name": "short", "confidence": 0.88}]
    assert draw_boxes(src, dets, out).exists()

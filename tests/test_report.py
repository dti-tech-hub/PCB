"""Tests for PDF report service."""

from pathlib import Path

from app.services.report import export_csv, generate_pdf


def test_generate_pdf(tmp_path: Path) -> None:
    result = {"image_name": "a.png", "status": "FAIL", "total_defects": 1,
              "sahi_used": 0, "confidence": 0.25, "processing_time": 1.1,
              "created_at": "now"}
    dets = [{"class_name": "short", "confidence": 0.9, "x1": 1, "y1": 2,
             "x2": 3, "y2": 4}]
    out = tmp_path / "r.pdf"
    generate_pdf(result, dets, out, None, None, username="tester")
    assert out.exists() and out.stat().st_size > 1000


def test_export_csv(tmp_path: Path) -> None:
    rows = [{"id": 1, "filename": "a.png", "verdict": "Pass",
             "num_defects": 0, "method": "yolo", "created_at": "now"}]
    out = tmp_path / "h.csv"
    export_csv(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "filename" in text and "a.png" in text

"""PDF reports with ReportLab (Platypus): title, meta, images, tables."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def _img(path: Path | None, max_w: float = 80 * mm,
         max_h: float = 60 * mm):
    """Flowable image or None if missing (never crash the report)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return RLImage(str(p), width=max_w, height=max_h,
                       kind="proportional")
    except Exception:
        return None


def generate_pdf(result: dict[str, Any], detections: list[dict[str, Any]],
                 output_path: Path,
                 original_image: Path | None = None,
                 annotated_image: Path | None = None,
                 username: str = "-") -> Path:
    """Build a full inspection PDF report.

    Contains: title, date/time, user, original + annotated images,
    defect summary table, detection details table, PASS/FAIL result.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    story: list = []

    # Title
    story.append(Paragraph("PCB Defect Inspector — Inspection Report",
                           styles["Title"]))
    story.append(Spacer(1, 4 * mm))

    created = result.get("created_at") or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    verdict = result.get("status") or result.get("verdict") or "-"
    conf = result.get("confidence", "-")
    sahi = ("Yes" if str(result.get("sahi_used", "0")) in ("1", "True", "yes")
            else "No")
    method = result.get("method", "sahi" if sahi == "Yes" else "yolo")
    proc = result.get("processing_time", "-")

    meta = [
        ["Field", "Value"],
        ["Date / Time", str(created)],
        ["User", str(username)],
        ["Image", str(result.get("image_name") or result.get("filename", "-"))],
        ["Status", str(verdict)],
        ["Total defects", str(result.get("total_defects",
                                        result.get("num_defects",
                                                   len(detections))))],
        ["SAHI mode", sahi],
        ["Method", str(method)],
        ["Confidence", str(conf)],
        ["Processing time (s)",
         f"{float(proc):.2f}" if isinstance(proc, (int, float)) else str(proc)],
    ]
    t = Table(meta, colWidths=[45 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eff6ff")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    # Images side by side
    oi, ai = _img(original_image), _img(annotated_image)
    if oi is not None or ai is not None:
        story.append(Paragraph("Original vs Annotated", styles["Heading2"]))
        img_tbl = Table(
            [[oi or Paragraph("—", styles["Normal"]),
              ai or Paragraph("—", styles["Normal"])]],
            colWidths=[77 * mm, 77 * mm])
        img_tbl.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(img_tbl)
        story.append(Spacer(1, 5 * mm))

    # Summary per class
    story.append(Paragraph("Defect summary (count per class)",
                           styles["Heading2"]))
    from app.config import DEFECT_CLASSES
    counts = {c: 0 for c in DEFECT_CLASSES}
    for d in detections:
        if str(d.get("class_name")) in counts:
            counts[str(d.get("class_name"))] += 1
    summ = [["Defect class", "Count"]] + [[c, counts[c]] for c in DEFECT_CLASSES]
    st = Table(summ, colWidths=[90 * mm, 65 * mm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(st)
    story.append(Spacer(1, 5 * mm))

    # Details
    story.append(Paragraph("Detection details", styles["Heading2"]))
    if not detections:
        story.append(Paragraph("No defects above threshold. PASS.",
                               styles["Normal"]))
    else:
        rows = [["#", "Class", "Conf.", "Box (x1,y1,x2,y2)"]]
        for i, d in enumerate(detections, 1):
            rows.append([
                str(i), str(d.get("class_name")),
                f"{float(d.get('confidence', 0)):.2f}",
                f"{d.get('x1', 0):.0f},{d.get('y1', 0):.0f},"
                f"{d.get('x2', 0):.0f},{d.get('y2', 0):.0f}",
            ])
        dt = Table(rows, colWidths=[12 * mm, 45 * mm, 25 * mm, 73 * mm])
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(dt)

    story.append(Spacer(1, 6 * mm))
    color = (colors.green if str(verdict).upper() == "PASS" else colors.red)
    story.append(Paragraph(f"<b>Result: {verdict}</b>", styles["Heading1"]))
    # Colour the last paragraph via extra style is complex; keep bold text.

    doc.build(story)
    return output_path


def report_path_for(result_id: int, base: Path) -> Path:
    """PDF path for an inspection id."""
    return base / f"report_{result_id}.pdf"


def export_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Compat CSV export (history rows) using stdlib csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "filename", "verdict", "num_defects",
                           "method", "created_at"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    return output_path

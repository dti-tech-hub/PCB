"""Upload/detect/history/report routes. Pipeline order enforced here."""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import (
    ALLOWED_EXTENSIONS,
    ANNOTATED_DIR,
    APP_DIR,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    REPORT_DIR,
    UPLOAD_DIR,
)
from app.services import db as db_service
from app.services import detector as detector_service
from app.services import pcb_check as pcb_service
from app.services import quality as quality_service
from app.services import report as report_service

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def validate_upload(filename: str, size: int) -> tuple[bool, str]:
    """Validate file extension and size.

    Args:
        filename: Original client filename.
        size: File size in bytes.

    Returns:
        Tuple of (valid, message).
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}."
    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File too large ({size / 1024 / 1024:.1f} MB > {MAX_FILE_SIZE_MB} MB)."
    if size == 0:
        return False, "Empty file uploaded."
    return True, "OK"


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the upload/home page.

    Args:
        request: FastAPI request.

    Returns:
        HTML response.
    """
    return templates.TemplateResponse(request, "index.html", {"error": None})


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Handle upload through the full pipeline: validate -> quality -> PCB -> detect.

    Args:
        request: FastAPI request.
        file: Uploaded image file.

    Returns:
        Result page or home page with a clear error message (never a traceback).
    """
    try:
        data = await file.read()
        ok, msg = validate_upload(file.filename or "", len(data))
        if not ok:
            return templates.TemplateResponse(request, "index.html", {"error": msg}, status_code=400)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "upload.png").suffix.lower()
        saved_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = UPLOAD_DIR / saved_name
        saved_path.write_bytes(data)

        # 2. quality check
        passed, qmsg, _ = quality_service.check_image_quality(saved_path)
        if not passed:
            return templates.TemplateResponse(request, "index.html", {"error": qmsg}, status_code=400)

        # 3. PCB check
        is_pcb, pmsg, _ = pcb_service.is_pcb(saved_path)
        if not is_pcb:
            return templates.TemplateResponse(request, "index.html", {"error": pmsg}, status_code=400)

        # 4. detection (graceful if weights missing)
        try:
            detections, method = detector_service.detect(saved_path)
        except FileNotFoundError as e:
            return templates.TemplateResponse(request, "index.html", {"error": str(e)}, status_code=503)
        except Exception:
            return templates.TemplateResponse(
                request, "index.html", {"error": "Detection failed. Try another image."}, status_code=500
            )

        # 5. pass/fail
        verdict = detector_service.decide_pass_fail(detections)

        # 6. annotate
        annotated_name = f"annotated_{saved_name}"
        annotated_path = ANNOTATED_DIR / annotated_name
        detector_service.draw_boxes(saved_path, detections, annotated_path)

        # 7. save to sqlite
        result_id = db_service.save_result(
            filename=file.filename or saved_name,
            verdict=verdict,
            detections=detections,
            annotated_image=annotated_name,
            method=method,
        )
        # 8. result page
        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "result_id": result_id,
                "filename": file.filename,
                "verdict": verdict,
                "detections": detections,
                "method": method,
                "annotated": annotated_name,
            },
        )
    except Exception:
        return templates.TemplateResponse(
            request, "index.html", {"error": "Upload failed. Please try again."}, status_code=500
        )


@router.get("/history", response_class=HTMLResponse)
def history(request: Request) -> HTMLResponse:
    """Render inspection history page.

    Args:
        request: FastAPI request.

    Returns:
        HTML response.
    """
    rows = db_service.get_all()
    return templates.TemplateResponse(request, "history.html", {"rows": rows})


@router.get("/result/{result_id}", response_class=HTMLResponse)
def result_detail(request: Request, result_id: int) -> HTMLResponse:
    """Render a single saved result.

    Args:
        request: FastAPI request.
        result_id: DB row id.

    Returns:
        HTML response.
    """
    row = db_service.get_by_id(result_id)
    if row is None:
        return templates.TemplateResponse(request, "index.html", {"error": "Result not found."}, status_code=404)
    try:
        detections = json.loads(row.get("detections") or "[]")
    except Exception:
        detections = []
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result_id": row["id"],
            "filename": row["filename"],
            "verdict": row["verdict"],
            "detections": detections,
            "method": row.get("method", "yolo"),
            "annotated": row.get("annotated_image", ""),
        },
    )


@router.get("/report/{result_id}", response_model=None)
def report_pdf(result_id: int) -> FileResponse | RedirectResponse:
    """Generate and return a PDF report for a result.

    Args:
        result_id: DB row id.

    Returns:
        PDF file response or redirect on error.
    """
    row = db_service.get_by_id(result_id)
    if row is None:
        return RedirectResponse(url="/history")
    try:
        detections = json.loads(row.get("detections") or "[]")
    except Exception:
        detections = []
    out = report_service.report_path_for(result_id, REPORT_DIR)
    ann_name = row.get("annotated_image") or ""
    ann_path = (ANNOTATED_DIR / ann_name) if ann_name else None
    try:
        report_service.generate_pdf(dict(row), detections, out, ann_path)
        return FileResponse(str(out), media_type="application/pdf", filename=out.name)
    except Exception:
        return RedirectResponse(url="/history")


@router.get("/export.csv")
def export_csv() -> FileResponse:
    """Export history to CSV.

    Returns:
        CSV file response.
    """
    rows = db_service.get_all()
    out = REPORT_DIR / "history.csv"
    report_service.export_csv(rows, out)
    return FileResponse(str(out), media_type="text/csv", filename="history.csv")


@router.get("/files/annotated/{name}")
def annotated_file(name: str) -> FileResponse:
    """Serve an annotated image by filename (safe join).

    Args:
        name: Image filename.

    Returns:
        Image file response.
    """
    path = (ANNOTATED_DIR / Path(name).name).resolve()
    if not str(path).startswith(str(ANNOTATED_DIR.resolve())) or not path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))

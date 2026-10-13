"""PCB Defect Inspector — FastAPI entrypoint.

Workflow (in order):
1. auth (register/login/logout, session cookie)
2. upload validation (jpg/jpeg/png, 20 MB)
3. quality check -> STOP with clear message on fail
4. PCB verification -> STOP with clear message on fail
5. YOLO26 detection (model loaded once at startup)
6. optional SAHI 4-slice mode
7. annotate + save original/annotated under uploads/
8. save inspection in SQLite (per user)
9. history list/view/delete (own records only)
10. PDF report (ReportLab) from result + history pages.
"""

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               Response)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import auth as auth_mod
from app import database as db
from app.config import (ALLOWED_EXTENSIONS, ANNOTATED_DIR, APP_DIR,
                        CONF_THRESHOLD, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB,
                        ORIGINAL_DIR, REPORT_DIR, SESSION_COOKIE)
from app.services import detector as detector_service
from app.services import pcb_verify as pcb_service
from app.services import quality_check as quality_service
from app.services import report as report_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init DB + preload YOLO model once at startup."""
    db.init_db()
    try:
        detector_service.warm_model()
    except Exception:
        pass  # missing weights -> friendly 503 later, never crash boot
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="PCB Defect Inspector", lifespan=lifespan)
    # Ensure tables exist even when lifespan is skipped (e.g. TestClient).
    db.init_db()
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")),
              name="static")
    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

    # ---------- helpers ----------

    def current_user(request: Request) -> Optional[dict]:
        return auth_mod.get_current_user(request)

    def login_required(request: Request):
        user = current_user(request)
        if user is None:
            return None
        return user

    def validate_upload(filename: str, size: int) -> tuple[bool, str]:
        ext = Path(filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, (
                f"Invalid file type '{ext}'. Allowed: jpg, jpeg, png.")
        if size == 0:
            return False, "Empty file uploaded."
        if size > MAX_FILE_SIZE_BYTES:
            return False, (
                f"File too large ({size / 1024 / 1024:.1f} MB > "
                f"{MAX_FILE_SIZE_MB:.0f} MB).")
        return True, "OK"

    # ---------- auth pages (public) ----------

    @app.get("/register", response_class=HTMLResponse)
    def register_page(request: Request):
        if current_user(request):
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "register.html",
                                          {"error": None})

    @app.post("/register")
    def register_post(request: Request, username: str = Form(""),
                      password: str = Form(""), confirm: str = Form("")):
        err = auth_mod.validate_registration(username, password, confirm)
        if err:
            return templates.TemplateResponse(
                request, "register.html", {"error": err}, status_code=400)
        try:
            uid = db.create_user(username.strip(),
                                 auth_mod.hash_password(password))
        except Exception:
            return templates.TemplateResponse(
                request, "register.html",
                {"error": "Username already taken."}, status_code=400)
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(SESSION_COOKIE, auth_mod.create_session_token(uid),
                        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
        return resp

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if current_user(request):
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "login.html",
                                          {"error": None})

    @app.post("/login")
    def login_post(request: Request, username: str = Form(""),
                   password: str = Form("")):
        ok, payload = auth_mod.login_response_redirect(username, password)
        if not ok:
            return templates.TemplateResponse(
                request, "login.html", {"error": payload}, status_code=401)
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(SESSION_COOKIE,
                        auth_mod.create_session_token(int(payload["id"])),
                        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
        return resp

    @app.get("/logout")
    def logout():
        resp = RedirectResponse(url="/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    # ---------- main pages (login required) ----------

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        rows = db.get_inspections_by_user(int(user["id"]))
        total = len(rows)
        fails = sum(1 for r in rows if str(r.get("status")).upper() == "FAIL")
        return templates.TemplateResponse(
            request, "home.html",
            {"user": user, "total": total, "fails": fails,
             "passes": total - fails})

    @app.get("/inspect", response_class=HTMLResponse)
    def inspect_page(request: Request):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        return templates.TemplateResponse(
            request, "inspect.html",
            {"user": user, "error": None,
             "default_conf": CONF_THRESHOLD})

    @app.post("/inspect", response_class=HTMLResponse)
    async def inspect_post(request: Request,
                           file: UploadFile = File(...),
                           confidence: float = Form(CONF_THRESHOLD),
                           sahi: Optional[str] = Form(None)):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        use_sahi = bool(sahi)  # checkbox "on" when toggled
        conf = detector_service.set_conf_threshold(confidence)
        try:
            data = await file.read()
        except Exception:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": "Upload failed. Please try again.",
                 "default_conf": conf}, status_code=400)
        ok, msg = validate_upload(file.filename or "", len(data))
        if not ok:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": msg, "default_conf": conf},
                status_code=400)
        # Save original
        ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "upload.png").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            ext = ".png"
        saved_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = ORIGINAL_DIR / saved_name
        try:
            saved_path.write_bytes(data)
        except Exception:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": "Could not save upload.",
                 "default_conf": conf}, status_code=500)

        # 3. quality check (STOP on fail, no detection)
        passed, qmsg, _ = quality_service.check_image_quality(saved_path)
        if not passed:
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": qmsg, "default_conf": conf},
                status_code=400)

        # 4. PCB verification (STOP on fail)
        is_pcb, pmsg, _ = pcb_service.is_pcb(saved_path)
        if not is_pcb:
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": pmsg, "default_conf": conf},
                status_code=400)

        # 5-6. detection
        try:
            detections, method, seconds = detector_service.detect(
                saved_path, conf=conf, use_sahi=use_sahi)
        except FileNotFoundError as e:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": str(e), "default_conf": conf},
                status_code=503)
        except Exception:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user,
                 "error": "Detection failed. Try another image.",
                 "default_conf": conf}, status_code=500)

        # 7. annotate
        status = detector_service.decide_pass_fail(detections)
        annotated_name = f"annotated_{saved_name}"
        annotated_path = ANNOTATED_DIR / annotated_name
        try:
            detector_service.draw_boxes(saved_path, detections, annotated_path)
        except Exception:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": "Could not annotate image.",
                 "default_conf": conf}, status_code=500)

        # 8. save inspection
        try:
            rid = db.save_inspection(
                user_id=int(user["id"]),
                image_name=file.filename or saved_name,
                original_path=saved_name,
                annotated_path=annotated_name,
                detections=detections,
                status=status,
                sahi_used=use_sahi,
                confidence=conf,
                processing_time=seconds,
            )
        except Exception:
            return templates.TemplateResponse(
                request, "inspect.html",
                {"user": user, "error": "Could not save inspection.",
                 "default_conf": conf}, status_code=500)
        return RedirectResponse(url=f"/result/{rid}", status_code=303)

    @app.get("/result/{result_id}", response_class=HTMLResponse)
    def result_detail(request: Request, result_id: int):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        row = db.get_inspection(result_id, int(user["id"]))
        if row is None:
            return RedirectResponse(url="/history", status_code=303)
        try:
            detections = json.loads(row.get("detections") or "[]")
        except Exception:
            detections = []
        counts = detector_service.count_per_class(detections)
        return templates.TemplateResponse(
            request, "result.html",
            {"user": user, "row": row, "detections": detections,
             "counts": counts})

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        rows = db.get_inspections_by_user(int(user["id"]))
        return templates.TemplateResponse(request, "history.html",
                                          {"user": user, "rows": rows})

    @app.post("/delete/{result_id}")
    def delete_post(request: Request, result_id: int):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        row = db.get_inspection(result_id, int(user["id"]))
        if row:
            # remove files best-effort, then DB row
            for key, folder in (("original_path", ORIGINAL_DIR),
                                ("annotated_path", ANNOTATED_DIR)):
                try:
                    p = (folder / Path(str(row.get(key) or "")).name)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            db.delete_inspection(result_id, int(user["id"]))
        return RedirectResponse(url="/history", status_code=303)

    @app.get("/delete/{result_id}")
    def delete_get(request: Request, result_id: int):
        # Allow GET delete for simple button links (calls same logic).
        return delete_post(request, result_id)

    @app.get("/report/{result_id}")
    def report_pdf(request: Request, result_id: int):
        user = login_required(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        row = db.get_inspection(result_id, int(user["id"]))
        if row is None:
            return RedirectResponse(url="/history", status_code=303)
        try:
            detections = json.loads(row.get("detections") or "[]")
        except Exception:
            detections = []
        out = report_service.report_path_for(result_id, REPORT_DIR)
        orig = ORIGINAL_DIR / Path(str(row.get("original_path") or "")).name
        ann = ANNOTATED_DIR / Path(str(row.get("annotated_path") or "")).name
        try:
            report_service.generate_pdf(dict(row), detections, out,
                                        orig if orig.exists() else None,
                                        ann if ann.exists() else None,
                                        username=str(user.get("username")))
            return FileResponse(str(out), media_type="application/pdf",
                                filename=out.name)
        except Exception:
            return RedirectResponse(url=f"/result/{result_id}",
                                    status_code=303)

    # ---------- uploaded image serving (login + safe join) ----------

    def _serve(folder: Path, name: str):
        path = (folder / Path(name).name).resolve()
        if (not str(path).startswith(str(folder.resolve()))
                or not path.exists()):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(str(path))

    @app.get("/uploads/original/{name}")
    def serve_original(request: Request, name: str):
        if login_required(request) is None:
            return RedirectResponse(url="/login", status_code=303)
        return _serve(ORIGINAL_DIR, name)

    @app.get("/uploads/annotated/{name}")
    def serve_annotated(request: Request, name: str):
        if login_required(request) is None:
            return RedirectResponse(url="/login", status_code=303)
        return _serve(ANNOTATED_DIR, name)

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> Response:
        # Never leak tracebacks to the browser.
        try:
            templates_env = Jinja2Templates(
                directory=str(APP_DIR / "templates"))
            user = None
            try:
                user = auth_mod.get_current_user(request)
            except Exception:
                pass
            if user is None and request.url.path not in (
                    "/login", "/register"):
                return RedirectResponse(url="/login", status_code=303)
            return templates_env.TemplateResponse(
                request, "inspect.html" if user else "login.html",
                {"user": user, "error": "Something went wrong. "
                                       "Please try again.",
                 "default_conf": CONF_THRESHOLD}, status_code=500)
        except Exception:
            return HTMLResponse("Something went wrong.", status_code=500)

    return app


app = create_app()

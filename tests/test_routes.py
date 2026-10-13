"""Route tests for PCB Defect Inspector (auth + pipeline, mocked detector)."""

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import database as db
from app.main import create_app


def _png_bytes(w: int = 400, h: int = 400, color=(0, 128, 0)) -> bytes:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (w - 50, h - 50), (255, 255, 255), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def _authed_client(tmp_path: Path, monkeypatch, username="u1") -> TestClient:
    dbp = tmp_path / f"{username}.db"
    monkeypatch.setattr(db, "DB_PATH", dbp)
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.post("/register", data={"username": username, "password": "pass1234",
                                  "confirm": "pass1234"})
    assert r.status_code in (200, 303)
    return c


def test_login_required_redirects():
    c = TestClient(create_app(), raise_server_exceptions=False)
    for path in ("/", "/inspect", "/history"):
        r = c.get(path, follow_redirects=False)
        assert r.status_code == 303
        assert "/login" in r.headers["location"]


def test_register_login_validation(tmp_path, monkeypatch):
    dbp = tmp_path / "v.db"
    monkeypatch.setattr(db, "DB_PATH", dbp)
    c = TestClient(create_app(), raise_server_exceptions=False)
    assert c.post("/register", data={"username": "ab", "password": "x",
                                     "confirm": "x"}).status_code == 400
    assert c.post("/register", data={"username": "good",
                                     "password": "p", "confirm": "q"}
                  ).status_code == 400
    assert c.post("/login", data={"username": "no", "password": "no"}
                  ).status_code == 401


def test_upload_rejects_bad_extension(tmp_path, monkeypatch):
    c = _authed_client(tmp_path, monkeypatch, "u_bad")
    r = c.post("/inspect", files={"file": ("x.txt", b"hi", "text/plain")},
               data={"confidence": "0.25"})
    assert r.status_code == 400 and "Invalid file type" in r.text


def test_upload_rejects_quality(tmp_path, monkeypatch):
    from app.main import create_app as _ca  # noqa
    from app.services import quality_check as qc
    c = _authed_client(tmp_path, monkeypatch, "u_q")
    monkeypatch.setattr(qc, "check_image_quality",
                        lambda p: (False, "Image quality is poor (blurry). "
                                          "Please upload a clearer image.", {}))
    r = c.post("/inspect", files={"file": ("a.png", _png_bytes(),
                                           "image/png")},
               data={"confidence": "0.25"})
    assert r.status_code == 400 and "blurry" in r.text.lower()


def test_upload_rejects_nonpcb(tmp_path, monkeypatch):
    from app.services import pcb_verify as pv
    from app.services import quality_check as qc
    c = _authed_client(tmp_path, monkeypatch, "u_pcb")
    monkeypatch.setattr(qc, "check_image_quality",
                        lambda p: (True, "ok", {}))
    monkeypatch.setattr(pv, "is_pcb",
                        lambda p: (False, "This does not look like a PCB "
                                          "image. Please upload a valid PCB "
                                          "image.", {}))
    r = c.post("/inspect", files={"file": ("a.png", _png_bytes(),
                                           "image/png")},
               data={"confidence": "0.25"})
    assert r.status_code == 400 and "PCB" in r.text


def test_full_flow_history_pdf_delete(tmp_path, monkeypatch):
    from app.services import detector as det
    from app.services import pcb_verify as pv
    from app.services import quality_check as qc
    c = _authed_client(tmp_path, monkeypatch, "u_full")
    monkeypatch.setattr(qc, "check_image_quality",
                        lambda p: (True, "ok", {}))
    monkeypatch.setattr(pv, "is_pcb", lambda p: (True, "ok", {}))
    dets = [{"x1": 1, "y1": 2, "x2": 30, "y2": 40, "class_id": 1,
             "class_name": "mouse_bite", "confidence": 0.9}]
    monkeypatch.setattr(det, "detect", lambda p, conf=0.25, use_sahi=False:
                        (dets, "sahi" if use_sahi else "yolo", 0.5))
    r = c.post("/inspect", files={"file": ("pcb.png", _png_bytes(),
                                           "image/png")},
               data={"confidence": "0.3", "sahi": "on"})
    assert r.status_code == 200 and "FAIL" in r.text
    assert "mouse_bite" in r.text and "Download PDF" in r.text
    h = c.get("/history")
    assert h.status_code == 200 and "pcb.png" in h.text
    # result id 1 in isolated db
    p = c.get("/report/1")
    assert p.status_code == 200
    assert p.headers["content-type"] == "application/pdf"
    d = c.post("/delete/1", follow_redirects=False)
    assert d.status_code == 303
    assert "/result/1" not in c.get("/history").text


def test_history_isolation(tmp_path, monkeypatch):
    from app.services import detector as det
    from app.services import pcb_verify as pv
    from app.services import quality_check as qc
    monkeypatch.setattr(qc, "check_image_quality",
                        lambda p: (True, "ok", {}))
    monkeypatch.setattr(pv, "is_pcb", lambda p: (True, "ok", {}))
    monkeypatch.setattr(det, "detect",
                        lambda p, conf=0.25, use_sahi=False: ([], "yolo", 0.1))
    c1 = _authed_client(tmp_path, monkeypatch, "iso1")
    c1.post("/inspect", files={"file": ("a.png", _png_bytes(),
                                        "image/png")},
            data={"confidence": "0.25"})
    # second user on separate db sees nothing (per-user isolation)
    dbp2 = tmp_path / "iso2.db"
    monkeypatch.setattr(db, "DB_PATH", dbp2)
    c2 = TestClient(create_app(), raise_server_exceptions=False)
    c2.post("/register", data={"username": "iso2", "password": "pass1234",
                               "confirm": "pass1234"})
    assert "No inspections yet" in c2.get("/history").text

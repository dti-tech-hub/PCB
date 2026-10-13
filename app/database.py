"""SQLite storage for users + inspections (stdlib sqlite3, no ORM)."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app import config as _cfg


def _resolve(db_path: Optional[Path] = None) -> Path:
    """Resolve DB path dynamically (so tests can monkeypatch config)."""
    if db_path is not None:
        return db_path
    # Respect monkeypatched app.database.DB_PATH in tests.
    try:
        g = globals().get("DB_PATH")
        if isinstance(g, Path):
            return g
    except Exception:
        pass
    return _cfg.DB_PATH


# Expose DB_PATH for backwards-compat patching (tests set db.DB_PATH).
DB_PATH: Path = _cfg.DB_PATH


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory."""
    # Support both explicit path and patched module attr.
    if db_path is None:
        try:
            db_path = DB_PATH
        except Exception:
            db_path = _cfg.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create users + inspections tables if missing."""
    conn = get_connection(_resolve(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_name TEXT NOT NULL DEFAULT '',
                original_path TEXT NOT NULL DEFAULT '',
                annotated_path TEXT NOT NULL DEFAULT '',
                detections TEXT NOT NULL DEFAULT '[]',
                total_defects INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'PASS',
                sahi_used INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.25,
                processing_time REAL NOT NULL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------- Users ----------------

def create_user(username: str, password_hash: str,
                db_path: Optional[Path] = None) -> int:
    """Insert a user. Raises sqlite3.IntegrityError if username taken."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip(), password_hash),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_user_by_username(username: str,
                         db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Fetch a user row by username (case-sensitive)."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int,
                   db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Fetch a user row by id."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------- Inspections ----------------

def save_inspection(user_id: int, image_name: str, original_path: str,
                    annotated_path: str, detections: list[dict[str, Any]],
                    status: str, sahi_used: bool, confidence: float,
                    processing_time: float,
                    db_path: Optional[Path] = None) -> int:
    """Save one inspection and return its id."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO inspections
               (user_id, image_name, original_path, annotated_path,
                detections, total_defects, status, sahi_used,
                confidence, processing_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, image_name, original_path, annotated_path,
             json.dumps(detections), len(detections), status,
             1 if sahi_used else 0, float(confidence), float(processing_time)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_inspections_by_user(user_id: int,
                            db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """List a user's inspections, newest first."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM inspections WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_inspection(inspection_id: int, user_id: int,
                   db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Fetch one inspection only if it belongs to the user."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ? AND user_id = ?",
            (inspection_id, user_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_inspection(inspection_id: int, user_id: int,
                      db_path: Optional[Path] = None) -> bool:
    """Delete one inspection owned by the user. Returns True if deleted."""
    db_path = _resolve(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ? AND user_id = ?",
            (inspection_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

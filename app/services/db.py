"""SQLite storage using stdlib sqlite3 (no ORM)."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row factory.

    Args:
        db_path: Path to the database file.

    Returns:
        SQLite connection object.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the results table if it does not exist.

    Args:
        db_path: Path to the database file.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                num_defects INTEGER NOT NULL DEFAULT 0,
                detections TEXT NOT NULL DEFAULT '[]',
                annotated_image TEXT NOT NULL DEFAULT '',
                method TEXT NOT NULL DEFAULT 'yolo',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_result(
    filename: str,
    verdict: str,
    detections: list[dict[str, Any]],
    annotated_image: str = "",
    method: str = "yolo",
    db_path: Path = DB_PATH,
) -> int:
    """Save one inspection result.

    Args:
        filename: Original upload filename.
        verdict: "Pass" or "Fail".
        detections: List of detection dicts.
        annotated_image: Saved annotated image filename/path.
        method: Inference method ("yolo" or "sahi").
        db_path: Path to the database file.

    Returns:
        Inserted row id.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO results (filename, verdict, num_defects, detections, annotated_image, method)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (filename, verdict, len(detections), json.dumps(detections), annotated_image, method),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_all(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Fetch all results newest-first.

    Args:
        db_path: Path to the database file.

    Returns:
        List of result dicts.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM results ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_by_id(result_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Fetch one result by id.

    Args:
        result_id: Row id.
        db_path: Path to the database file.

    Returns:
        Result dict or None.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

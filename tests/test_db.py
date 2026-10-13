"""Tests for database (users + inspections)."""

from pathlib import Path

from app import database as db


def test_user_roundtrip(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    uid = db.create_user("alice", "hash123", dbp)
    assert uid == 1
    assert db.get_user_by_username("alice", dbp)["id"] == 1
    assert db.get_user_by_id(1, dbp)["username"] == "alice"
    assert db.get_user_by_username("nobody", dbp) is None


def test_inspection_save_fetch_delete(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    uid = db.create_user("bob", "h", dbp)
    rid = db.save_inspection(uid, "a.png", "o.png", "a.png",
                             [{"class_name": "short"}], "FAIL", False,
                             0.25, 1.2, dbp)
    assert rid == 1
    rows = db.get_inspections_by_user(uid, dbp)
    assert len(rows) == 1 and rows[0]["status"] == "FAIL"
    assert db.get_inspection(rid, uid, dbp)["image_name"] == "a.png"
    assert db.get_inspection(rid, 999, dbp) is None  # ownership enforced
    assert db.delete_inspection(rid, uid, dbp) is True
    assert db.get_inspections_by_user(uid, dbp) == []

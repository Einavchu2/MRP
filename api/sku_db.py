"""
sku_db.py — SQLite storage for SKU Procurement Tracking, owned by the API.

Replaces outputs/sku_tracking.csv (kept as a backup, not deleted). A real
SQL database instead of read-whole-file/rewrite-whole-file: concurrent
edits to different rows are now genuinely safe, and editing one row no
longer means transferring the entire table over HTTP.

WAL mode + a fresh connection per call is the standard safe pattern for
SQLite under a threaded server (FastAPI runs sync handlers across a
threadpool) at this scale — no connection pooling needed for a table this
size.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sku_tracking.db")

COLUMNS = [
    "id", "item", "description", "supplier", "procurement_rep", "planner_code",
    "request_type", "request_date", "next_action_date", "priority", "status",
    "update_notes", "comments_procurement", "owner", "last_updated",
]


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    cols_sql = ", ".join(f'"{c}" TEXT' for c in COLUMNS if c != "id")
    conn = _connect()
    try:
        with conn:
            conn.execute(f'CREATE TABLE IF NOT EXISTS sku_tracking (id TEXT PRIMARY KEY, {cols_sql})')
            conn.execute(
                'CREATE TABLE IF NOT EXISTS user_filters '
                '(uid TEXT PRIMARY KEY, filters_json TEXT, updated_at TEXT)'
            )
    finally:
        conn.close()


def list_rows() -> list:
    conn = _connect()
    try:
        rows = conn.execute(f'SELECT {", ".join(COLUMNS)} FROM sku_tracking').fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_row(row: dict) -> dict:
    row = dict(row)
    row["id"] = row.get("id") or uuid.uuid4().hex[:10]
    row["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in COLUMNS:
        row.setdefault(c, "")
    placeholders = ", ".join(f":{c}" for c in COLUMNS)
    conn = _connect()
    try:
        with conn:
            conn.execute(
                f'INSERT INTO sku_tracking ({", ".join(COLUMNS)}) VALUES ({placeholders})',
                row,
            )
    finally:
        conn.close()
    return row


def update_row(row_id: str, fields: dict) -> bool:
    fields = {k: v for k, v in fields.items() if k in COLUMNS and k != "id"}
    fields["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_sql = ", ".join(f'"{c}" = :{c}' for c in fields)
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(f'UPDATE sku_tracking SET {set_sql} WHERE id = :id', {**fields, "id": row_id})
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_row(row_id: str) -> bool:
    conn = _connect()
    try:
        with conn:
            cur = conn.execute("DELETE FROM sku_tracking WHERE id = ?", (row_id,))
        return cur.rowcount > 0
    finally:
        conn.close()


def get_user_filters(uid: str) -> dict:
    """Last-saved SKU Tracking filter selection for one browser (identified by
    the mrp_uid cookie). Empty dict if this uid has never saved anything."""
    conn = _connect()
    try:
        row = conn.execute("SELECT filters_json FROM user_filters WHERE uid = ?", (uid,)).fetchone()
        return json.loads(row["filters_json"]) if row else {}
    finally:
        conn.close()


def save_user_filters(uid: str, filters: dict) -> None:
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO user_filters (uid, filters_json, updated_at) VALUES (:uid, :fj, :ts) "
                "ON CONFLICT(uid) DO UPDATE SET filters_json = :fj, updated_at = :ts",
                {
                    "uid": uid,
                    "fj": json.dumps(filters, ensure_ascii=False),
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
    finally:
        conn.close()


init_db()

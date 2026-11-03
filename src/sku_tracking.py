"""
sku_tracking.py — thin HTTP client for SKU Procurement Tracking.

The API (api/sku_db.py) owns the actual storage now — a real SQLite
database instead of outputs/sku_tracking.csv (kept as a migration backup,
not read at runtime anymore). Same pattern as src/db.py: this module just
calls the API and converts JSON to/from DataFrames/dicts.
"""
import os

import pandas as pd
import requests

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sku_tracking.csv")
SKU_TRACKING_FILE = _FILE  # legacy CSV — retained on disk as a backup, not read/written anymore

API_BASE_URL = os.environ.get("MRP_API_URL", "http://127.0.0.1:8502")

COLUMNS = [
    "id",
    "item",
    "description",
    "supplier",
    "procurement_rep",
    "planner_code",
    "request_type",
    "request_date",
    "next_action_date",
    "priority",
    "status",
    "update_notes",
    "comments_procurement",
    "owner",
    "last_updated",
]

STATUS_OPTIONS   = ["🟢 הושלם", "🟡 בטיפול", "🔴 טרם טופל"]
PRIORITY_OPTIONS = ["🔴 דחוף מאוד", "🔴 דחוף", "🟡 בינוני", "⚪ רגיל"]
REQUEST_TYPES    = [
    "הקדמת הזמנה", "הגדלת הזמנה", "הקדמת אספקה", "אישור אספקה",
    "בירור סטטוס", "עדכון PO", "חסר PD", "אין PD",
    "ביטול הזמנה", "קליטת מלאי", "החזרה ואספקת מלאי תקין", "הזמנה מתעכבת",
    "תיקון PD", "החזרת מלאי", "שינוי גודל הזמנה", "לא הוקמה הזמנה", "אחר",
]


def _request(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the data API at {API_BASE_URL} — is it running? "
            f"(start it with run_api.bat / run_api.sh)"
        ) from e
    return resp.json() if resp.content else None


def load_tracking() -> pd.DataFrame:
    """Load every SKU tracking row, normalized to COLUMNS."""
    rows = _request("GET", "/sku-tracking")
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS].fillna("") if not df.empty else pd.DataFrame(columns=COLUMNS)


def create_tracking_row(row: dict) -> dict:
    """Create one new tracking row. `row` need not include "id" or
    "last_updated" — both are assigned server-side."""
    return _request("POST", "/sku-tracking", json=row)


def update_tracking_row(row_id: str, fields: dict) -> None:
    """Update only the given fields of one existing row by id."""
    _request("PATCH", f"/sku-tracking/{row_id}", json=fields)


def delete_tracking_row(row_id: str) -> None:
    _request("DELETE", f"/sku-tracking/{row_id}")


def load_user_filters(uid: str) -> dict:
    """Last-saved SKU Tracking filter selection for this browser. Fails soft
    (empty dict) if the API is unreachable — a convenience feature must never
    break the page."""
    try:
        return _request("GET", f"/user-filters/{uid}") or {}
    except RuntimeError:
        return {}


def save_user_filters(uid: str, filters: dict) -> None:
    try:
        _request("PUT", f"/user-filters/{uid}", json=filters)
    except RuntimeError:
        pass

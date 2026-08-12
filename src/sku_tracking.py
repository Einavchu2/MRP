"""
sku_tracking.py — SKU-level procurement follow-up tracker.
Persists one row per open/closed procurement issue for a SKU
(status, priority, request/follow-up dates, supplier, running notes).
"""
import os
import uuid
import pandas as pd

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sku_tracking.csv")
SKU_TRACKING_FILE = _FILE

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
    "owner",
    "last_updated",
]

STATUS_OPTIONS   = ["🟢 הושלם", "🟡 בטיפול", "🔴 טרם טופל"]
PRIORITY_OPTIONS = ["🔴 דחוף", "🟡 בינוני", "⚪ רגיל"]
REQUEST_TYPES    = [
    "הקדמת הזמנה", "הגדלת הזמנה", "הקדמת אספקה", "אישור אספקה",
    "בירור סטטוס", "עדכון PO", "חסר PD", "אחר",
]


def _ensure_file() -> None:
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    if not os.path.exists(_FILE):
        pd.DataFrame(columns=COLUMNS).to_csv(_FILE, index=False)


def load_tracking() -> pd.DataFrame:
    """Load the SKU tracking CSV, normalized to COLUMNS."""
    _ensure_file()
    df = pd.read_csv(_FILE, dtype=str).fillna("")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def save_tracking(df: pd.DataFrame) -> None:
    """Persist the full tracking table back to CSV."""
    _ensure_file()
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS]
    df.to_csv(_FILE, index=False)


def add_entry(
    item: str,
    description: str = "",
    supplier: str = "",
    procurement_rep: str = "",
    planner_code: str = "",
    request_type: str = "",
    request_date: str = "",
    next_action_date: str = "",
    priority: str = "⚪ רגיל",
    status: str = "🔴 טרם טופל",
    update_notes: str = "",
    owner: str = "",
) -> pd.DataFrame:
    """Append a new SKU tracking row and return the updated table."""
    df = load_tracking()
    row = {
        "id":               uuid.uuid4().hex[:10],
        "item":              str(item).strip(),
        "description":       str(description).strip(),
        "supplier":          str(supplier).strip(),
        "procurement_rep":   str(procurement_rep).strip(),
        "planner_code":      str(planner_code).strip(),
        "request_type":      str(request_type).strip(),
        "request_date":      str(request_date).strip(),
        "next_action_date":  str(next_action_date).strip(),
        "priority":          str(priority).strip(),
        "status":            str(status).strip(),
        "update_notes":      str(update_notes).strip(),
        "owner":             str(owner).strip(),
        "last_updated":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_tracking(df)
    return load_tracking()

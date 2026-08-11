"""
master_data.py — Load and serve mrp_current.csv master data.
Provides per-item: Safety Stock, Lead Time, Shelf Life, Max Inventory, Planner Code.
"""
import os
import pandas as pd
import numpy as np

_MASTER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mrp_current.csv")

_COL_MAP = {
    "item":                         "item",
    "list_price":                   "list_price",
    "ABC class":                    "abc_class",
    "LT_current":                   "lead_time",          # months
    "safety_stock (cover months)":  "safety_stock",       # months of coverage
    "MAX INVENTORY":                "max_inventory",       # months
    "SL_current":                   "shelf_life",          # months
    "planner_code":                 "planner_code",
}

_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]


def _read_csv_with_fallback(path: str, **kwargs) -> pd.DataFrame:
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    # Last resort: allow any byte through with latin-1
    return pd.read_csv(path, encoding="latin-1", **kwargs)


def load_master() -> pd.DataFrame:
    """Load mrp_current.csv and return a clean master DataFrame indexed by item."""
    df = _read_csv_with_fallback(_MASTER_FILE, dtype={"item": str})
    df = df.rename(columns=_COL_MAP)
    df["item"] = df["item"].astype(str).str.strip()

    # Numeric conversions — treat 'TBD' and blanks as NaN
    for col in ["lead_time", "safety_stock", "max_inventory", "shelf_life", "list_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Defaults for missing values
    df["safety_stock"]  = df["safety_stock"].fillna(7.0)    # default 7 months
    df["lead_time"]     = df["lead_time"].fillna(3.0)       # default 3 months
    df["shelf_life"]    = df["shelf_life"].fillna(np.nan)   # leave NaN = unknown
    df["max_inventory"] = df["max_inventory"].fillna(15.0)  # default 15 months

    return df.set_index("item")


def get_item_params(item: str, master: pd.DataFrame) -> dict:
    """Return master-data params for one item, with sensible defaults."""
    if item in master.index:
        row = master.loc[item]
        return {
            "safety_stock":  float(row.get("safety_stock", 7.0) or 7.0),
            "lead_time":     float(row.get("lead_time",    3.0) or 3.0),
            "shelf_life":    float(row.get("shelf_life",   np.nan)),
            "max_inventory": float(row.get("max_inventory",15.0) or 15.0),
            "planner_code":  str(row.get("planner_code",  "Unknown")),
            "list_price":    float(row.get("list_price",   0.0) or 0.0),
            "abc_class":     str(row.get("abc_class",      "")),
        }
    return {
        "safety_stock":  7.0,
        "lead_time":     3.0,
        "shelf_life":    np.nan,
        "max_inventory": 15.0,
        "planner_code":  "Unknown",
        "list_price":    0.0,
        "abc_class":     "",
    }


def get_master_map(master: pd.DataFrame) -> dict:
    """Convert master DataFrame to a dict keyed by item string for fast lookup."""
    result = {}
    for item, row in master.iterrows():
        result[str(item)] = {
            "safety_stock":  float(row.get("safety_stock",  7.0) or 7.0),
            "lead_time":     float(row.get("lead_time",     3.0) or 3.0),
            "shelf_life":    float(row.get("shelf_life",    np.nan)),
            "max_inventory": float(row.get("max_inventory", 15.0) or 15.0),
            "planner_code":  str(row.get("planner_code",   "Unknown")),
            "list_price":    float(row.get("list_price",    0.0) or 0.0),
            "abc_class":     str(row.get("abc_class",       "")),
        }
    return result

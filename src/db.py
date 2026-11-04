"""
db.py — thin HTTP client for the MRP data API (see api/).

The API owns the actual SQL Server connection (pooled, credentials in its
own .env) — this module just calls it and converts JSON back into the same
DataFrames these functions always returned, behind the same three function
signatures, so nothing that imports from here needs to change.

Requires the API running (run_api.bat / run_api.sh) — see MRP_API_URL below.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("MRP_API_URL", "http://127.0.0.1:8502")


def _get(path: str, **params):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the data API at {API_BASE_URL} — is it running? "
            f"(start it with run_api.bat / run_api.sh)"
        ) from e
    return resp.json()


@st.cache_data(ttl=300, show_spinner="Loading BOM from ORACLE_DWH…")
def load_full_bom(root_product: str) -> pd.DataFrame:
    return pd.DataFrame(_get("/bom", root_product=root_product))


@st.cache_data(ttl=300, show_spinner="Loading transactions & ASCP from ORACLE_DWH…")
def load_dwh_data() -> dict:
    data = _get("/dwh-data")
    return {
        "transactions": pd.DataFrame(data["transactions"]),
        "ascp": pd.DataFrame(data["ascp"]),
        "uom": pd.DataFrame(data["uom"]),
    }


@st.cache_data(ttl=300, show_spinner="Loading inventory history from ORACLE_DWH…")
def load_inventory_history(days: int = 365) -> dict:
    data = _get("/inventory-history", days=days)
    return {k: pd.DataFrame(v) for k, v in data.items()}


@st.cache_data(ttl=300, show_spinner="Loading substitute items…")
def load_substitutes() -> pd.DataFrame:
    """
    Load Main_Item → Substitute_Item mapping from BTG_Substitute_Items.
    Returns DataFrame with columns: Main_Item, Substitute_Item
    """
    try:
        return pd.DataFrame(_get("/substitutes"))
    except Exception:
        return pd.DataFrame(columns=["Main_Item", "Substitute_Item"])


@st.cache_data(ttl=300, show_spinner="Loading BOM for RM analysis…")
def load_rm_bom() -> pd.DataFrame:
    return pd.DataFrame(_get("/rm/bom"))


@st.cache_data(ttl=300, show_spinner="Loading production plan & WIP history…")
def load_rm_plan() -> dict:
    return {k: pd.DataFrame(v) for k, v in _get("/rm/plan").items()}


@st.cache_data(ttl=300, show_spinner="Loading prices, on-hand & ASCP for the raw materials…")
def load_rm_items(items: tuple) -> dict:
    return {k: pd.DataFrame(v) for k, v in _get("/rm/items", items=",".join(items)).items()}

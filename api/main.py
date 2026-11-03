"""
main.py — the MRP data API. Owns every SQL Server connection for the app,
plus SKU Tracking's own SQLite storage; Streamlit (src/db.py,
src/sku_tracking.py) talks to this over HTTP instead of touching either
directly. Run with run_api.bat / run_api.sh, bound to 127.0.0.1 only —
nothing outside this machine needs to reach it.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import queries, sku_db

app = FastAPI(title="MRP Data API")

# The unified SKU Tracking table's JS calls this API through the IIS reverse
# proxy (port 8503) from a page served by Streamlit (port 8501) — different
# ports means different origins to the browser, so CORS is required even
# though both ultimately point at this same server. Scoped to "any host on
# port 8501" rather than "*" so it stays bounded to wherever Streamlit itself
# is actually reachable.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://[^/]+:8501$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/bom")
def bom(root_product: str):
    return queries.get_bom(root_product)


@app.get("/dwh-data")
def dwh_data():
    return queries.get_dwh_data()


@app.get("/inventory-history")
def inventory_history(days: int = 365):
    return queries.get_inventory_history(days)


@app.get("/substitutes")
def substitutes():
    return queries.get_substitutes()


# ── SKU Procurement Tracking (SQLite) ──────────────────────────────

@app.get("/sku-tracking")
def sku_tracking_list():
    return sku_db.list_rows()


@app.post("/sku-tracking")
def sku_tracking_create(row: dict):
    return sku_db.create_row(row)


@app.patch("/sku-tracking/{row_id}")
def sku_tracking_update(row_id: str, fields: dict):
    if not sku_db.update_row(row_id, fields):
        raise HTTPException(status_code=404, detail=f"No SKU tracking row with id={row_id}")
    return {"status": "ok"}


@app.delete("/sku-tracking/{row_id}")
def sku_tracking_delete(row_id: str):
    if not sku_db.delete_row(row_id):
        raise HTTPException(status_code=404, detail=f"No SKU tracking row with id={row_id}")
    return {"status": "ok"}


# ── Per-browser SKU Tracking filter memory ─────────────────────────

@app.get("/user-filters/{uid}")
def user_filters_get(uid: str):
    return sku_db.get_user_filters(uid)


@app.put("/user-filters/{uid}")
def user_filters_put(uid: str, filters: dict):
    sku_db.save_user_filters(uid, filters)
    return {"status": "ok"}

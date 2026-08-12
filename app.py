"""
app.py  –  MRP / BOM Simulation Dashboard
Run:  streamlit run app.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess, numpy as np, pandas as pd, streamlit as st

def _ensure_packages():
    for pkg, imp in [("plotly","plotly"), ("scikit-learn","sklearn"), ("openai","openai")]:
        try: __import__(imp)
        except ImportError: subprocess.run([sys.executable,"-m","pip","install",pkg,"--quiet","--break-system-packages"], check=False)
_ensure_packages()

from src.db import load_full_bom, load_dwh_data, load_substitutes
from src.data_prep import enrich_bom, build_mrp_pivot
from src.simulation import SimulationConfig, run_simulation, _compute_inv_cover as _compute_inv_cover_orig
from src.master_data import load_master, get_master_map
from src.sku_tracking import (
    load_tracking as load_sku_tracking,
    save_tracking as save_sku_tracking,
    add_entry as add_sku_tracking_entry,
    STATUS_OPTIONS as SKU_STATUS_OPTIONS,
    PRIORITY_OPTIONS as SKU_PRIORITY_OPTIONS,
    REQUEST_TYPES as SKU_REQUEST_TYPES,
)

def _apply_substitutes(pivot_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Main_Item → Substitute_Item) pair:
    - Find the Substitute_Item On Hand rows in pivot
    - Add their quantities to the Main_Item On Hand rows
    - Remove the Substitute On Hand rows (absorbed into Main)
    Uses index lists (not boolean masks) to avoid index-mismatch after concat.
    """
    if sub_df.empty:
        return pivot_df

    pivot = pivot_df.copy().reset_index(drop=True)
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    rows_to_drop = []

    for _, sub_row in sub_df.iterrows():
        main_item = str(sub_row["Main_Item"]).strip()
        sub_item  = str(sub_row["Substitute_Item"]).strip()

        # Get integer indices of substitute On Hand rows
        sub_oh_idx = pivot.index[
            (pivot["item"] == sub_item) &
            (pivot["ORDER_TYPE_FINAL"].str.contains("on hand|3.on hand", case=False, na=False))
        ].tolist()

        if not sub_oh_idx:
            continue

        for m in month_cols:
            sub_val = pd.to_numeric(pivot.loc[sub_oh_idx, m], errors="coerce").sum()
            if pd.isna(sub_val) or sub_val == 0:
                continue

            # Get integer indices of main item On Hand rows
            main_oh_idx = pivot.index[
                (pivot["item"] == main_item) &
                (pivot["ORDER_TYPE_FINAL"].str.contains("on hand|3.on hand", case=False, na=False))
            ].tolist()

            if main_oh_idx:
                cur = pd.to_numeric(pivot.at[main_oh_idx[0], m], errors="coerce")
                pivot.at[main_oh_idx[0], m] = (cur if not pd.isna(cur) else 0) + sub_val
            else:
                # No On Hand row for main item — create one
                new_row = {c: np.nan for c in pivot.columns}
                new_row["item"]             = main_item
                new_row["ORDER_TYPE_FINAL"] = "3.On Hand"
                new_row[m]                  = sub_val
                pivot = pd.concat([pivot, pd.DataFrame([new_row])],
                                   ignore_index=True)

        # Mark substitute On Hand rows as SUBSTITUTE_OH instead of removing them
        # This keeps them visible in the pivot with special highlighting
        for idx in sub_oh_idx:
            pivot.at[idx, "ORDER_TYPE_FINAL"] = "SUBSTITUTE_OH"
            # Store which main item this substitute belongs to
            pivot.at[idx, "description"] = (
                str(pivot.at[idx, "description"]) + f" [→{main_item}]"
                if "→" not in str(pivot.at[idx, "description"])
                else str(pivot.at[idx, "description"])
            )

    return pivot


def _compute_inv_cover(pivot, master_map=None):
    """Safe wrapper — works with both old (1 arg) and new (2 arg) simulation.py."""
    import inspect
    sig = inspect.signature(_compute_inv_cover_orig)
    if len(sig.parameters) >= 2 and master_map is not None:
        return _compute_inv_cover_orig(pivot, master_map)
    return _compute_inv_cover_orig(pivot)

# ══════════════════════════════════════════════════════════════
# PO RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════

COVER_TRIGGER   = 7    # months: trigger a PO recommendation below this
COVER_TARGET    = 15   # months: fill inventory up to this coverage level


def compute_po_recommendations(pivot_df: pd.DataFrame, month_cols: list, master_map: dict = None) -> pd.DataFrame:
    """
    PO Recommendation logic:

    1. LEAD TIME constraint:
       PO cannot be placed before first_active_month + LT.
       LT is measured from the first month that appears in the pivot.

    2. EXCEPTION flag:
       If cover < 80% of SS at the trigger point → row type = PO_EXCEPTION
       (displayed in a different color to highlight urgency)

    3. COVER coloring:
       Cover cells colored red when cover < per-item SS (not fixed 7).

    4. COVER_MONTHS_UPDATED:
       Forward-propagated coverage after all PO injections.
    """
    master_map = master_map or {}
    if not month_cols:
        return pivot_df

    keep_mask = ~pivot_df["ORDER_TYPE_FINAL"].isin(
        ["PO_RECOMMENDATION", "PO_EXCEPTION", "COVER_MONTHS_UPDATED"]
    )
    pivot_df = pivot_df[keep_mask].copy()

    new_rows: list[dict] = []

    # First month in the pivot = time-zero for LT calculation
    first_pivot_month = month_cols[0]

    for item, grp in pivot_df.groupby("item", sort=False):
        description = grp["description"].dropna().iloc[0] if not grp["description"].dropna().empty else ""

        _mp         = master_map.get(str(item), {})
        item_ss     = float(_mp.get("safety_stock",  COVER_TRIGGER) or COVER_TRIGGER)
        item_target = float(_mp.get("max_inventory",  COVER_TARGET)  or COVER_TARGET)
        item_lt     = float(_mp.get("lead_time",      0)             or 0)
        item_sl     = float(_mp.get("shelf_life",     9999)          or 9999)

        # Exception threshold: cover < 80% of SS
        exception_threshold = item_ss * 0.80

        # ── LT constraint ──────────────────────────────────────────────────
        # LT means the supplier needs LT months from NOW (first On Hand month)
        # to commit/deliver. So earliest PO = first_oh_month_idx + LT months.
        # Exception: if cover < 80% SS → allow earlier (emergency) order.
        lt_months_int = int(round(item_lt))

        # Find first On Hand month for this item
        oh_rows_item = grp[grp["ORDER_TYPE_FINAL"].str.contains("on hand|3.on", case=False, na=False)]
        first_oh_idx = 0   # default = first month in pivot
        for _mi, _m in enumerate(month_cols):
            if not oh_rows_item.empty and _m in oh_rows_item.columns:
                _v = pd.to_numeric(oh_rows_item[_m].iloc[0], errors="coerce")
                if pd.notna(_v) and _v > 0:
                    first_oh_idx = _mi
                    break

        # Earliest possible PO = first_oh_idx + LT
        first_po_idx = min(first_oh_idx + lt_months_int, len(month_cols) - 1)

        cover_row = grp[grp["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
        if cover_row.empty:
            continue
        inv_row = grp[grp["ORDER_TYPE_FINAL"] == "INV"]

        n          = len(month_cols)
        cover_vals = np.full(n, np.nan)
        inv_vals   = np.full(n, np.nan)

        for i, m in enumerate(month_cols):
            if m in cover_row.columns:
                cover_vals[i] = pd.to_numeric(cover_row[m].iloc[0], errors="coerce")
            if not inv_row.empty and m in inv_row.columns:
                inv_vals[i]   = pd.to_numeric(inv_row[m].iloc[0], errors="coerce")

        # Demand rate: INV[t] / cover[t]
        # Note: cover/inv may have NaN gaps → forward-fill demand rate
        demand_rate = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(inv_vals[i]) and not np.isnan(cover_vals[i]) and cover_vals[i] > 0:
                demand_rate[i] = inv_vals[i] / cover_vals[i]
        # Forward-fill: NaN months should use last known demand rate (not 0)
        last_rate = np.nan
        for i in range(n):
            if not np.isnan(demand_rate[i]):
                last_rate = demand_rate[i]
            elif not np.isnan(last_rate):
                demand_rate[i] = last_rate
        # Back-fill: if first months have no rate, use first available
        first_rate = next((demand_rate[i] for i in range(n) if not np.isnan(demand_rate[i])), 0.0)
        for i in range(n):
            if np.isnan(demand_rate[i]):
                demand_rate[i] = first_rate

        po_rec        = np.zeros(n)
        po_is_except  = np.zeros(n, dtype=bool)   # True = exception PO
        updated_cover = cover_vals.copy()

        # proj_inv: use ACTUAL INV values where available.
        # Only estimate (decay) where INV row has no data.
        proj_inv = inv_vals.copy()
        for i in range(1, n):
            if np.isnan(proj_inv[i]):
                if not np.isnan(proj_inv[i - 1]):
                    decay = demand_rate[i - 1] if not np.isnan(demand_rate[i - 1]) else 0
                    proj_inv[i] = max(proj_inv[i - 1] - decay, 0)

        # Store original INV for resync after PO injection
        original_inv = proj_inv.copy()

        for i in range(n):
            cur_cover = updated_cover[i]
            if np.isnan(cur_cover):
                continue

            if cur_cover < item_ss:
                rate = demand_rate[i] if not np.isnan(demand_rate[i]) else 0
                if rate > 0:
                    needed_inv  = rate * item_target
                    current_inv = proj_inv[i] if not np.isnan(proj_inv[i]) else 0
                    order_qty   = max(needed_inv - current_inv, 0)
                    if item_sl < 9999:
                        order_qty = min(order_qty, rate * item_sl)
                else:
                    order_qty = 0

                if order_qty > 0:
                    # ── LT constraint ──────────────────────────────────────
                    # Normal PO: cannot be placed before first_po_idx
                    #            (= On Hand month + LT months)
                    # Exception: cover < 80% SS → can be placed immediately
                    #            (emergency order despite LT)

                    is_exception = cur_cover < exception_threshold

                    if is_exception:
                        # Emergency: place PO as early as possible (at depletion or now)
                        place_idx = i
                    else:
                        # Normal: must wait until first_po_idx
                        # If depletion is before first_po_idx → PO at first_po_idx
                        place_idx = max(i, first_po_idx)
                        place_idx = min(place_idx, len(month_cols) - 1)

                    po_rec[place_idx] = round(order_qty, 0)
                    po_is_except[place_idx] = is_exception

                    # ── Re-simulate from place_idx forward ────────────────────
                    # IMPORTANT: re-sync proj_inv to original_inv at each step
                    # to avoid accumulated drift from NaN forecast gaps.
                    # PO is added on top of the real INV at place_idx.
                    cumulative_po = order_qty  # total injected PO carried forward
                    for j in range(place_idx, n):
                        # Base: original INV at this month (accounts for real POs in data)
                        base_inv = original_inv[j] if not np.isnan(original_inv[j]) else (
                            max((original_inv[j-1] if j>0 and not np.isnan(original_inv[j-1]) else 0)
                                - (demand_rate[j-1] if not np.isnan(demand_rate[j-1]) else 0), 0)
                        )
                        proj_inv[j]      = base_inv + cumulative_po
                        r                = demand_rate[j] if not np.isnan(demand_rate[j]) else 0
                        updated_cover[j] = (proj_inv[j] / r) if r > 0 else np.nan
                        # Decay the injected portion by this month's demand
                        cumulative_po = max(cumulative_po - r, 0)

        base    = {"item": item, "description": description}
        po_row  = {**base, "ORDER_TYPE_FINAL": "PO_RECOMMENDATION"}
        exc_row = {**base, "ORDER_TYPE_FINAL": "PO_EXCEPTION"}
        upd_row = {**base, "ORDER_TYPE_FINAL": "COVER_MONTHS_UPDATED"}

        has_exceptions = False
        for i, m in enumerate(month_cols):
            qty = po_rec[i] if po_rec[i] > 0 else np.nan
            if qty and not np.isnan(qty):
                if po_is_except[i]:
                    exc_row[m] = qty
                    po_row[m]  = np.nan
                    has_exceptions = True
                else:
                    po_row[m]  = qty
                    exc_row[m] = np.nan
            else:
                po_row[m]  = np.nan
                exc_row[m] = np.nan
            upd_row[m] = updated_cover[i] if not np.isnan(updated_cover[i]) else np.nan

        new_rows.append(po_row)
        if has_exceptions:
            new_rows.append(exc_row)
        new_rows.append(upd_row)

    if not new_rows:
        return pivot_df

    new_df = pd.DataFrame(new_rows)
    return pd.concat([pivot_df, new_df], ignore_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MRP Simulation",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Deploy button
st.markdown(
    """
    <style>
        .stDeployButton {display: none !important;}
        [data-testid="stDeployButton"] {display: none !important;}
        #MainMenu {visibility: hidden !important;}
        header [data-testid="stToolbar"] {visibility: hidden !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

for key in ["full_bom", "pivot_df", "original_pivot", "sim_result", "data_loaded", "load_error", "changed_items", "current_parent", "sub_df", "sub_items_map", "app_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.app_mode is None:
    st.session_state.app_mode = "Simulation"

# ══════════════════════════════════════════════════════════════
# AUTO-LOAD DATA ON PAGE START
# ══════════════════════════════════════════════════════════════

_default_parent = "6306683000"

if st.session_state.app_mode == "Simulation" and not st.session_state.data_loaded:
    with st.spinner("🔄 Loading data from database…"):
        try:
            # ── 1. Load master data FIRST (needed for SS/LT/SL thresholds) ──
            try:
                _master_df  = load_master()
                _master_map = get_master_map(_master_df)
            except Exception:
                _master_df  = pd.DataFrame()
                _master_map = {}
            st.session_state.master_df  = _master_df
            st.session_state.master_map = _master_map

            # ── 2. Load ERP data and build pivot ──
            dwh = load_dwh_data()
            raw_bom = load_full_bom(_default_parent)
            full_bom = enrich_bom(raw_bom, dwh["uom"])
            pivot_df = build_mrp_pivot(dwh["transactions"], dwh["ascp"])

            # ── 2b. Apply substitute items (merge Sub On Hand → Main On Hand) ──
            sub_df = load_substitutes()
            st.session_state.sub_df = sub_df
            if not sub_df.empty:
                pivot_df = _apply_substitutes(pivot_df, sub_df)
                # Build reverse map: {sub_sku: main_sku} for highlighting
                st.session_state.sub_items_map = dict(
                    zip(sub_df["Substitute_Item"], sub_df["Main_Item"])
                )
            else:
                st.session_state.sub_items_map = {}

            # ── 3. Compute INV/COVER with per-item safety stock ──
            pivot_df = _compute_inv_cover(pivot_df, _master_map)

            # ── 4. Compute PO recommendations with per-item LT/SL/SS/MAX ──
            _month_cols_init = sorted([c for c in pivot_df.columns if str(c).startswith("202")])
            pivot_df = compute_po_recommendations(pivot_df, _month_cols_init, master_map=_master_map)

            st.session_state.full_bom = full_bom
            st.session_state.pivot_df = pivot_df
            st.session_state.original_pivot = pivot_df.copy()
            st.session_state.current_parent = _default_parent
            st.session_state.data_loaded = True
            st.session_state.load_error = None

        except Exception as e:
            st.session_state.load_error = str(e)
            st.session_state.data_loaded = False

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

if st.session_state.app_mode == "SKU Tracking":
    st.title("🔎 SKU Procurement Tracking")
    st.caption("Track open procurement issues per SKU — status, priority, next follow-up date, supplier and running notes.")
else:
    st.title("🏭 MRP / BOM Simulation Engine")
    st.caption("Supply Chain · What-If Analysis")

if st.session_state.app_mode == "Simulation" and st.session_state.load_error:
    st.error(f"❌ Failed to load data from database:\n\n{st.session_state.load_error}")
    if st.button("🔄 Retry"):
        st.session_state.data_loaded = None
        st.rerun()
    st.stop()

if st.session_state.app_mode == "SKU Tracking":
    tracking_df = load_sku_tracking()

    # ── Item picker sourced from master data (auto-fills description/planner) ──
    _mm_track = getattr(st.session_state, "master_map", {}) or {}
    _pivot_track = getattr(st.session_state, "pivot_df", None)
    if _mm_track:
        track_item_options = sorted(_mm_track.keys())
    elif _pivot_track is not None:
        track_item_options = sorted(_pivot_track["item"].dropna().unique().tolist())
    else:
        track_item_options = []

    st.subheader("➕ Open a new tracking item")
    # SKU picker lives outside the form so description/planner auto-fill live when it changes.
    if track_item_options:
        new_item = st.selectbox("מק\"ט · SKU", options=track_item_options, key="new_track_item")
    else:
        new_item = st.text_input("מק\"ט · SKU", value="", key="new_track_item")
    mp = _mm_track.get(str(new_item), {}) if new_item else {}
    desc_default = ""
    if new_item and _pivot_track is not None:
        _d = _pivot_track[_pivot_track["item"] == new_item]["description"].dropna()
        desc_default = _d.iloc[0] if not _d.empty else ""

    with st.form("create_sku_tracking_form"):
        fc1, fc2 = st.columns([2, 2])
        with fc1:
            new_description = st.text_input("תיאור · Description", value=desc_default)
            new_supplier = st.text_input("ספק · Supplier", value="")
            new_planner = st.text_input("פלנר · Planner code", value=str(mp.get("planner_code", "")))
        with fc2:
            new_request_type = st.selectbox("סוג הבקשה · Request type", SKU_REQUEST_TYPES)
            new_request_date = st.date_input("תאריך פנייה · Request date", value=pd.Timestamp.now().date())
            new_next_action = st.date_input("מועד הבא לטיפול · Next follow-up date", value=pd.Timestamp.now().date())
            new_priority = st.selectbox("תעדוף · Priority", SKU_PRIORITY_OPTIONS)
        new_owner = st.text_input("אחראי · Owner", value="")
        new_notes = st.text_area("עדכון · Update / notes", value="", height=80)
        create_clicked = st.form_submit_button("➕ Add tracking item")

    if create_clicked:
        if not new_item:
            st.error("SKU is required.")
        else:
            add_sku_tracking_entry(
                item=new_item,
                description=new_description,
                supplier=new_supplier,
                planner_code=new_planner,
                request_type=new_request_type,
                request_date=str(new_request_date),
                next_action_date=str(new_next_action),
                priority=new_priority,
                status="🔴 טרם טופל",
                update_notes=new_notes,
                owner=new_owner,
            )
            st.success(f"✅ Tracking item added for {new_item}.")
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Tracking table")
    tracking_df = load_sku_tracking()

    if tracking_df.empty:
        st.info("No SKU tracking items yet — add one above.")
    else:
        n_open     = int((tracking_df["status"] == "🔴 טרם טופל").sum())
        n_progress = int((tracking_df["status"] == "🟡 בטיפול").sum())
        n_done     = int((tracking_df["status"] == "🟢 הושלם").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("🔴 טרם טופל", n_open)
        k2.metric("🟡 בטיפול", n_progress)
        k3.metric("🟢 הושלם", n_done)

        # ── Filters (for the read-only colored view below) ──────────
        tf1, tf2, tf3 = st.columns(3)
        with tf1:
            status_filter = st.multiselect("Filter by status", SKU_STATUS_OPTIONS, default=[])
        with tf2:
            planner_opts = sorted([p for p in tracking_df["planner_code"].unique().tolist() if p])
            planner_filter = st.multiselect("Filter by planner code", planner_opts, default=[])
        with tf3:
            item_filter = st.multiselect("Filter by SKU", sorted(tracking_df["item"].unique().tolist()), default=[])

        view_df = tracking_df.copy()
        if status_filter:  view_df = view_df[view_df["status"].isin(status_filter)]
        if planner_filter: view_df = view_df[view_df["planner_code"].isin(planner_filter)]
        if item_filter:    view_df = view_df[view_df["item"].isin(item_filter)]

        def _status_color(v):
            if "הושלם" in str(v):    return "background-color:#dcfce7;color:#14532d;font-weight:600"
            if "בטיפול" in str(v):   return "background-color:#fef9c3;color:#713f12;font-weight:600"
            if "טרם טופל" in str(v): return "background-color:#fee2e2;color:#991b1b;font-weight:600"
            return ""
        def _priority_color(v):
            if "דחוף" in str(v):   return "background-color:#fee2e2;color:#991b1b;font-weight:600"
            if "בינוני" in str(v): return "background-color:#fef9c3;color:#713f12"
            return ""

        display_cols = ["item","description","supplier","planner_code","request_type",
                         "request_date","next_action_date","priority","status","update_notes","owner"]
        st.dataframe(
            view_df[display_cols].style.map(_status_color, subset=["status"]).map(_priority_color, subset=["priority"]),
            use_container_width=True, height=min(500, max(120, len(view_df) * 38 + 40)),
        )

        st.markdown("---")
        st.subheader("✏️ Edit / add / remove rows")
        st.caption("Edit any cell, add rows at the bottom, or select a row and press delete. Click **💾 Save changes** to persist.")
        edited_df = st.data_editor(
            tracking_df[display_cols],
            column_config={
                "item":             st.column_config.TextColumn("מק\"ט"),
                "description":      st.column_config.TextColumn("תיאור", width="medium"),
                "supplier":         st.column_config.TextColumn("ספק"),
                "planner_code":     st.column_config.TextColumn("פלנר"),
                "request_type":     st.column_config.SelectboxColumn("סוג הבקשה", options=SKU_REQUEST_TYPES),
                "request_date":     st.column_config.TextColumn("תאריך פנייה"),
                "next_action_date": st.column_config.TextColumn("מועד הבא לטיפול"),
                "priority":         st.column_config.SelectboxColumn("תעדוף", options=SKU_PRIORITY_OPTIONS),
                "status":           st.column_config.SelectboxColumn("סטטוס רכש", options=SKU_STATUS_OPTIONS),
                "update_notes":     st.column_config.TextColumn("עדכון", width="large"),
                "owner":            st.column_config.TextColumn("אחראי"),
            },
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            key="sku_tracking_editor",
            height=min(600, max(200, len(tracking_df) * 38 + 60)),
        )

        if st.button("💾 Save changes"):
            import uuid as _uuid
            to_save = edited_df.copy().reset_index(drop=True)
            # Preserve stable ids for rows that already existed (by position); new/reordered
            # rows get a fresh id — fine for this lightweight tracker (no cross-table joins on id).
            ids = [tracking_df["id"].iloc[i] if i < len(tracking_df) else _uuid.uuid4().hex[:10] for i in range(len(to_save))]
            to_save["id"] = ids
            to_save["last_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            save_sku_tracking(to_save)
            st.success("✅ Saved.")
            st.rerun()

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            tracking_df.to_excel(w, index=False)
        st.download_button("📥 Download tracking table", data=buf.getvalue(), file_name="sku_tracking.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.stop()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Application Settings")
    app_mode = st.selectbox(
        "App Mode",
        ["Simulation", "SKU Tracking"],
        index=["Simulation", "SKU Tracking"].index(st.session_state.app_mode or "Simulation"),
    )
    st.session_state.app_mode = app_mode
    st.divider()

    # ── Substitute Items Info ──────────────────────────────────
    if hasattr(st.session_state, "sub_df") and st.session_state.sub_df is not None:
        sub_df = st.session_state.sub_df
        if not sub_df.empty:
            with st.expander(f"🔄 Substitute Items ({len(sub_df)})", expanded=False):
                st.caption("On Hand inventory of Substitute SKUs is merged into Primary SKU.")
                st.dataframe(sub_df, use_container_width=True, height=min(300, len(sub_df)*36+40))
        else:
            st.sidebar.caption("ℹ️ No substitute items found.")

    if st.session_state.data_loaded:
        st.success(
            f"✅ Data loaded\n\n"
            f"- BOM: {len(st.session_state.full_bom):,} rows\n"
            f"- Pivot: {len(st.session_state.pivot_df):,} rows"
        )
        if "BOM_CONV" in st.session_state.full_bom.columns:
            sample = st.session_state.full_bom["BOM_CONV"].dropna()
            st.caption(f"BOM_CONV range: {sample.min():.4f} – {sample.max():.4f}")
    st.divider()

    parent_product = st.text_input(
        "Parent Product",
        value=st.session_state.current_parent or "6306683000",
        help="Root SKU — changing this reloads the BOM automatically",
    )

    if parent_product and parent_product != st.session_state.current_parent:
        with st.spinner(f"Loading BOM for {parent_product}…"):
            try:
                dwh = load_dwh_data()
                raw_bom = load_full_bom(parent_product)
                full_bom = enrich_bom(raw_bom, dwh["uom"])
                st.session_state.full_bom = full_bom
                st.session_state.current_parent = parent_product
                st.session_state.sim_result = None
                st.session_state.changed_items = None
                st.success(f"✅ BOM reloaded for {parent_product}")
            except Exception as e:
                st.error(f"BOM load error: {e}")

    action = st.radio("Action", ["ADD", "REMOVE"], horizontal=True)

    pivot = st.session_state.pivot_df
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")]) if pivot is not None else []
    if month_cols:
        month = st.selectbox("Forecast Month", month_cols, index=len(month_cols) - 1)
    else:
        month = st.text_input("Forecast Month (YYYY-MM)", value="2026-08")

    production_change = st.number_input(
        "Production Qty Change",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    search_mode = st.selectbox(
        "Search Mode",
        ["PARTIAL", "DEEP", "DIRECT"],
        help=(
            "PARTIAL: recursive, skips excluded products\n"
            "DEEP: full recursive explosion\n"
            "DIRECT: only immediate children"
        ),
    )

    exclude_text = st.text_area(
        "Products to Exclude *(PARTIAL mode)*\nOne SKU per line",
        value='6316181000\n6316181001\n6316186031\n6306187073\n6316186032\n6306187074\n6306188021\n6306187076\n6306188031\n6306187103\n6306187104\n6306188071\n6306187092\n6306187095\n6306188060\n6306188061\n6306187093\n6306187094\n6316186041\n6316186042\n6306183011\n6306183010\n6306183053\n6306183052\n6306183001\n6306183000\n6306183050\n6306183051\n6306928000\n6306921000\n6316921001\n6316181100\n6316186200\n6316186202\n6316186201\n6316187201\n6316187202\n6316921222\n6316921200\n6316927200\n6301182010\n6306083008\n6306083005\n6306083004\n6306083003\n6306083007\n6316082003\n6316081002\n6316201000\n6316202000\n6306203000\n6316521000\n6316522000\n6316523000\n6306523001\n6316681000\n6316682000\n6316683000\n6306682000\n6306683000\n6316902001\n6316081100\n6319500003\n6319500001\n6319500002\n6319500005\n6319500004\n6306803001\n6316803201\n6316803000\n6316803200\n6316801000\n6316801200\n6306804050\n6306804060\n6306804070\n6316803051\n6316803055\n6316803050\n6316801050\n6306844000\n6306843000\n6316843000\n6316841000\n6319000001\n6319000002\n6319000003\n6319000004\n6319000005\n6319100001\n6319100002\n6319100003\n6319100004\n6319100008\n6319100005\n6319100006\n6319100007\n6319000030\n6319000024\n6319100027\n6319100009\n6319000022\n6319000025\n6319000016\n6319100019\n6319100010\n6319100011\n6319000008\n6319000010\n6319000032\n6319100023',
        height=200,
    )
    products_to_exclude = [p.strip() for p in exclude_text.splitlines() if p.strip()]

    # Auto-remove parent product from exclude list so simulation always runs on it
    if parent_product and parent_product.strip() in products_to_exclude:
        products_to_exclude = [p for p in products_to_exclude if p != parent_product.strip()]
        st.caption(f"ℹ️ Parent product **{parent_product}** was automatically removed from the exclude list.")

    st.divider()
    run_btn = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    st.divider()
    if st.button("🔄 Reload Data", use_container_width=True):
        st.session_state.data_loaded = None
        st.session_state.full_bom = None
        st.session_state.pivot_df = None
        st.session_state.sim_result = None
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════
# RUN SIMULATION
# ══════════════════════════════════════════════════════════════

if run_btn:
    cfg = SimulationConfig(
        parent_product=parent_product,
        month=month,
        production_change=production_change,
        action=action,
        search_mode=search_mode,
        products_to_exclude=products_to_exclude,
    )
    with st.spinner("Running BOM explosion…"):
        try:
            _mm = getattr(st.session_state, "master_map", None) or {}
            result = run_simulation(
                full_bom=st.session_state.full_bom,
                pivot_df=st.session_state.original_pivot.copy(),
                cfg=cfg,
            )
            # Recompute INV/COVER with master data after simulation
            result["pivot"] = _compute_inv_cover(result["pivot"], master_map=_mm)
            # Re-run PO recommendation engine on the updated pivot
            _sim_month_cols = sorted([c for c in result["pivot"].columns if str(c).startswith("202")])
            _mm_sim2 = getattr(st.session_state, "master_map", {}) or {}
            result["pivot"] = compute_po_recommendations(result["pivot"], _sim_month_cols, master_map=_mm_sim2)
            result["month"] = cfg.month
            st.session_state.sim_result = result
            if not result["summary_df"].empty:
                st.session_state.changed_items = set(result["summary_df"]["raw_material"].tolist())
            else:
                st.session_state.changed_items = set()
            st.success("✅ Simulation completed!")
        except Exception as e:
            st.error(f"Simulation error: {e}")

# ══════════════════════════════════════════════════════════════
# HTML PIVOT RENDERER  –  frozen header + frozen Item/Desc/Type
# ══════════════════════════════════════════════════════════════


def inject_master_rows(df: pd.DataFrame, master_map: dict) -> pd.DataFrame:
    """Add a MASTER_DATA row (LT/SS/SL/MAX badges) at the top of each item group."""
    if not master_map:
        return df
    rows = []
    for item in df["item"].dropna().unique():
        mp   = master_map.get(str(item), {})
        desc = df[df["item"]==item]["description"].dropna()
        desc = desc.iloc[0] if not desc.empty else ""
        rows.append({
            "item":             item,
            "description":      desc,
            "ORDER_TYPE_FINAL": "MASTER_DATA",
            "_lt":  mp.get("lead_time"),
            "_ss":  mp.get("safety_stock"),
            "_sl":  mp.get("shelf_life"),
            "_max": mp.get("max_inventory"),
            "_pc":  mp.get("planner_code",""),
        })
    if not rows:
        return df
    master_df = pd.DataFrame(rows)
    combined  = pd.concat([master_df, df], ignore_index=True)
    # Re-sort: MASTER_DATA (0) always first in each item group
    order = {
        "MASTER_DATA":0,"1.Forecast":1,"2.ACTUAL":2,
        "Planned order demand":3,"Work order demand":4,"Planned order":6,
        "Purchase order":9,"Purchase requisition":10,"3.On Hand":11,
        "INV":12,"COVER_MONTHS":13,"PO_RECOMMENDATION":14,
        "COVER_MONTHS_UPDATED":15,"Other":99,
    }
    combined["_s"] = combined["ORDER_TYPE_FINAL"].map(order).fillna(50)
    combined = combined.sort_values(["item","_s"]).drop(columns="_s")
    combined["description"] = combined.groupby("item")["description"].transform("first")
    return combined


@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def render_pivot_html(df, month_cols, changed_items=None, master_map_key=None, lt_markers=None, sub_items=None):
    """
    Lean HTML pivot renderer — uses CSS classes (not inline styles) for sticky
    columns so the browser only computes sticky layout once per class, not
    once per cell. This is the key fix for scroll lag with many rows.
    """
    if isinstance(changed_items, frozenset): changed_items = set(changed_items)
    if isinstance(month_cols, tuple): month_cols = list(month_cols)
    changed_items = changed_items or set()
    # sub_items: dict {sub_sku: main_sku} for highlighting substitute On Hand rows
    if isinstance(sub_items, frozenset): sub_items = dict(sub_items)
    sub_items = sub_items or {}
    # master_map_key is a frozenset of tuples (item, ss) for cache stability
    _ss_map = dict(master_map_key) if master_map_key else {}
    # lt_markers: frozenset of (item, lt_col) — which column is the LT boundary per item
    _lt_map = dict(lt_markers) if lt_markers else {}   # {item: col_name}

    order_labels = {
        "1.Forecast": "Forecast", "2.ACTUAL": "Actual",
        "Planned order demand": "Planned demand", "Work order demand": "WO demand",
        "Purchase order": "Purchase order", "Purchase requisition": "PR",
        "3.On Hand": "On Hand", "INV": "INV",
        "COVER_MONTHS": "Cover (mo)", "PO_RECOMMENDATION": "PO Rec.",
        "COVER_MONTHS_UPDATED": "Cover Updated",
        "MASTER_DATA": "📋 Master Data",
        "SUBSTITUTE_OH": "🔄 Substitute OH",
        "Other": "Other",
    }
    row_class = {
        "3.On Hand": "r-oh", "INV": "r-inv",
        "COVER_MONTHS": "r-cov", "PO_RECOMMENDATION": "r-po",
        "PO_EXCEPTION": "r-exc", "COVER_MONTHS_UPDATED": "r-covu",
        "2.ACTUAL": "r-act", "1.Forecast": "r-fcst", "MASTER_DATA": "r-master",
        "SUBSTITUTE_OH": "r-sub",
    }
    type_color_class = {
        "COVER_MONTHS": "tc-cov", "PO_RECOMMENDATION": "tc-po",
        "INV": "tc-inv", "3.On Hand": "tc-oh",
        "COVER_MONTHS_UPDATED": "tc-covu", "ITEM_PARAMS": "tc-params",
        "PO_EXCEPTION": "tc-exc",
        "SUBSTITUTE_OH": "tc-sub",
    }

    non_data = {"item", "description", "ORDER_TYPE_FINAL", "SKU", "SKU_full", "_sort_key"}
    cols = [c for c in (month_cols or []) if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c not in non_data]

    # ── Header (built once, classes only) ──────────────────────
    header = (
        '<th class="h h-item">Item</th>'
        '<th class="h h-desc">Description</th>'
        '<th class="h h-type">Type</th>'
    )
    for c in cols:
        lt_cls = " lt-head" if any(_lt_map.get(it) == c for it in _lt_map) else ""
        header += f'<th class="h h-month{lt_cls}">{c}</th>'

    # ── Rows — build with list + join (much faster than += in loop) ──
    row_parts = []
    prev_item = None

    for _, row in df.iterrows():
        item  = str(row.get("item", ""))
        desc  = str(row.get("description", ""))
        otype = str(row.get("ORDER_TYPE_FINAL", ""))

        is_new = item != prev_item
        is_chg = item in changed_items
        is_sub = (item in sub_items) and (otype in ("3.On Hand", "On Hand"))

        sep_cls = " sep" if (is_new and prev_item is not None) else ""
        if is_sub:
            rcls = "r-sub"
        else:
            rcls = row_class.get(otype, "r-chg" if is_chg else "")
        row_classes = f"{rcls}{sep_cls}".strip()
        tr_cls  = f' class="{row_classes}"' if row_classes else ""

        item_text = item if is_new else ""
        desc_text = desc if (is_new and desc not in ("nan","None","")) else ""
        # Add substitute badge to item label
        if is_sub and item_text:
            main = sub_items.get(item, "")
            item_text = f'{item_text}<span class="sub-badge">→{main}</span>'

        type_label = order_labels.get(otype, otype)
        tcls = type_color_class.get(otype, "")
        tcls_attr = f' class="tc {tcls}"' if tcls else ' class="tc"'

        cells = [
            f'<td class="c-item">{item_text}</td>',
            f'<td class="c-desc" title="{desc_text}">{desc_text}</td>',
            f'<td{tcls_attr}>{type_label}</td>',
        ]

        # MASTER_DATA row: show LT / SS / SL / MAX badges in first 4 cols, rest empty
        if otype == "MASTER_DATA":
            lt_v  = row.get("_lt");  ss_v = row.get("_ss")
            sl_v  = row.get("_sl");  mx_v = row.get("_max")
            badges = [
                f'<span style="background:#dbeafe;color:#1e40af;padding:1px 5px;border-radius:3px;font-size:10px;margin-right:2px">LT {lt_v:.0f}mo</span>' if lt_v is not None and not (isinstance(lt_v,float) and pd.isna(lt_v)) else "",
                f'<span style="background:#fef9c3;color:#713f12;padding:1px 5px;border-radius:3px;font-size:10px;margin-right:2px">SS {ss_v:.0f}mo</span>' if ss_v is not None and not (isinstance(ss_v,float) and pd.isna(ss_v)) else "",
                f'<span style="background:#dcfce7;color:#14532d;padding:1px 5px;border-radius:3px;font-size:10px;margin-right:2px">SL {sl_v:.0f}mo</span>' if sl_v is not None and not (isinstance(sl_v,float) and pd.isna(sl_v)) else "",
                f'<span style="background:#f3e8ff;color:#6b21a8;padding:1px 5px;border-radius:3px;font-size:10px">MAX {mx_v:.0f}mo</span>' if mx_v is not None and not (isinstance(mx_v,float) and pd.isna(mx_v)) else "",
            ]
            badge_html = "".join(b for b in badges if b)
            for j, c in enumerate(cols):
                if j == 0:
                    cells.append(f'<td class="v" style="text-align:left">{badge_html}</td>')
                else:
                    cells.append('<td class="v"></td>')
        else:
            for c in cols:
                val = row.get(c)
                try:
                    fval = float(val)
                    if pd.isna(fval):
                        cells.append('<td class="v">–</td>')
                    elif otype == "COVER_MONTHS" and fval < float(_ss_map.get(item, 7)):
                        cells.append(f'<td class="v alert-cov">{fval:,.1f}</td>')
                    elif otype == "COVER_MONTHS_UPDATED" and fval < float(_ss_map.get(item, 7)):
                        cells.append(f'<td class="v alert-covu">{fval:,.1f}</td>')
                    elif otype == "PO_RECOMMENDATION" and fval > 0:
                        cells.append(f'<td class="v alert-po">{fval:,.0f}</td>')
                    elif otype == "PO_EXCEPTION" and fval > 0:
                        cells.append(f'<td class="v alert-exc">⚠️ {fval:,.0f}</td>')
                    else:
                        cells.append(f'<td class="v">{fval:,.1f}</td>')
                except (TypeError, ValueError):
                    cells.append('<td class="v">–</td>')

        row_parts.append(f'<tr{tr_cls}>{"".join(cells)}</tr>')
        prev_item = item

    rows_html = "".join(row_parts)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:100%;height:100%;overflow:hidden}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;background:#fff}}
#wrap{{position:absolute;top:0;left:0;right:0;bottom:0;overflow:auto;-webkit-overflow-scrolling:touch;contain:strict}}
#wrap::-webkit-scrollbar{{height:10px;width:10px}}
#wrap::-webkit-scrollbar-track{{background:#f1f1f1;border-radius:4px}}
#wrap::-webkit-scrollbar-thumb{{background:#b0b8c8;border-radius:4px;border:2px solid #f1f1f1}}
#wrap::-webkit-scrollbar-thumb:hover{{background:#7a8499}}

table{{border-collapse:separate;border-spacing:0;white-space:nowrap;min-width:100%;table-layout:fixed}}
th,td{{padding:0}}

/* Header */
.h{{position:sticky;top:0;z-index:20;background:#e8eaf0;padding:7px 8px;font-size:11px;font-weight:700;color:#444;border-bottom:2px solid #b0b8c8;white-space:nowrap;user-select:none}}
.h-item{{left:0;z-index:30;width:110px;min-width:110px;max-width:110px;border-right:1px solid #ccd}}
.h-desc{{left:110px;z-index:30;width:200px;min-width:200px;max-width:200px;border-right:1px solid #ccd}}
.h-type{{left:310px;z-index:30;width:110px;min-width:110px;max-width:110px;border-right:2px solid #99a}}
.h-month{{width:75px;min-width:75px;text-align:right}}

/* Frozen body columns — single class, computed once by browser */
.c-item{{position:sticky;left:0;z-index:5;background:inherit;width:110px;min-width:110px;max-width:110px;
  font-weight:600;font-size:12px;padding:5px 8px;border-right:1px solid #e0e4ee;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.c-desc{{position:sticky;left:110px;z-index:5;background:inherit;width:200px;min-width:200px;max-width:200px;
  font-size:11px;color:#555;padding:5px 8px;border-right:1px solid #e0e4ee;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.tc{{position:sticky;left:310px;z-index:5;background:inherit;width:110px;min-width:110px;max-width:110px;
  font-size:11px;font-weight:600;color:#555;padding:5px 8px;border-right:2px solid #99a;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}

/* Value cells */
.v{{padding:4px 8px;text-align:right;font-size:12px;font-variant-numeric:tabular-nums;width:75px;min-width:75px;
   background:inherit}}

/* Row background colors — applied once per <tr>, inherited by sticky cells via background:inherit */
tr{{background:#fff}}
tr.r-oh{{background:#e8f4e8}}
tr.r-inv{{background:#eef2ff}}
tr.r-cov{{background:#fff8e1}}
tr.r-po{{background:#fff3e0}}
tr.r-covu{{background:#e8f4fd}}
tr.r-act{{background:#f7f7f7}}
tr.r-chg{{background:#fffde7}}
tr.r-master{{background:#f0f9ff;font-size:11px}}
tr.sep td{{border-top:2.5px solid #99a}}

/* Type label colors */
.tc-cov{{color:#b45309}}
.tc-po{{color:#c2410c}}
.tc-inv{{color:#4338ca}}
.tc-oh{{color:#166534}}
.tc-covu{{color:#0369a1}}
.tc-params{{color:#6d28d9;font-size:10px}}
.tc-exc{{color:#991b1b;font-weight:700}}
tr.r-exc{{background:#fee2e2}}
tr.r-sub{{background:#fff7ed;border-left:3px solid #f97316}}
tr.r-sub .c-item{{color:#c2410c;font-weight:700}}
.tc-sub{{color:#c2410c;font-weight:700}}
.sub-badge{{background:#fed7aa;color:#9a3412;padding:1px 5px;border-radius:3px;font-size:10px;margin-left:4px}}
.alert-exc{{background:#b91c1c !important;color:#fff;font-weight:700}}
td.lt-mark{{border-left:3px solid #1d4ed8 !important;position:relative}}
th.lt-head{{border-left:3px solid #1d4ed8 !important;color:#1d4ed8}}
tr.r-master{{background:#f5f3ff}}
tr.r-master .c-item,.r-master .c-desc,.r-master .tc{{background:#f5f3ff}}

/* Alert cells — solid color, override inherited bg */
.alert-cov{{background:#dc2626 !important;color:#fff;font-weight:700}}
.alert-covu{{background:#ef4444 !important;color:#fff;font-weight:700}}
.alert-po{{background:#ea580c !important;color:#fff;font-weight:700}}
</style></head><body>
<div id="wrap">
<table>
<thead><tr>{header}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
</body></html>"""


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_sim, tab_changes, tab_analysis, tab_data, tab_help = st.tabs(["🔬 Simulation Results", "📋 Forecast Changes", "🧠 Analysis & Insights", "📊 Data Preview", "❓ Help"])

# ──────────────────────────────────────────────────────────────
# TAB 1 – SIMULATION RESULTS
# ──────────────────────────────────────────────────────────────

with tab_sim:
    if not st.session_state.sim_result:
        st.info("👈 Set parameters in the sidebar and click **Run Simulation**.")
    else:
        res = st.session_state.sim_result
        results_df   = res["results_df"]
        summary_df   = res["summary_df"]
        updated_pivot = res["pivot"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw Materials Found", len(summary_df))
        m2.metric("BOM Paths Explored",  len(results_df))
        total_qty = summary_df["qty_change"].sum() if not summary_df.empty else 0
        m3.metric("Total Qty Change", f"{total_qty:,.2f}")

        st.divider()

        with st.expander("📋 BOM Explosion Detail", expanded=False):
            if results_df.empty:
                st.warning("No raw materials found for the given parameters.")
            else:
                st.dataframe(results_df, use_container_width=True)

        st.subheader("📊 Raw Material Summary")
        # Debug: show if CPC is in results
        with st.expander("🔍 Debug: BOM explosion detail for CPC / missing items", expanded=False):
            if not res["results_df"].empty:
                cpc_rows = res["results_df"][res["results_df"]["raw_material"].astype(str) == "6305050310"]
                st.write(f"CPC (6305050310) rows in explosion: {len(cpc_rows)}")
                if not cpc_rows.empty:
                    st.dataframe(cpc_rows)
                else:
                    st.warning("CPC NOT found in explosion results")
                st.write(f"component_types in results: {res['results_df']['component_type'].value_counts().to_dict()}")
                st.write(f"Total unique raw_materials in summary: {len(summary_df)}")
        if not summary_df.empty:
            # Add description from pivot
            desc_map = (
                updated_pivot[["item","description"]]
                .drop_duplicates("item")
                .dropna(subset=["description"])
                .set_index("item")["description"]
                .to_dict()
            )
            summary_display = summary_df.copy()
            summary_display.insert(
                1, "description",
                summary_display["raw_material"].map(desc_map).fillna("")
            )
            st.dataframe(
                summary_display.style.format({"qty_change": "{:,.3f}"}),
                use_container_width=True,
            )

        st.subheader("📈 Updated MRP Pivot")
        st.caption(
            "ℹ️ The **Forecast row** shows the **total forecast after simulation** "
            "(existing forecast + simulation change). "
            "To see only the change per material, check the **📊 Raw Material Summary** above "
            "or the **📋 Forecast Changes** tab."
        )

        all_month_cols = sorted([c for c in updated_pivot.columns if str(c).startswith("202")])

        onhand_rows = updated_pivot[
            updated_pivot["ORDER_TYPE_FINAL"].str.contains("on hand", case=False, na=False)
        ]
        first_onhand_month = None
        for m in all_month_cols:
            if m in onhand_rows.columns and pd.to_numeric(onhand_rows[m], errors="coerce").sum() > 0:
                first_onhand_month = m
                break

        # ── Pre-compute filter options (fast, no HTML render) ──────
        changed_items   = st.session_state.changed_items or set()
        all_items_list  = updated_pivot["item"].dropna().unique().tolist()

        # Pre-compute low-cover items once (avoids iterrows on every filter change)
        @st.cache_data(ttl=120, show_spinner=False)
        def _low_cover_items(_plen, _sim_len):
            pv = st.session_state.sim_result["pivot"] if st.session_state.sim_result else None
            if pv is None: return set()
            mc = sorted([c for c in pv.columns if str(c).startswith("202")])
            cr = pv[pv["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
            if cr.empty or not mc: return set()
            vals = cr[mc].apply(pd.to_numeric, errors="coerce")
            return set(cr.loc[(vals < 7).any(axis=1), "item"].tolist())

        low_cover_set = _low_cover_items(len(updated_pivot), len(changed_items))

        # ── Filters row ──────────────────────────────────────────────
        _mm_pivot_filter = getattr(st.session_state, "master_map", {}) or {}
        pc_options = sorted({
            str(_mm_pivot_filter.get(str(it), {}).get("planner_code", "Unknown")) or "Unknown"
            for it in all_items_list
        })

        fc0, fc1, fc2, fc3, fc4 = st.columns([1.4, 2.6, 1, 1, 1])
        with fc0:
            filter_planner = st.multiselect(
                "🧭 Filter by Planner Code", options=pc_options,
                default=[], placeholder="All planner codes…",
                help="e.g. RM1 / RM2 — narrows the SKUs available in the item filter below",
                key="pivot_planner_filter",
            )
        planner_items = (
            {it for it in all_items_list
             if str(_mm_pivot_filter.get(str(it), {}).get("planner_code", "Unknown")) in filter_planner}
            if filter_planner else set(all_items_list)
        )
        with fc1:
            filter_items = st.multiselect(
                "Filter by item", options=sorted(planner_items),
                default=[], placeholder="Show all items…",
            )
        with fc2:
            filter_from_onhand = st.checkbox(
                "📅 From On Hand month", value=True,
                help=f"First On Hand month: {first_onhand_month}" if first_onhand_month else "Not detected",
            )
        with fc3:
            filter_changed_only = st.checkbox(
                "🟡 Changed items only", value=False,
                help="Show only SKUs updated by the simulation",
            )
        with fc4:
            filter_low_cover = st.checkbox(
                "🔴 COVER < 7 only", value=False,
                help="Show only SKUs with at least one COVER month below 7",
            )

        # ── Apply filters (pure pandas, no render yet) ───────────────
        # Determine which items to show
        show_items = set(all_items_list)
        if filter_planner:       show_items &= planner_items
        if filter_items:         show_items &= set(filter_items)
        if filter_changed_only:  show_items &= changed_items
        if filter_low_cover:     show_items &= low_cover_set

        if filter_from_onhand and first_onhand_month and first_onhand_month in all_month_cols:
            month_display = all_month_cols[all_month_cols.index(first_onhand_month):]
        else:
            month_display = all_month_cols

        # ── Build pivot_view only with needed rows/cols ──────────────
        @st.cache_data(ttl=60, show_spinner=False, max_entries=20)
        def _build_pivot_view(_plen, items_key, months_key):
            """Cache the sorted/filtered view so repeated filter tweaks are instant."""
            pv = st.session_state.sim_result["pivot"]
            # Exclude display-only hidden rows
            pv = pv[pv["ORDER_TYPE_FINAL"] != "PO_EXCEPTION"]
            items = list(items_key)
            mths  = list(months_key)
            view  = pv[pv["item"].isin(items)].copy() if items else pv.copy()
            order_sort = {
                "1.Forecast":1,"2.ACTUAL":2,"Planned order demand":3,"Work order demand":4,
                "Planned order":6,"Purchase order":9,"Purchase requisition":10,
                "3.On Hand":11,"INV":12,"COVER_MONTHS":13,"PO_RECOMMENDATION":14,
                "PO_EXCEPTION":14.5,"COVER_MONTHS_UPDATED":15,"Other":99,
            }
            view["_s"] = view["ORDER_TYPE_FINAL"].map(order_sort).fillna(50)
            view = view.sort_values(["item","_s"]).drop(columns="_s")
            view["description"] = view.groupby("item")["description"].transform("first")
            # Keep all columns needed for display + meta cols for master row
            extra_meta = [c for c in ["_lt","_ss","_sl","_max","_pc"] if c in view.columns]
            dcols = ["item","description","ORDER_TYPE_FINAL"] + extra_meta + [c for c in mths if c in view.columns]
            return view[dcols]

        # Hide PO_EXCEPTION from display (kept in data for logic, not shown)
        updated_pivot_display = updated_pivot[
            updated_pivot["ORDER_TYPE_FINAL"] != "PO_EXCEPTION"
        ]
        pivot_view = _build_pivot_view(
            len(updated_pivot_display),
            frozenset(show_items),
            tuple(month_display),
        )

        # ── Inject master data badge row (LT/SS/SL/MAX) ─────────────
        _mm_rend = getattr(st.session_state, "master_map", {}) or {}
        pivot_view = inject_master_rows(pivot_view, _mm_rend)

        # ── Render ───────────────────────────────────────────────────
        import streamlit.components.v1 as _comp
        MAX_RENDER_ROWS = 400
        if len(pivot_view) > MAX_RENDER_ROWS:
            n_pages = (len(pivot_view) - 1) // MAX_RENDER_ROWS + 1
            pg = st.number_input(f"Page (showing {MAX_RENDER_ROWS} rows/page, {len(pivot_view)} total)", min_value=1, max_value=n_pages, value=1, key="sim_pivot_page")
            page_view = pivot_view.iloc[(pg-1)*MAX_RENDER_ROWS : pg*MAX_RENDER_ROWS]
        else:
            page_view = pivot_view

        n_rows = len(page_view)
        h = min(680, max(280, n_rows * 29 + 40))
        _ss_key = frozenset((str(k), float(v.get("safety_stock", 7))) for k, v in _mm_rend.items())
        _comp.html(
            render_pivot_html(page_view, tuple(sorted(month_display)), frozenset(changed_items or set()), master_map_key=_ss_key,
                              sub_items=frozenset((k,v) for k,v in (getattr(st.session_state,"sub_items_map",{}) or {}).items())),
            height=h, scrolling=True,
        )

        legend_parts = ["🔴 COVER < SS months", "🟠 PO Recommendation", "🔴⚠️ PO Exception (cover < 80% SS)", "🔵 Coverage Updated (post-PO)"]
        if changed_items: legend_parts.append("🟡 Updated by simulation")
        if first_onhand_month:
            st.caption(f"📅 First On Hand: **{first_onhand_month}** | {len(show_items)} items | {'  |  '.join(legend_parts)}")

        # ── Forecast vs Actual chart — per SKU ──────────────────────
        st.divider()
        st.subheader("📊 Forecast vs Actual Consumption — by SKU")
        chart_candidates = sorted(show_items) if show_items else all_items_list
        if not chart_candidates:
            st.info("No SKUs available for the current filters.")
        else:
            sel_chart_item = st.selectbox(
                "Select SKU to chart", options=chart_candidates, key="pivot_fcst_act_item",
            )
            if sel_chart_item:
                try:
                    import plotly.graph_objects as _go
                    fcst_row = updated_pivot[
                        (updated_pivot["item"] == sel_chart_item) & (updated_pivot["ORDER_TYPE_FINAL"] == "1.Forecast")
                    ]
                    act_row = updated_pivot[
                        (updated_pivot["item"] == sel_chart_item) & (updated_pivot["ORDER_TYPE_FINAL"] == "2.ACTUAL")
                    ]
                    chart_months = month_display if month_display else all_month_cols
                    desc_s = updated_pivot[updated_pivot["item"] == sel_chart_item]["description"].dropna()
                    desc_s = desc_s.iloc[0] if not desc_s.empty else ""

                    if fcst_row.empty and act_row.empty:
                        st.info("No Forecast/Actual rows found for this SKU.")
                    else:
                        f_vals = (pd.to_numeric(fcst_row[chart_months].iloc[0], errors="coerce")
                                  if not fcst_row.empty else pd.Series([np.nan]*len(chart_months), index=chart_months))
                        a_vals = (pd.to_numeric(act_row[chart_months].iloc[0], errors="coerce")
                                  if not act_row.empty else pd.Series([np.nan]*len(chart_months), index=chart_months))

                        fig_fa = _go.Figure()
                        fig_fa.add_trace(_go.Bar(x=chart_months, y=a_vals.values, name="Actual", marker_color="#3b82f6"))
                        fig_fa.add_trace(_go.Scatter(x=chart_months, y=f_vals.values, name="Forecast",
                                                      mode="lines+markers", line=dict(color="#f97316", width=2)))
                        fig_fa.update_layout(
                            title=f"{sel_chart_item} — {desc_s}",
                            height=380, xaxis_title="Month", yaxis_title="Qty",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(t=60),
                        )
                        st.plotly_chart(fig_fa, use_container_width=True)
                except ImportError:
                    st.warning("Plotly not available — install `plotly` to see this chart.")

        # Downloads
        st.subheader("💾 Download Results")

        def to_excel(df):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, index=False)
            return buf.getvalue()

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("📥 Explosion Detail",  data=to_excel(results_df),   file_name="mrp_simulation_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d2:
            st.download_button("📥 Material Summary",  data=to_excel(summary_df),   file_name="mrp_material_summary.xlsx",   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d3:
            st.download_button("📥 Full Updated Pivot", data=to_excel(updated_pivot), file_name="mrp_final_simulation.xlsx",   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ──────────────────────────────────────────────────────────────
# TAB 2 – DATA PREVIEW
# ──────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# LOCAL FALLBACK  (used when Databricks API is unavailable)
# ══════════════════════════════════════════════════════════════

def _local_fallback(question, insights):
    """Answer from pre-computed ML insights without calling any API."""
    q  = question.lower()
    NL = "\n"
    r_df = insights.get("risk", pd.DataFrame())

    if any(w in q for w in ["risk","critical","urgent","danger","attention"]):
        if r_df.empty: return "Risk data not available — ensure COVER_MONTHS rows exist."
        crit = r_df[r_df["risk_level"]=="🔴 Critical"]
        high = r_df[r_df["risk_level"]=="🟠 High"]
        lines = [f"**Risk Summary — {len(r_df)} items analyzed:**",
                 f"🔴 Critical ({len(crit)}): {', '.join(crit['item'].tolist()) or 'None'}",
                 f"🟠 High ({len(high)}): {', '.join(high['item'].tolist()[:8]) or 'None'}"]
        if not crit.empty:
            lines.append(f"{NL}**Critical item details:**")
            for _, r in crit.iterrows():
                lines.append(f"- **{r['item']}** ({r['description']}) | Min cover: {r['min_cover']}mo | Score: {r['risk_score']:.0f} | {r['flags']}")
        lines.append(f"{NL}💡 Issue POs for critical items immediately.")
        return NL.join(lines)

    if any(w in q for w in ["po","purchase","order","reorder","buy","procure"]):
        po = insights.get("po_items", [])
        if not po: return "No PO recommendations — all items have sufficient coverage (>7 months)."
        lines = [f"**PO Recommendations — {len(po)} items require ordering:**"]
        for p in po[:15]:
            lines.append(f"- **{p['item']}** ({p['description']}): {p['total_po_recommended']:,.0f} units")
        lines.append(f"{NL}💡 These quantities will bring coverage up to 15 months.")
        return NL.join(lines)

    if any(w in q for w in ["cover","coverage","low","below","month"]):
        lines = [f"**Coverage Summary:**",
                 f"- Avg coverage: {insights.get('cv_mean','N/A')} months",
                 f"- Min coverage: {insights.get('cv_min','N/A')} months",
                 f"- Items below 7 months: **{insights.get('items_below_7',0)}**",
                 f"- Items with negative INV: **{insights.get('inv_negative_items',0)}**"]
        return NL.join(lines)

    if any(w in q for w in ["trend","grow","declin","increas","decreas","demand"]):
        t = insights.get("trends", pd.DataFrame())
        if t.empty: return "No trend data available (need actual consumption ≥3 months)."
        g = t[t["trend_label"]=="📈 Growing"]
        d = t[t["trend_label"]=="📉 Declining"]
        s = t[t["trend_label"]=="➡️ Stable"]
        lines = [f"**Demand Trends — {len(t)} items:**",
                 f"📈 Growing: {len(g)} | ➡️ Stable: {len(s)} | 📉 Declining: {len(d)}"]
        if not d.empty:
            lines.append(f"{NL}**📉 Declining items:**")
            for _, r in d.head(8).iterrows():
                lines.append(f"- **{r['item']}** ({r['description']}) | {r['norm_slope']:.1%}/month | Avg: {r['avg_value']:,.1f}")
        if not g.empty:
            lines.append(f"{NL}**📈 Growing items:**")
            for _, r in g.head(8).iterrows():
                lines.append(f"- **{r['item']}** ({r['description']}) | {r['norm_slope']:.1%}/month | Avg: {r['avg_value']:,.1f}")
        return NL.join(lines)

    if any(w in q for w in ["anomal","spike","unusual","outlier","strange"]):
        an = insights.get("anomalies", pd.DataFrame())
        if an.empty or "anomaly" not in an.columns: return "No anomaly data available."
        anom = an[an["anomaly"]].sort_values("anomaly_score", ascending=False)
        lines = [f"**{len(anom)} anomalies detected across {an['item'].nunique()} items:**"]
        for _, r in anom.head(10).iterrows():
            lines.append(f"- **{r['item']}** | {str(r['month'])[:10]} | Value: {r['value']:,.1f} | Score: {r['anomaly_score']:.3f}")
        return NL.join(lines)

    if any(w in q for w in ["forecast","accuracy","mape","error","bias"]):
        fa = insights.get("accuracy", pd.DataFrame())
        if fa.empty: return "No forecast accuracy data available."
        om = insights.get("overall_mape", "N/A")
        im = fa.groupby(["item","description"])["mape"].mean().reset_index().sort_values("mape", ascending=False)
        lines = [f"**Forecast Accuracy — Overall MAPE: {om}%**"]
        for _, r in im.head(8).iterrows():
            lines.append(f"- **{r['item']}** ({r['description']}): MAPE {r['mape']:.1f}%")
        return NL.join(lines)

    # General summary
    n = insights.get("n_items", 0); mc = insights.get("months", [])
    return (f"**MRP Data Summary:**{NL}"
            f"- {n} items | {len(mc)} months ({mc[0] if mc else '?'} – {mc[-1] if mc else '?'}){NL}"
            f"- Coverage < 7mo: {insights.get('items_below_7',0)} | Negative INV: {insights.get('inv_negative_items',0)}{NL}"
            f"- Anomalies: {insights.get('n_anomalies',0)} | Forecast MAPE: {insights.get('overall_mape','N/A')}%{NL}{NL}"
            f"Try: *'Which items are critical?'*, *'What POs do I need?'*, *'Show anomalies'*, *'Declining trends'*")


# ══════════════════════════════════════════════════════════════
# TAB 2 – ANALYSIS & INSIGHTS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# TAB: FORECAST CHANGES
# ══════════════════════════════════════════════════════════════

with tab_changes:
    if not st.session_state.sim_result:
        st.info("👈 Run a simulation first to see forecast changes.")
    else:
        res          = st.session_state.sim_result
        upd_pivot    = res["pivot"]
        orig_pivot   = st.session_state.original_pivot
        summary_df   = res["summary_df"]
        month_cols_c = sorted([c for c in upd_pivot.columns if str(c).startswith("202")])
        sim_month    = res.get("month", "")

        st.header("📋 Forecast Changes from Simulation")
        st.caption(f"Showing items whose **Forecast changed** in the simulation — month: **{sim_month}**")

        # ── Build change table ─────────────────────────────────
        def _build_changes():
            """Compare updated vs original pivot Forecast rows — no cache to avoid stale results."""
            upd  = st.session_state.sim_result["pivot"]
            orig = st.session_state.original_pivot
            mc   = sorted([c for c in upd.columns if str(c).startswith("202")])

            fcst_upd  = upd[upd["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)].copy()
            fcst_orig = orig[orig["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)].copy()

            # Check ALL items from BOTH pivots (catches new items added by simulation)
            all_items = set(fcst_upd["item"].dropna().unique()) | set(fcst_orig["item"].dropna().unique())

            rows = []
            for item in all_items:
                fu   = fcst_upd[fcst_upd["item"] == item]
                fo   = fcst_orig[fcst_orig["item"] == item]
                desc = ""
                if not fu.empty and not fu["description"].dropna().empty:
                    desc = fu["description"].dropna().iloc[0]
                elif not fo.empty and not fo["description"].dropna().empty:
                    desc = fo["description"].dropna().iloc[0]

                for m in mc:
                    val_new = float(pd.to_numeric(fu[m].values[0], errors="coerce") or 0)                               if not fu.empty and m in fu.columns and len(fu[m].values) > 0 else 0.0
                    val_old = float(pd.to_numeric(fo[m].values[0], errors="coerce") or 0)                               if not fo.empty and m in fo.columns and len(fo[m].values) > 0 else 0.0
                    delta = round(val_new - val_old, 6)
                    if abs(delta) > 0.00001:
                        rows.append({
                            "item":            item,
                            "description":     desc,
                            "month":           m,
                            "forecast_before": round(val_old, 6),
                            "forecast_after":  round(val_new, 6),
                            "change":          round(delta, 6),
                        })
            if not rows:
                return pd.DataFrame(columns=["item","description","month",
                                             "forecast_before","forecast_after","change"])
            return pd.DataFrame(rows).sort_values(["item","month"])

        changes_df = _build_changes()

        if changes_df.empty:
            st.info("No forecast changes detected.")
        else:
            # ── KPI row ────────────────────────────────────────
            k1, k2, k3 = st.columns(3)
            k1.metric("Items with changes",   changes_df["item"].nunique())
            k2.metric("Total months changed", len(changes_df))
            k3.metric("Net qty change",       f"{changes_df['change'].sum():,.3f}")

            st.divider()

            # ── Filter controls ────────────────────────────────
            f1, f2 = st.columns([2, 1])
            sel_items = f1.multiselect(
                "Filter by item", options=sorted(changes_df["item"].unique()),
                default=[], placeholder="Show all changed items…",
                key="changes_item_filter"
            )
            show_month = f2.selectbox(
                "Filter by month", options=["All"] + sorted(changes_df["month"].unique()),
                key="changes_month_filter"
            )

            view_ch = changes_df.copy()
            if sel_items:
                view_ch = view_ch[view_ch["item"].isin(sel_items)]
            if show_month != "All":
                view_ch = view_ch[view_ch["month"] == show_month]

            # ── Styled table ───────────────────────────────────
            def _color_delta(val):
                if isinstance(val, (int, float)):
                    if val > 0: return "background-color:#dcfce7;color:#15803d;font-weight:600"
                    if val < 0: return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                return ""

            st.dataframe(
                view_ch.style
                    .map(_color_delta, subset=["change"])
                    .format({
                        "forecast_before": "{:,.4f}",
                        "forecast_after":  "{:,.4f}",
                        "change":          "{:+,.4f}",
                    }),
                use_container_width=True,
                height=min(600, max(300, len(view_ch) * 36 + 40)),
            )

            # ── Pivot view: before vs after side by side ───────
            st.divider()
            st.subheader("📊 Before / After Pivot")
            pivot_items = list(changes_df["item"].unique())
            sel_pivot_item = st.selectbox(
                "Select item to compare", options=pivot_items,
                key="changes_pivot_sel"
            )

            if sel_pivot_item:
                mc_focus = sorted(changes_df[changes_df["item"]==sel_pivot_item]["month"].unique())
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown("**Before simulation:**")
                    orig_item = orig_pivot[orig_pivot["item"] == sel_pivot_item]
                    if not orig_item.empty:
                        display_mc = [m for m in mc_focus if m in orig_item.columns]
                        st.dataframe(
                            orig_item[["ORDER_TYPE_FINAL"] + display_mc]
                            .style.format({m: "{:,.3f}" for m in display_mc}, na_rep="–"),
                            use_container_width=True,
                        )

                with col_r:
                    st.markdown("**After simulation:**")
                    upd_item = upd_pivot[upd_pivot["item"] == sel_pivot_item]
                    if not upd_item.empty:
                        # exclude computed rows from comparison
                        upd_show = upd_item[~upd_item["ORDER_TYPE_FINAL"].isin(
                            ["PO_RECOMMENDATION","COVER_MONTHS_UPDATED","MASTER_DATA","PO_EXCEPTION"]
                        )]
                        display_mc2 = [m for m in mc_focus if m in upd_show.columns]
                        def _hl_fcst(row):
                            if "forecast" in str(row.get("ORDER_TYPE_FINAL","")).lower():
                                return ["background-color:#dbeafe"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            upd_show[["ORDER_TYPE_FINAL"] + display_mc2]
                            .style.apply(_hl_fcst, axis=1)
                            .format({m: "{:,.3f}" for m in display_mc2}, na_rep="–"),
                            use_container_width=True,
                        )

            # ── Download ───────────────────────────────────────
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                changes_df.to_excel(w, index=False, sheet_name="Forecast Changes")
                if not summary_df.empty:
                    summary_df.to_excel(w, index=False, sheet_name="BOM Summary")
            st.download_button(
                "📥 Download Changes Report",
                data=buf.getvalue(),
                file_name="simulation_changes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab_analysis:
    pivot = st.session_state.pivot_df
    if pivot is None:
        st.info("Data not loaded yet.")
    else:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.ensemble import IsolationForest
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from src.analysis import detect_anomalies, compute_trends, compute_risk, compute_forecast_accuracy, extract_item_series, compute_demand_spikes
            HAS_DEPS = True
        except ImportError as e:
            HAS_DEPS = False
            st.error(f"Missing dependency: {e}. Run: `pip install plotly scikit-learn`")

        # ── Filter pivot to RM1/RM2 items only for all analysis ──────────
        _mm_analysis = getattr(st.session_state, "master_map", {}) or {}
        _rm_items = {
            it for it, mp in _mm_analysis.items()
            if str(mp.get("planner_code","")).strip() in ("B_RM1","B_RM2","RM1","RM2")
        }
        def get_rm_pivot(pv):
            if not _rm_items:
                return pv
            return pv[pv["item"].isin(_rm_items)].copy()
        pivot_rm = get_rm_pivot(pivot)
        n_rm     = int(pivot_rm["item"].nunique())

        st.header("🧠 ML Analysis & Insights")
        a_tab1, a_tab2, a_tab6, a_tab3, a_tab4, a_tab5 = st.tabs([
            "⚠️ Risk Dashboard", "🔍 Anomaly Detection", "🚀 Spike Alerts",
            "📈 Trend Analysis", "🎯 Forecast Accuracy", "💬 AI Data Chat"
        ])

        # ── Risk Dashboard ──────────────────────────────────────────────
        with a_tab1:
            # ── Cache all computations ─────────────────────────────────
            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_risk(_plen, _rm_len):
                if not HAS_DEPS: return pd.DataFrame(), pd.DataFrame()
                pv = get_rm_pivot(st.session_state.pivot_df)
                t = compute_trends(pv)
                r = compute_risk(pv, t)
                return t, r

            @st.cache_data(ttl=300, show_spinner=False)
            def _compute_inventory_kpis(_plen, _rm_len):
                """Compute Out-of-Stock, Below Safety Stock, Coverage Risk KPIs — RM1/RM2 only."""
                pv = get_rm_pivot(st.session_state.pivot_df)
                if pv is None: return {}
                mc = sorted([c for c in pv.columns if str(c).startswith("202")])
                if not mc: return {}

                # First active month (has On Hand)
                oh_rows = pv[pv["ORDER_TYPE_FINAL"].str.contains("on hand|3.on hand", case=False, na=False)]
                first_oh = None
                for m in mc:
                    if m in oh_rows.columns and pd.to_numeric(oh_rows[m], errors="coerce").sum() > 0:
                        first_oh = m
                        break
                active_mc = mc[mc.index(first_oh):] if first_oh and first_oh in mc else mc
                future_7  = active_mc[:7]

                all_items = pv["item"].dropna().unique().tolist()
                n_total   = len(all_items)

                # ── 1. Out of Stock ─────────────────────────────────────
                # On Hand = 0 in first active month AND no similar item exists
                oos_items = []
                item_names_norm = {}
                for it in all_items:
                    # Normalize: lowercase, remove digits+pack sizes, strip
                    import re as _re
                    norm = _re.sub(r'\b(\d+\s*(kg|g|ml|l|mg|pack|pk|pcs?|units?|x\d+))\b', '', str(it).lower())
                    norm = _re.sub(r'\s+', ' ', norm).strip()
                    item_names_norm[it] = norm

                oh_first = {}
                for it in all_items:
                    r = oh_rows[oh_rows["item"] == it]
                    if r.empty or first_oh not in r.columns:
                        oh_first[it] = 0.0
                        continue
                    oh_first[it] = float(pd.to_numeric(r[first_oh].values[0], errors="coerce") or 0)

                # Group items by normalized name to find alternatives
                from collections import defaultdict as _dd
                norm_groups = _dd(list)
                for it, nm in item_names_norm.items():
                    norm_groups[nm].append(it)

                for it in all_items:
                    if oh_first.get(it, 0) <= 0:
                        # Check if alternative exists with stock
                        norm = item_names_norm[it]
                        alternatives = [a for a in norm_groups.get(norm, []) if a != it and oh_first.get(a, 0) > 0]
                        if not alternatives:
                            oos_items.append(it)

                # ── 2. Below Safety Stock ───────────────────────────────
                # Cover < per-item Safety Stock AND no incoming supply in next 7 months
                bss_items = []
                cover_rows = pv[pv["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
                po_rows    = pv[pv["ORDER_TYPE_FINAL"].str.contains("purchase|po", case=False, na=False)]
                mm_local   = getattr(st.session_state, "master_map", {}) or {}

                for it in all_items:
                    cr = cover_rows[cover_rows["item"] == it]
                    if cr.empty: continue
                    # Get per-item safety stock threshold
                    item_ss = float(mm_local.get(str(it), {}).get("safety_stock", 7.0) or 7.0)
                    # Check if cover < safety_stock in first active month
                    if first_oh and first_oh in cr.columns:
                        cov = float(pd.to_numeric(cr[first_oh].values[0], errors="coerce") or 99)
                    else:
                        vals = pd.to_numeric(cr[active_mc].values.flatten(), errors="coerce")
                        vals = vals[~np.isnan(vals) & (vals > 0)]
                        cov = float(vals[0]) if len(vals) > 0 else 99

                    if cov >= item_ss:
                        continue

                    # No incoming PO in next 7 months?
                    pr = po_rows[po_rows["item"] == it]
                    has_incoming = False
                    if not pr.empty:
                        for m in future_7:
                            if m in pr.columns:
                                v = pd.to_numeric(pr[m].values[0], errors="coerce") if len(pr) > 0 else 0
                                if pd.notna(v) and float(v or 0) > 0:
                                    has_incoming = True
                                    break
                    if not has_incoming:
                        bss_items.append(it)

                # ── 3. Coverage Risk (Potential Risk Items) ─────────────
                # Cover < 7 months in any of next 7 months
                risk_items = []
                for it in all_items:
                    cr = cover_rows[cover_rows["item"] == it]
                    if cr.empty: continue
                    item_ss_r = float(mm_local.get(str(it), {}).get("safety_stock", 7.0) or 7.0)
                    vals = pd.to_numeric(cr[future_7].values.flatten() if future_7 else [], errors="coerce")
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0 and (vals < item_ss_r).any():
                        risk_items.append(it)

                return {
                    "n_total":      n_total,
                    "oos_items":    oos_items,
                    "bss_items":    bss_items,
                    "risk_items":   risk_items,
                    "first_oh":     first_oh,
                    "future_7":     future_7,
                }

            with st.spinner("Loading dashboard…"):
                trends_df, risk_df = _cached_risk(len(pivot), n_rm)
                kpis = _compute_inventory_kpis(len(pivot), n_rm)

            n_total = kpis.get("n_total", 1) or 1
            st.caption(f"📊 Showing **{n_rm} RM1/RM2 items** out of {pivot['item'].nunique()} total items in pivot")

            # ══════════════════════════════════════════════════════════
            # KPI CARDS
            # ══════════════════════════════════════════════════════════
            st.markdown("## 📦 Inventory Risk Dashboard")
            st.markdown(f"*Monitoring **{n_total} active items** | First On-Hand month: **{kpis.get('first_oh','N/A')}** | Next 7 months: **{', '.join(kpis.get('future_7',[])[:3])}…***")
            st.divider()

            n_oos  = len(kpis.get("oos_items", []))
            n_bss  = len(kpis.get("bss_items", []))
            n_risk = len(kpis.get("risk_items", []))
            n_crit = int((risk_df["risk_level"]=="🔴 Critical").sum()) if not risk_df.empty else 0
            n_high = int((risk_df["risk_level"]=="🟠 High").sum())     if not risk_df.empty else 0

            def pct(n): return f"{n/n_total*100:.1f}%"

            # CSS for KPI cards
            st.markdown("""
<style>
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.kpi-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 18px 16px;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;border-radius:12px 12px 0 0}
.kpi-red::before{background:#dc2626}
.kpi-orange::before{background:#f97316}
.kpi-yellow::before{background:#eab308}
.kpi-blue::before{background:#3b82f6}
.kpi-green::before{background:#22c55e}
.kpi-label{font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.kpi-value{font-size:34px;font-weight:700;line-height:1;margin-bottom:4px}
.kpi-pct{font-size:13px;color:#94a3b8;margin-bottom:6px}
.kpi-sub{font-size:11px;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:8px;margin-top:8px}
.kpi-red .kpi-value{color:#dc2626}
.kpi-orange .kpi-value{color:#f97316}
.kpi-yellow .kpi-value{color:#d97706}
.kpi-blue .kpi-value{color:#3b82f6}
.kpi-green .kpi-value{color:#16a34a}
</style>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-red">
    <div class="kpi-label">🚫 Out of Stock</div>
    <div class="kpi-value">{n_oos}</div>
    <div class="kpi-pct">{pct(n_oos)} of all items</div>
    <div class="kpi-sub">On Hand = 0 · no alternative item in stock</div>
  </div>
  <div class="kpi-card kpi-orange">
    <div class="kpi-label">⚠️ Below Safety Stock</div>
    <div class="kpi-value">{n_bss}</div>
    <div class="kpi-pct">{pct(n_bss)} of all items</div>
    <div class="kpi-sub">Cover &lt; 7 months · no incoming PO in next 7 months</div>
  </div>
  <div class="kpi-card kpi-yellow">
    <div class="kpi-label">📉 Potential Risk Items</div>
    <div class="kpi-value">{n_risk}</div>
    <div class="kpi-pct">{pct(n_risk)} of all items</div>
    <div class="kpi-sub">Cover drops below 7 months within 7-month horizon</div>
  </div>
  <div class="kpi-card kpi-red">
    <div class="kpi-label">🔴 Critical Risk Score</div>
    <div class="kpi-value">{n_crit}</div>
    <div class="kpi-pct">{pct(n_crit)} of all items</div>
    <div class="kpi-sub">ML risk score ≥ 70 · immediate action required</div>
  </div>
  <div class="kpi-card kpi-orange">
    <div class="kpi-label">🟠 High Risk Score</div>
    <div class="kpi-value">{n_high}</div>
    <div class="kpi-pct">{pct(n_high)} of all items</div>
    <div class="kpi-sub">ML risk score 45–69 · review this week</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Drill-down tables ──────────────────────────────────────
            st.divider()
            d1, d2, d3 = st.tabs(["🚫 Out of Stock", "⚠️ Below Safety Stock", "📉 Potential Risk Items"])

            with d1:
                oos = kpis.get("oos_items", [])
                if not oos:
                    st.success("✅ No out-of-stock items detected.")
                else:
                    st.markdown(f"**{len(oos)} items with zero On Hand stock and no alternative:**")
                    oos_df = pivot[pivot["item"].isin(oos)][["item","description"]].drop_duplicates("item")
                    st.dataframe(oos_df.reset_index(drop=True), use_container_width=True)
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w: oos_df.to_excel(w, index=False)
                    st.download_button("📥 Download OOS list", data=buf.getvalue(), file_name="out_of_stock.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with d2:
                bss = kpis.get("bss_items", [])
                if not bss:
                    st.success("✅ All items have sufficient coverage or incoming supply.")
                else:
                    st.markdown(f"**{len(bss)} items below safety stock with no incoming PO:**")
                    cover_rows_all = pivot[pivot["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
                    mc_all = sorted([c for c in pivot.columns if str(c).startswith("202")])
                    bss_rows = []
                    for it in bss:
                        desc = pivot[pivot["item"]==it]["description"].dropna()
                        desc = desc.iloc[0] if not desc.empty else ""
                        cr = cover_rows_all[cover_rows_all["item"]==it]
                        if not cr.empty and kpis.get("first_oh") and kpis["first_oh"] in cr.columns:
                            cov = float(pd.to_numeric(cr[kpis["first_oh"]].values[0], errors="coerce") or 0)
                        else:
                            cov = 0.0
                        bss_rows.append({"item": it, "description": desc, "current_cover_mo": round(cov,1), "incoming_po_7mo": "None"})
                    bss_df = pd.DataFrame(bss_rows)
                    def bss_style(val):
                        if isinstance(val, (int,float)) and val < 7: return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                        return ""
                    st.dataframe(bss_df.style.map(bss_style, subset=["current_cover_mo"]).format({"current_cover_mo":"{:.1f}"}), use_container_width=True)
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w: bss_df.to_excel(w, index=False)
                    st.download_button("📥 Download BSS list", data=buf.getvalue(), file_name="below_safety_stock.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with d3:
                risk_items = kpis.get("risk_items", [])
                if not risk_items:
                    st.success("✅ No items with coverage risk in the next 7 months.")
                else:
                    st.markdown(f"**{len(risk_items)} items with coverage dropping below 7 months in the next 7-month window:**")
                    cover_rows_all = pivot[pivot["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
                    future_7 = kpis.get("future_7", [])
                    risk_rows = []
                    for it in risk_items:
                        desc = pivot[pivot["item"]==it]["description"].dropna()
                        desc = desc.iloc[0] if not desc.empty else ""
                        cr = cover_rows_all[cover_rows_all["item"]==it]
                        row_d = {"item": it, "description": desc}
                        if not cr.empty:
                            for m in future_7:
                                if m in cr.columns:
                                    row_d[m] = round(float(pd.to_numeric(cr[m].values[0], errors="coerce") or 0), 1)
                        risk_rows.append(row_d)
                    risk_detail_df = pd.DataFrame(risk_rows)
                    mo_cols = [c for c in risk_detail_df.columns if c in future_7]
                    def cov_color(val):
                        if isinstance(val,(int,float)) and val < 7 and val > 0: return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                        if isinstance(val,(int,float)) and val == 0: return "color:#94a3b8"
                        return ""
                    st.dataframe(risk_detail_df.style.map(cov_color, subset=mo_cols).format({m:"{:.1f}" for m in mo_cols}, na_rep="–"), use_container_width=True)
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w: risk_detail_df.to_excel(w, index=False)
                    st.download_button("📥 Download Risk Items list", data=buf.getvalue(), file_name="potential_risk_items.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ── ML Risk charts (existing) ──────────────────────────────
            st.divider()
            st.subheader("🧮 ML Risk Score Analysis")
            st.subheader("📋 Full Risk Table")
            lf = st.multiselect("Filter by risk level", risk_df["risk_level"].unique().tolist(), default=risk_df["risk_level"].unique().tolist(), key="risk_filter")
            fr = risk_df[risk_df["risk_level"].isin(lf)] if not risk_df.empty else pd.DataFrame()
            def color_risk(val):
                if "Critical" in str(val): return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                if "High"     in str(val): return "background-color:#ffedd5;color:#9a3412;font-weight:600"
                if "Medium"   in str(val): return "background-color:#fef9c3;color:#713f12;font-weight:600"
                if "Low"      in str(val): return "background-color:#dcfce7;color:#14532d"
                return ""
            if not fr.empty:
                st.dataframe(fr[["item","description","risk_level","risk_score","min_cover","avg_cover","months_below_7","months_negative_inv","trend","flags"]]
                    .style.map(color_risk, subset=["risk_level"])
                    .format({"risk_score":"{:.1f}","min_cover":"{:.1f}","avg_cover":"{:.1f}"}),
                    use_container_width=True, height=400)

                # ── Score Breakdown ────────────────────────────────
                st.divider()
                st.subheader("🔎 Explain Risk Score for Specific SKU")
                explain_item = st.selectbox(
                    "Select item to explain:",
                    options=[""] + fr["item"].tolist(),
                    format_func=lambda x: x if x == "" else f"{x} — {fr[fr['item']==x]['description'].values[0] if len(fr[fr['item']==x]) > 0 else ''}",
                    key="explain_item_sel"
                )
                if explain_item:
                    row = fr[fr["item"] == explain_item]
                    if not row.empty:
                        r = row.iloc[0]
                        pv = st.session_state.pivot_df
                        mc = sorted([c for c in pv.columns if str(c).startswith("202")]) if pv is not None else []

                        # Recompute score components for display
                        min_c   = float(r["min_cover"])
                        avg_c   = float(r["avg_cover"])
                        neg_inv = int(r["months_negative_inv"])
                        trend   = str(r["trend"])

                        cover_score = round(min(40, max(0, (7 - min_c) / 7 * 40)) if min_c < 7 else 0, 1)
                        neg_score   = min(30, neg_inv * 8)
                        trend_score = 20 if ("Declining" in trend and min_c < 15) else 0
                        # Recompute volatility
                        cov_row = pv[(pv["item"] == explain_item) & (pv["ORDER_TYPE_FINAL"] == "COVER_MONTHS")] if pv is not None else pd.DataFrame()
                        cover_std = 0.0
                        if not cov_row.empty:
                            cv = pd.to_numeric(cov_row[mc].values.flatten(), errors="coerce")
                            cv = cv[~np.isnan(cv) & (cv != 0)]
                            cover_std = float(cv.std()) if len(cv) >= 3 else 0.0
                        vol_score = round(min(10, cover_std / max(avg_c, 1) * 10), 1) if avg_c > 0 else 0

                        total = cover_score + neg_score + trend_score + vol_score

                        st.markdown(f"### {r['risk_level']} &nbsp; **{explain_item}** — {r['description']}")
                        st.markdown(f"**Total Risk Score: {total:.1f} / 100**")

                        # Score bar chart
                        score_data = pd.DataFrame([
                            {"Component": "📉 Coverage depth",    "Score": cover_score, "Max": 40,
                             "Explanation": f"Min cover = {min_c:.1f} mo {'(< 7 → penalty)' if min_c < 7 else '(≥ 7 → no penalty)'}"},
                            {"Component": "📦 Negative inventory", "Score": neg_score,   "Max": 30,
                             "Explanation": f"{neg_inv} months with negative INV × 8pts each"},
                            {"Component": "📉 Declining trend",    "Score": trend_score, "Max": 20,
                             "Explanation": f"Trend = {trend} {'+ cover < 15mo → 20pts' if trend_score > 0 else '→ 0pts'}"},
                            {"Component": "〰️ Volatility",         "Score": vol_score,   "Max": 10,
                             "Explanation": f"Cover std = {cover_std:.1f} / avg {avg_c:.1f} = {cover_std/max(avg_c,1):.0%}"},
                        ])

                        if HAS_DEPS:
                            import plotly.express as px
                            fig = px.bar(
                                score_data, x="Score", y="Component", orientation="h",
                                color="Score",
                                color_continuous_scale=["#22c55e","#eab308","#f97316","#dc2626"],
                                range_color=[0, 40],
                                range_x=[0, 40],
                                title=f"Risk Score Breakdown — {explain_item}",
                                text="Score",
                                hover_data=["Explanation","Max"],
                            )
                            fig.update_traces(textposition="outside")
                            fig.update_layout(height=280, showlegend=False,
                                              yaxis=dict(autorange="reversed"),
                                              xaxis=dict(title="Points scored"))
                            st.plotly_chart(fig, use_container_width=True)

                        # Detail cards
                        c1, c2, c3, c4 = st.columns(4)
                        def score_color(s, mx):
                            if s == 0:     return "🟢"
                            if s < mx*0.5: return "🟡"
                            if s < mx:     return "🟠"
                            return "🔴"

                        c1.metric(f"{score_color(cover_score,40)} Coverage",
                                  f"{cover_score:.0f} / 40 pts",
                                  f"Min cover: {min_c:.1f} mo | Avg: {avg_c:.1f} mo",
                                  delta_color="off")
                        c2.metric(f"{score_color(neg_score,30)} Negative INV",
                                  f"{neg_score:.0f} / 30 pts",
                                  f"{neg_inv} months negative",
                                  delta_color="off")
                        c3.metric(f"{score_color(trend_score,20)} Demand Trend",
                                  f"{trend_score:.0f} / 20 pts",
                                  trend,
                                  delta_color="off")
                        c4.metric(f"{score_color(vol_score,10)} Volatility",
                                  f"{vol_score:.1f} / 10 pts",
                                  f"Std: {cover_std:.1f}",
                                  delta_color="off")

                        st.markdown(f"**Flags:** {r['flags']}")

                        # Show item's cover row from pivot
                        if pv is not None and not cov_row.empty:
                            st.markdown("**Coverage values by month:**")
                            cover_display = cov_row[mc].apply(pd.to_numeric, errors="coerce")
                            st.dataframe(
                                cover_display.style
                                    .format("{:.1f}", na_rep="–")
                                    .map(lambda v: "background-color:#dc2626;color:white;font-weight:bold"
                                                  if isinstance(v, (int,float)) and not np.isnan(v) and v < 7
                                                  else ("background-color:#f0fdf4" if isinstance(v,(int,float)) and not np.isnan(v) and v >= 7 else "")),
                                use_container_width=True,
                            )

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w: fr.to_excel(w, index=False)
            st.download_button("📥 Download Risk Report", data=buf.getvalue(), file_name="mrp_risk_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── Anomaly Detection ───────────────────────────────────────────
        with a_tab2:
            st.subheader("🔍 Anomaly Detection — Isolation Forest")
            st.info("**How it works:** Isolation Forest builds random trees and isolates each point. Anomalies are isolated faster (fewer splits) because they are rare and different. Higher anomaly score = more anomalous.")
            c1, c2 = st.columns([1,3])
            sensitivity = c1.slider("Sensitivity", 5, 30, 10, 5, help="% of data expected to be anomalous") / 100
            c2.markdown(f"**Contamination = {sensitivity:.0%}** — flags the most isolated `{sensitivity:.0%}` of data points.")
            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_anomalies(_plen, _rm_len, sens):
                return detect_anomalies(get_rm_pivot(st.session_state.pivot_df), contamination=sens) if HAS_DEPS else pd.DataFrame()
            with st.spinner("Running Isolation Forest…"):
                anomaly_df = _cached_anomalies(len(pivot), n_rm, sensitivity)
            if anomaly_df.empty:
                st.warning("Not enough data (need ≥4 data points per item).")
            else:
                n_an = anomaly_df["anomaly"].sum()
                a1,a2,a3 = st.columns(3)
                a1.metric("Total data points", len(anomaly_df))
                a2.metric("Anomalies detected", int(n_an))
                a3.metric("Anomaly rate", f"{n_an/len(anomaly_df)*100:.1f}%")
                items_with_an = ["All"] + anomaly_df[anomaly_df["anomaly"]]["item"].unique().tolist()
                sel_an = st.selectbox("Select item to inspect", items_with_an, key="anomaly_item")
                view_an = anomaly_df if sel_an == "All" else anomaly_df[anomaly_df["item"] == sel_an]
                if HAS_DEPS:
                    if sel_an == "All":
                        # Show summary bar chart when all items selected (avoids facet spacing error)
                        anom_summary = (view_an[view_an["anomaly"]]
                            .groupby("item")["anomaly_score"].count()
                            .reset_index().rename(columns={"anomaly_score":"count"})
                            .sort_values("count", ascending=False).head(30))
                        if not anom_summary.empty:
                            fig = px.bar(anom_summary, x="count", y="item", orientation="h",
                                title="Anomaly Count by Item (Top 30)",
                                labels={"count":"# Anomalies","item":"Item"},
                                color="count", color_continuous_scale=["#fbbf24","#dc2626"])
                            fig.update_layout(height=min(600, max(300, len(anom_summary)*22)),
                                yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                        st.caption("Select a specific item above to see its time-series chart.")
                    else:
                        fig = px.scatter(view_an, x="month", y="value", color="anomaly",
                            color_discrete_map={True:"#dc2626",False:"#3b82f6"},
                            size="anomaly_score", size_max=20,
                            hover_data=["item","description","anomaly_score"],
                            title=f"Anomaly Detection — {sel_an}")
                        fig.update_layout(height=380)
                        st.plotly_chart(fig, use_container_width=True)
                st.dataframe(view_an[view_an["anomaly"]][["item","description","month","value","anomaly_score"]]
                    .sort_values("anomaly_score", ascending=False)
                    .style.format({"value":"{:,.2f}","anomaly_score":"{:.3f}"}), use_container_width=True)

        # ── Spike Alerts ──────────────────────────────────────────────────
        with a_tab6:
            st.subheader("🚀 Demand Spike Alerts — Forward-Looking Forecast")
            st.caption("🔎 מצא פריטים עם חריגת Spike בביקוש קדימה — find SKUs whose **future forecast** spikes above their own baseline.")
            st.info("**How it works:** for each SKU, the baseline is the mean/std of its forecast history "
                    "up to the current On-Hand month. A future month is flagged when it is both well above "
                    "the % threshold **and** the σ (z-score) threshold below.")
            sp1, sp2 = st.columns(2)
            spike_pct = sp1.slider("Min. % above baseline", 20, 200, 60, 10, help="e.g. 60% = forecast must be ≥60% above the item's baseline", key="spike_pct") / 100
            spike_z   = sp2.slider("Min. z-score (σ above baseline)", 0.5, 4.0, 2.0, 0.5, help="How many standard deviations above baseline", key="spike_z")

            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_spikes(_plen, _rm_len, pct, z):
                return compute_demand_spikes(get_rm_pivot(st.session_state.pivot_df), z_thresh=z, pct_thresh=pct) if HAS_DEPS else pd.DataFrame()

            with st.spinner("Scanning forward forecast for spikes…"):
                spike_df = _cached_spikes(len(pivot), n_rm, spike_pct, spike_z)

            if spike_df.empty:
                st.success("✅ No forward demand spikes detected at the current thresholds.")
            else:
                n_spike_items  = spike_df["item"].nunique()
                n_spike_months = len(spike_df)
                sk1, sk2, sk3 = st.columns(3)
                sk1.metric("🚩 SKUs flagged", n_spike_items)
                sk2.metric("📅 Spike months", n_spike_months)
                sk3.metric("📈 Max % above baseline", f"{spike_df['pct_above_baseline'].max():.0f}%")

                item_summary = (spike_df.groupby(["item","description"])
                                 .agg(max_pct=("pct_above_baseline","max"),
                                      n_months=("month","count"),
                                      first_spike_month=("month","min"))
                                 .reset_index().sort_values("max_pct", ascending=False))

                if HAS_DEPS:
                    top_sp = item_summary.head(30)
                    fig_sp = px.bar(top_sp, x="max_pct", y="item", orientation="h",
                        title="Top Spike SKUs — Max % Above Baseline (forward months)",
                        labels={"max_pct":"% above baseline","item":"Item"},
                        color="max_pct", color_continuous_scale=["#fbbf24","#dc2626"],
                        hover_data=["description","n_months","first_spike_month"])
                    fig_sp.update_layout(height=min(650, max(320, len(top_sp)*24)), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_sp, use_container_width=True)

                sel_sp_item = st.selectbox("🔎 Drill into SKU forecast", ["All"] + item_summary["item"].tolist(), key="spike_item_sel")
                if sel_sp_item != "All" and HAS_DEPS:
                    pv_sp = get_rm_pivot(st.session_state.pivot_df)
                    mc_sp = sorted([c for c in pv_sp.columns if str(c).startswith("202")])
                    fr_sp = pv_sp[(pv_sp["item"]==sel_sp_item) & (pv_sp["ORDER_TYPE_FINAL"]=="1.Forecast")]
                    if not fr_sp.empty:
                        f_series = pd.to_numeric(fr_sp[mc_sp].iloc[0], errors="coerce")
                        spike_months_set = set(spike_df[spike_df["item"]==sel_sp_item]["month"])
                        colors = ["#dc2626" if m in spike_months_set else "#3b82f6" for m in mc_sp]
                        fig_sp2 = go.Figure()
                        fig_sp2.add_trace(go.Bar(x=mc_sp, y=f_series.values, name="Forecast", marker_color=colors))
                        baseline_val = spike_df[spike_df["item"]==sel_sp_item]["baseline"]
                        if not baseline_val.empty:
                            fig_sp2.add_hline(y=float(baseline_val.iloc[0]), line_dash="dash", line_color="#94a3b8",
                                               annotation_text="baseline")
                        fig_sp2.update_layout(title=f"Forecast — {sel_sp_item} (red = flagged spike month)", height=360,
                                               xaxis_title="Month", yaxis_title="Qty")
                        st.plotly_chart(fig_sp2, use_container_width=True)

                st.markdown("**Flagged spike months (detail):**")
                st.dataframe(
                    spike_df[["item","description","month","forecast","baseline","pct_above_baseline","z_score"]]
                    .style.format({"forecast":"{:,.2f}","baseline":"{:,.2f}","pct_above_baseline":"{:.1f}%","z_score":"{:.2f}"}, na_rep="–"),
                    use_container_width=True, height=350,
                )
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w: spike_df.to_excel(w, index=False)
                st.download_button("📥 Download Spike Report", data=buf.getvalue(), file_name="mrp_demand_spikes.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── Trend Analysis ──────────────────────────────────────────────
        with a_tab3:
            st.subheader("📈 Demand Trend Analysis — Linear Regression")
            st.info("**Model:** OLS regression on actual monthly consumption. **R²** = goodness-of-fit. **Normalized slope** = monthly change % vs average.")
            with st.spinner("Fitting linear regression per item…"):
                t_df, _ = _cached_risk(len(pivot)) if HAS_DEPS else (pd.DataFrame(), pd.DataFrame())
            if t_df.empty:
                st.warning("No actual consumption data found.")
            else:
                t1,t2,t3 = st.columns(3)
                t1.metric("📈 Growing",   int((t_df["trend_label"]=="📈 Growing").sum()))
                t2.metric("➡️ Stable",   int((t_df["trend_label"]=="➡️ Stable").sum()))
                t3.metric("📉 Declining", int((t_df["trend_label"]=="📉 Declining").sum()))
                tf = st.multiselect("Filter trend", t_df["trend_label"].unique().tolist(), default=t_df["trend_label"].unique().tolist(), key="trend_filter")
                vt = t_df[t_df["trend_label"].isin(tf)]
                if HAS_DEPS:
                    fig = px.bar(vt.sort_values("norm_slope"), x="norm_slope", y="item", orientation="h",
                        color="trend_label", color_discrete_map={"📈 Growing":"#22c55e","➡️ Stable":"#94a3b8","📉 Declining":"#dc2626"},
                        title="Normalized Demand Slope (monthly % change)",
                        hover_data=["description","avg_value","last_value","r2","n_months"])
                    fig.update_layout(height=max(400, len(vt)*22), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                sel_t = st.selectbox("Deep-dive into item trend", vt["item"].tolist(), key="trend_item_sel")
                if sel_t and HAS_DEPS:
                    s  = extract_item_series(pivot, "actual")
                    sd = s[s["item"] == sel_t].sort_values("month")
                    if not sd.empty:
                        x = np.arange(len(sd)).reshape(-1,1); y = sd["value"].values
                        lr = LinearRegression().fit(x, y); sd = sd.copy(); sd["trend_line"] = lr.predict(x)
                        row = vt[vt["item"] == sel_t].iloc[0]
                        st.markdown(f"**{sel_t}** | {row['description']} | {row['trend_label']} | R²={row['r2']} | Avg={row['avg_value']:,.1f}")
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(x=sd["month"].astype(str), y=sd["value"], name="Actual", marker_color="#3b82f6"))
                        fig2.add_trace(go.Scatter(x=sd["month"].astype(str), y=sd["trend_line"], name="Trend", line=dict(color="#dc2626",width=2,dash="dash")))
                        fig2.update_layout(title=f"Demand Trend — {sel_t}", height=350)
                        st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(vt.style.format({"norm_slope":"{:.3f}","slope":"{:.3f}","r2":"{:.3f}","avg_value":"{:,.2f}","last_value":"{:,.2f}"}), use_container_width=True)

        # ── Forecast Accuracy ───────────────────────────────────────────
        with a_tab4:
            st.subheader("🎯 Forecast Accuracy — MAPE Analysis")
            st.info("**MAPE** = Mean Absolute Percentage Error (lower is better). Bias > 0 = under-forecasting. Bias < 0 = over-forecasting.")
            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_accuracy(_plen, _rm_len):
                return compute_forecast_accuracy(get_rm_pivot(st.session_state.pivot_df)) if HAS_DEPS else pd.DataFrame()
            with st.spinner("Computing forecast vs actual…"):
                acc_df = _cached_accuracy(len(pivot), n_rm)
            if acc_df.empty:
                st.warning("No overlapping forecast and actual data found.")
            else:
                ov = acc_df["mape"].dropna().mean(); bias = acc_df["error"].mean()
                f1,f2,f3 = st.columns(3)
                f1.metric("Overall MAPE", f"{ov:.1f}%"); f2.metric("Avg Bias (A-F)", f"{bias:+.2f}"); f3.metric("Items analyzed", acc_df["item"].nunique())
                if HAS_DEPS:
                    im = acc_df.groupby(["item","description"])["mape"].mean().reset_index().sort_values("mape", ascending=False).head(20)
                    fig = px.bar(im, x="mape", y="item", orientation="h", color="mape",
                        color_continuous_scale=["#22c55e","#eab308","#dc2626"],
                        title="MAPE by Item — Top 20 Worst Accuracy", hover_data=["description"])
                    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    me = acc_df.groupby("month")[["error","abs_error","mape"]].mean().reset_index()
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=me["month"], y=me["abs_error"], name="Avg Abs Error", marker_color="#f97316"))
                    fig2.add_trace(go.Scatter(x=me["month"], y=me["mape"], name="MAPE %", yaxis="y2", line=dict(color="#7c3aed")))
                    fig2.update_layout(title="Forecast Error Over Time", yaxis2=dict(overlaying="y",side="right"), height=350)
                    st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(acc_df.style.format({"forecast":"{:,.2f}","actual":"{:,.2f}","error":"{:+.2f}","abs_error":"{:,.2f}","mape":"{:.1f}%"}), use_container_width=True, height=300)

        # ── AI Data Chat ────────────────────────────────────────────────
        with a_tab5:
            st.subheader("💬 MRP AI Assistant")
            st.caption("Ask anything about your supply chain data in English or Hebrew. The assistant has full access to risk scores, trends, anomalies, coverage and PO recommendations.")

            if st.session_state.get("chat_history") is None:
                st.session_state.chat_history = []

            @st.cache_data(ttl=600, show_spinner=False, max_entries=3)
            def get_all_insights(_pivot_len, _rm_len):
                pv = get_rm_pivot(st.session_state.pivot_df)
                if pv is None or not HAS_DEPS: return {}
                mc = sorted([c for c in pv.columns if str(c).startswith("202")])
                t_df  = compute_trends(pv)
                r_df  = compute_risk(pv, t_df)
                an_df = detect_anomalies(pv, contamination=0.1)
                fa_df = compute_forecast_accuracy(pv)
                cv    = pv[pv["ORDER_TYPE_FINAL"]=="COVER_MONTHS"]
                cv_v  = pd.to_numeric(cv[mc].values.flatten(), errors="coerce") if not cv.empty else np.array([])
                cv_v  = cv_v[~np.isnan(cv_v)]
                inv_r = pv[pv["ORDER_TYPE_FINAL"]=="INV"]
                po_r  = pv[pv["ORDER_TYPE_FINAL"]=="PO_RECOMMENDATION"]
                po_items = []
                if not po_r.empty:
                    for _, row in po_r.iterrows():
                        vals = pd.to_numeric(row[mc], errors="coerce")
                        tot  = vals[vals > 0].sum()
                        if tot > 0: po_items.append({"item": row["item"], "description": row.get("description",""), "total_po_recommended": round(tot,0)})
                return {
                    "items": pv["item"].dropna().unique().tolist(), "months": mc,
                    "n_items": pv["item"].nunique(), "n_months": len(mc),
                    "risk": r_df, "trends": t_df, "anomalies": an_df, "accuracy": fa_df,
                    "cv_mean": round(float(cv_v.mean()),1) if len(cv_v)>0 else None,
                    "cv_min":  round(float(cv_v.min()),1)  if len(cv_v)>0 else None,
                    "items_below_7": int((pv[pv["ORDER_TYPE_FINAL"]=="COVER_MONTHS"][mc].apply(pd.to_numeric,errors="coerce").lt(7).any(axis=1)).sum()) if not cv.empty else 0,
                    "inv_negative_items": int((inv_r[mc].apply(pd.to_numeric,errors="coerce").lt(0).any(axis=1)).sum()) if not inv_r.empty else 0,
                    "po_items": po_items,
                    "n_anomalies": int(an_df["anomaly"].sum()) if not an_df.empty else 0,
                    "overall_mape": round(float(fa_df["mape"].dropna().mean()),1) if not fa_df.empty else None,
                }

            with st.spinner("Preparing ML insights…"):
                insights = get_all_insights(len(pivot), n_rm)

            def build_system_prompt(ins):
                if not ins: return "You are an MRP supply chain analyst. No data loaded."
                NL = "\n"
                r_df = ins.get("risk", pd.DataFrame())
                t_df = ins.get("trends", pd.DataFrame())
                an_df = ins.get("anomalies", pd.DataFrame())
                fa_df = ins.get("accuracy", pd.DataFrame())
                po    = ins.get("po_items", [])
                p = f"""You are an expert MRP supply chain analyst with direct access to the full dataset.
Answer with specific SKU numbers and values. Use markdown tables. Explain business implications and suggest actions.
If the user writes in Hebrew, respond in Hebrew.

=== DATASET: {ins["n_items"]} items | {ins["months"][0] if ins["months"] else "?"} → {ins["months"][-1] if ins["months"] else "?"} ({ins["n_months"]} months) ===
- Items with COVER < 7 months: {ins["items_below_7"]}
- Items with negative INV: {ins["inv_negative_items"]}
- Anomalies detected: {ins["n_anomalies"]}
- Forecast MAPE: {ins["overall_mape"]}% | Avg coverage: {ins["cv_mean"]} months | Min coverage: {ins["cv_min"]} months
- PO_RECOMMENDATION logic: trigger at <7 months coverage, fill to 15 months. COVER_MONTHS_UPDATED tracks updated coverage after PO injection.
"""
                if not r_df.empty:
                    crit = r_df[r_df["risk_level"]=="🔴 Critical"]
                    high = r_df[r_df["risk_level"]=="🟠 High"]
                    NL2 = "\n"
                    p += f"{NL2}=== RISK (score 0-100: coverage40+negINV30+trend20+vol10) ==={NL2}"
                    p += f"Critical ({len(crit)}): {', '.join(crit['item'].tolist()[:15])}{NL2}"
                    p += f"High ({len(high)}): {', '.join(high['item'].tolist()[:10])}{NL2}"
                    p += r_df[["item","description","risk_level","risk_score","min_cover","avg_cover","months_below_7","trend","flags"]].to_string(index=False) + NL2
                if not t_df.empty:
                    p += f"{NL}=== TRENDS ==={NL}"
                    p += t_df[["item","description","trend_label","norm_slope","r2","avg_value"]].to_string(index=False) + NL
                if not an_df.empty and "anomaly" in an_df.columns:
                    anom = an_df[an_df["anomaly"]][["item","description","month","value","anomaly_score"]].sort_values("anomaly_score",ascending=False)
                    p += f"{NL}=== ANOMALIES ==={NL}" + anom.head(30).to_string(index=False) + NL
                if po:
                    p += f"{NL}=== PO RECOMMENDATIONS ==={NL}"
                    p += NL.join([f"- {x['item']} ({x['description']}): {x['total_po_recommended']:,.0f} units" for x in po[:20]])
                return p

            # Suggested questions
            st.markdown("**💡 Quick questions:**")
            qs = ["Which items are at critical risk?","Show items below 3 months coverage",
                  "Which items have declining demand?","Where are the anomalies?",
                  "What POs do I need urgently?","Worst forecast accuracy items?",
                  "Summarize supply chain health","מה הסיכונים העיקריים?"]
            qcols = st.columns(4)
            for i, q in enumerate(qs):
                if qcols[i%4].button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    st.rerun()

            st.divider()

            # Display history
            for i, msg in enumerate(st.session_state.get("chat_history", [])):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "chart_data" in msg and HAS_DEPS:
                        cd = msg["chart_data"]
                        try:
                            fig = px.bar(pd.DataFrame(cd["data"]), x=cd["x"], y=cd["y"],
                                color=cd.get("color"), orientation="h", title=cd.get("title",""),
                                color_discrete_map={"🔴 Critical":"#dc2626","🟠 High":"#f97316","🟡 Medium":"#eab308","🟢 Low":"#22c55e","📈 Growing":"#22c55e","➡️ Stable":"#94a3b8","📉 Declining":"#dc2626"})
                            fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True, key=f"hchart_{i}")
                        except Exception: pass

            # Process pending message
            hist = st.session_state.get("chat_history", [])
            if hist and hist[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing MRP data…"):
                        history_for_api = [{"role":m["role"],"content":m["content"]} for m in hist[-12:] if m["role"] in ("user","assistant")]
                        sys_prompt = build_system_prompt(insights)
                        resp_text = ""; chart_data = None
                        try:
                            from openai import OpenAI as _OAI
                            import os as _os
                            _token = _os.environ.get("DATABRICKS_TOKEN","")
                            if not _token:
                                raise ValueError("DATABRICKS_TOKEN not set")
                            _cli = _OAI(
                                api_key=_token,
                                base_url="https://adb-7058912952674206.6.azuredatabricks.net/serving-endpoints",
                            )
                            _msgs = [{"role":"system","content":sys_prompt}] + history_for_api
                            _resp = _cli.chat.completions.create(
                                model="databricks-meta-llama-3-1-70b-instruct",
                                messages=_msgs,
                                max_tokens=2000,
                            )
                            resp_text = _resp.choices[0].message.content
                            ql = hist[-1]["content"].lower()
                            if any(w in ql for w in ["risk","critical","score"]) and not insights.get("risk",pd.DataFrame()).empty:
                                chart_data = {"type":"bar","data":insights["risk"].head(15)[["item","risk_score","risk_level"]].to_dict("records"),"x":"risk_score","y":"item","color":"risk_level","title":"Risk Scores — Top 15 Items"}
                            elif any(w in ql for w in ["trend","grow","declin"]) and not insights.get("trends",pd.DataFrame()).empty:
                                chart_data = {"type":"bar","data":insights["trends"][["item","norm_slope","trend_label"]].to_dict("records"),"x":"norm_slope","y":"item","color":"trend_label","title":"Demand Trends"}
                        except Exception as _e:
                            resp_text = _local_fallback(hist[-1]["content"], insights)
                        st.markdown(resp_text)
                        if chart_data and HAS_DEPS:
                            try:
                                fig = px.bar(pd.DataFrame(chart_data["data"]), x=chart_data["x"], y=chart_data["y"],
                                    color=chart_data.get("color"), orientation="h", title=chart_data.get("title",""),
                                    color_discrete_map={"🔴 Critical":"#dc2626","🟠 High":"#f97316","🟡 Medium":"#eab308","🟢 Low":"#22c55e","📈 Growing":"#22c55e","➡️ Stable":"#94a3b8","📉 Declining":"#dc2626"})
                                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception: pass
                        entry = {"role":"assistant","content":resp_text}
                        if chart_data: entry["chart_data"] = chart_data
                        st.session_state.chat_history.append(entry)
                        st.rerun()

            user_q = st.chat_input("Ask about your supply chain… e.g. 'Which items need a PO urgently?' or 'מה הסיכונים?'")
            if user_q:
                st.session_state.chat_history.append({"role":"user","content":user_q})
                st.rerun()

            cc1, cc2 = st.columns([1,4])
            if cc1.button("🗑️ Clear chat", key="clr_chat"):
                st.session_state.chat_history = []; st.rerun()
            cc2.caption(f"💬 {len(hist)//2} messages | {insights.get('n_items',0)} items | {insights.get('n_anomalies',0)} anomalies | MAPE: {insights.get('overall_mape','N/A')}%")




with tab_data:
    st.subheader("Data Preview")

    # ── Master Data viewer ──────────────────────────────────────
    if hasattr(st.session_state, "master_df") and st.session_state.master_df is not None:
        with st.expander("📋 Master Data (mrp_current.csv) — LT / SS / SL / Max Inv", expanded=False):
            md = st.session_state.master_df.reset_index()
            pc_filter = st.multiselect("Filter by Planner Code",
                sorted(md["planner_code"].dropna().unique().tolist()),
                default=["RM1","RM2"], key="md_pc_filter")
            md_view = md[md["planner_code"].isin(pc_filter)] if pc_filter else md

            def color_ss(val):
                try:
                    v = float(val)
                    if v <= 3:  return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                    if v <= 6:  return "background-color:#ffedd5;color:#9a3412"
                    if v >= 12: return "background-color:#dcfce7;color:#14532d"
                except: pass
                return ""
            def color_lt(val):
                try:
                    v = float(val)
                    if v >= 8: return "background-color:#fff3cd;color:#856404;font-weight:600"
                except: pass
                return ""

            st.dataframe(
                md_view[["item","planner_code","lead_time","safety_stock","max_inventory","shelf_life","list_price","abc_class"]]
                .style
                .map(color_ss, subset=["safety_stock"])
                .map(color_lt, subset=["lead_time"])
                .format({"lead_time":"{:.2f}","safety_stock":"{:.0f}","max_inventory":"{:.0f}","shelf_life":"{:.0f}","list_price":"{:,.2f}"}, na_rep="N/A"),
                use_container_width=True, height=350,
            )
            st.caption("🟡 LT ≥ 8 months (long lead time)  |  🔴 SS ≤ 3 months (very low safety stock)  |  🟢 SS ≥ 12 months")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w: md_view.to_excel(w, index=False)
            st.download_button("📥 Download Master Data", data=buf.getvalue(), file_name="mrp_master_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.session_state.full_bom is not None:
        with st.expander(f"🔩 BOM — {len(st.session_state.full_bom):,} rows", expanded=False):
            bom_view = st.session_state.full_bom

            # Filter by item for easy inspection
            bom_items = sorted(bom_view["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in bom_view.columns else []
            sel_bom   = st.selectbox("Filter BOM by product", ["All"] + bom_items, key="bom_prod_filter")
            if sel_bom != "All":
                bom_view = bom_view[bom_view["PRODUCT"] == sel_bom]

            # Highlight key conversion columns
            show_cols = [c for c in ["PRODUCT","INGREDIENT","INGREDIENT_DESCRIPTION",
                                      "PRUDUCT_QTY","PRODUCT_UOM","ING_QTY","ING_UOM",
                                      "CONVERSION_RATE","BOM_RATIO","BOM_CONV","CONV_STATUS",
                                      "level"] if c in bom_view.columns]
            fmt = {}
            for c in ["ING_QTY","PRUDUCT_QTY","CONVERSION_RATE","BOM_RATIO","BOM_CONV"]:
                if c in show_cols: fmt[c] = "{:,.6f}"

            def highlight_conv(val):
                if str(val) == "Missing Conversion": return "background-color:#fee2e2;color:#991b1b"
                if str(val) == "Ratio only (no UOM conv)": return "background-color:#fff3cd;color:#856404"
                return ""

            styled = bom_view[show_cols].head(500).style.format(fmt, na_rep="–")
            if "CONV_STATUS" in show_cols:
                styled = styled.map(highlight_conv, subset=["CONV_STATUS"])
            st.dataframe(styled, use_container_width=True)

            st.caption(
                "**BOM_CONV formula:** `(ING_QTY / PRODUCT_QTY) × CONVERSION_RATE`  "
                "— normalized to **1 unit** of parent product. "
                "**BOM_RATIO** = ING_QTY / PRODUCT_QTY (before UOM conversion)."
            )

    if st.session_state.pivot_df is not None:
        with st.expander(f"📊 MRP Pivot — {len(st.session_state.pivot_df):,} rows", expanded=True):
            pv    = st.session_state.pivot_df
            all_m = sorted([c for c in pv.columns if str(c).startswith("202")])

            oh_rows = pv[pv["ORDER_TYPE_FINAL"].str.contains("on hand", case=False, na=False)]
            first_oh = None
            for m in all_m:
                if m in oh_rows.columns and pd.to_numeric(oh_rows[m], errors="coerce").sum() > 0:
                    first_oh = m
                    break

            pc1, pc2 = st.columns([3, 1])
            with pc1:
                items = pv["item"].dropna().unique().tolist()
                sel   = st.multiselect("Filter items", items, default=items[:5], key="preview_filter")
            with pc2:
                from_oh = st.checkbox(
                    "📅 From first On Hand month", value=True, key="preview_from_oh",
                    help=f"First On Hand month: {first_oh}" if first_oh else "Not detected",
                )

            view = pv.copy()
            if sel:
                view = view[view["item"].isin(sel)]

            order_sort_p = {
                "1.Forecast": 1, "2.ACTUAL": 2, "Planned order demand": 3,
                "Work order demand": 4, "Planned order": 6, "Purchase order": 9,
                "Purchase requisition": 10, "3.On Hand": 11, "INV": 12,
                "COVER_MONTHS": 13, "PO_RECOMMENDATION": 14, "COVER_MONTHS_UPDATED": 15, "Other": 99,
            }
            view["_sort_key"] = view["ORDER_TYPE_FINAL"].map(order_sort_p).fillna(50)
            view = view.sort_values(["item", "_sort_key"]).drop(columns="_sort_key")
            view["description"] = view.groupby("item")["description"].transform("first")

            show_months = all_m
            if from_oh and first_oh and first_oh in all_m:
                show_months = all_m[all_m.index(first_oh):]

            view = view[["item", "description", "ORDER_TYPE_FINAL"] + [c for c in show_months if c in view.columns]]

            import streamlit.components.v1 as _comp

            MAX_RENDER_ROWS2 = 400
            total_rows2 = len(view)
            if total_rows2 > MAX_RENDER_ROWS2:
                n_pages2 = (total_rows2 - 1) // MAX_RENDER_ROWS2 + 1
                pg2 = st.number_input(f"Page (showing {MAX_RENDER_ROWS2} rows/page, {total_rows2} total)", min_value=1, max_value=n_pages2, value=1, key="preview_pivot_page")
                page_view2 = view.iloc[(pg2-1)*MAX_RENDER_ROWS2 : pg2*MAX_RENDER_ROWS2]
            else:
                page_view2 = view

            n_rows2 = len(page_view2); h2 = min(680, max(280, n_rows2 * 29 + 40))
            _comp.html(render_pivot_html(page_view2, tuple(sorted(show_months)), frozenset()), height=h2, scrolling=True)
            if first_oh:
                st.caption(f"📅 First On Hand month: **{first_oh}**  |  🔴 COVER < 7  |  🟠 PO Recommendation  |  🔵 Coverage Updated")

# ──────────────────────────────────────────────────────────────
# TAB 3 – HELP
# ──────────────────────────────────────────────────────────────

with tab_help:
    st.markdown("""
## How to use

Data loads automatically when the page opens. No manual connection needed.

### Run Simulation

| Parameter | Description |
|-----------|-------------|
| **Parent Product** | Root SKU to explode through the BOM |
| **Action** | ADD or REMOVE units from forecast |
| **Forecast Month** | Month to update (YYYY-MM) |
| **Production Qty Change** | Number of units |
| **Search Mode** | DIRECT / DEEP / PARTIAL |
| **Products to Exclude** | Stop recursion at these SKUs (PARTIAL mode) |

### Table Colors

| Color | Meaning |
|-------|---------|
| 🔴 Red cell | COVER < 7 months (original or updated coverage) |
| 🟠 Orange cell | PO recommended — order qty to reach 15 months cover |
| 🔵 Blue row | Coverage Months Updated — projected coverage after PO injections |
| 🟡 Yellow row | Item updated by simulation |

### PO Recommendation Logic

The system continuously monitors coverage across the planning horizon.

| Parameter | Value |
|-----------|-------|
| **Trigger threshold** | < 7 months coverage |
| **Target coverage** | 15 months |
| **Cycle** | Repeats — whenever updated coverage drops below 7 months again, a new PO is generated |

The **Cover Updated** row tracks projected coverage *after* all recommended POs are applied, enabling you to verify the replenishment plan and spot any subsequent dips that trigger additional orders.

### Reload Data
Click **🔄 Reload Data** in the sidebar to refresh from the database.
    """)

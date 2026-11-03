"""
app.py  –  MRP / BOM Simulation Dashboard
Run:  streamlit run app.py
"""

import io
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess, numpy as np, pandas as pd, streamlit as st

def _ensure_packages():
    for pkg, imp in [("plotly","plotly"), ("scikit-learn","sklearn"), ("openai","openai")]:
        try: __import__(imp)
        except ImportError: subprocess.run([sys.executable,"-m","pip","install",pkg,"--quiet","--break-system-packages"], check=False)
_ensure_packages()

from src.db import load_full_bom, load_dwh_data, load_substitutes
from src.data_prep import enrich_bom, build_mrp_pivot
from src.simulation import (
    SimulationConfig, run_simulation, _compute_inv_cover as _compute_inv_cover_orig,
    _sum_by_item as _sim_sum_by_item, _window_mean as _sim_window_mean,
    compute_item_inv_cover,
)
from src.master_data import load_master, get_master_map
from src.sku_tracking import (
    load_tracking as load_sku_tracking,
    load_user_filters,
    save_user_filters,
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


def compute_po_recommendations(pivot_df: pd.DataFrame, month_cols: list, master_map: dict = None):
    """
    PO Recommendation logic — a reorder-point simulation, one order at a time:

    Repeat, for each item:
      1. Simulate INV/COVER forward with whatever POs have been placed so
         far — reusing compute_item_inv_cover(), the exact same engine
         behind the displayed COVER_MONTHS row, so this never drifts from
         what coverage actually does.
      2. Find the first future month coverage would drop below safety stock
         — that month is when the item is truly *needed*.
      3. Try to place the order LT months earlier so it arrives exactly in
         time. If that ideal placement month has already passed (there
         isn't LT months of runway left before the item is needed), place
         it as early as possible instead — the order will still arrive
         late. Any lateness, however small, is flagged as a PO_EXCEPTION
         (red) instead of a normal PO_RECOMMENDATION (orange), and the
         month it was truly needed is recorded so the UI can show it on
         hover. The order is never withheld for being late — recommending
         something late is still better than recommending nothing.
      4. Size an order to bring inventory up to MAX_INVENTORY (item_target)
         months of coverage AT THE POINT IT ARRIVES, capped by usable shelf
         life (shelf life minus lead time — stock ages in transit too,
         before it's even on the shelf).
      5. Inject the order and repeat. The next recommendation only fires once
         *that* stock genuinely depletes again — not a fixed countdown.

    Returns (pivot_df_with_po_rows, needed_by_map) — needed_by_map is
    {(item, place_month): trigger_month}, one entry per PO_EXCEPTION cell,
    for _build_cover_tooltips() to surface as a hover tooltip.
    """
    master_map = master_map or {}
    if not month_cols:
        return pivot_df, {}

    keep_mask = ~pivot_df["ORDER_TYPE_FINAL"].isin(
        ["PO_RECOMMENDATION", "PO_EXCEPTION", "COVER_MONTHS_UPDATED"]
    )
    pivot_df = pivot_df[keep_mask].copy()

    n = len(month_cols)
    order_type = pivot_df["ORDER_TYPE_FINAL"]
    is_fcst   = order_type.str.contains("forecast", case=False, na=False)
    is_po     = order_type.isin(["Purchase order"])
    is_onhand = order_type.str.contains("on hand|onhand", case=False, na=False) | (order_type == "3.On Hand")

    fcst_by_item   = _sim_sum_by_item(pivot_df, is_fcst,   month_cols)
    po_by_item     = _sim_sum_by_item(pivot_df, is_po,     month_cols)
    onhand_by_item = _sim_sum_by_item(pivot_df, is_onhand, month_cols)
    desc_map = pivot_df.groupby("item")["description"].apply(
        lambda s: s.dropna().iloc[0] if not s.dropna().empty else ""
    ).to_dict()

    new_rows: list[dict] = []
    needed_by_map: dict = {}  # (item, place_month) -> trigger_month, for PO_EXCEPTION tooltips

    for item in pivot_df["item"].dropna().unique():
        forecast_raw = fcst_by_item.get(item)
        onhand_raw   = onhand_by_item.get(item)
        if forecast_raw is None and onhand_raw is None:
            continue
        forecast_arr = forecast_raw if forecast_raw is not None else np.full(n, np.nan)

        _mp         = master_map.get(str(item), {})
        item_ss     = float(_mp.get("safety_stock",  COVER_TRIGGER) or COVER_TRIGGER)
        item_target = float(_mp.get("max_inventory",  COVER_TARGET)  or COVER_TARGET)
        item_lt     = float(_mp.get("lead_time",      0)             or 0)
        item_sl     = float(_mp.get("shelf_life",     9999)          or 9999)
        lt_months_int = max(0, int(round(item_lt)))

        po_raw = po_by_item.get(item)
        sim_po = np.nan_to_num(po_raw, nan=0.0) if po_raw is not None else np.zeros(n)

        po_rec      = np.zeros(n)
        po_exc      = np.zeros(n)
        needed_by   = {}  # place_idx -> (trigger_month, arrival_month, lt_shortfall), exception cells only
        search_from = 0  # advances every pass so we always make forward progress

        for _pass in range(20):  # generous cap on reorder cycles; guards against any runaway loop
            # cover_arr here (built from sim_po, which includes exception
            # quantities too) is only used to find where the NEXT trigger is —
            # not for display. The displayed "Cover Updated" curve is computed
            # separately below, from po_rec only.
            inv_arr, cover_arr, first_oh_idx, _detail = compute_item_inv_cover(
                forecast_arr, sim_po, onhand_raw, month_cols
            )
            if first_oh_idx is None:
                break
            search_from = max(search_from, first_oh_idx)

            # "beyond_horizon" cover values are an honest lower bound (we ran
            # out of forecast data to confirm coverage further), not a real
            # depletion signal — they shrink toward 0 near the tail of any
            # month_cols array regardless of actual risk, so they must never
            # be treated as a genuine below-SS trigger.
            trigger_idx = next(
                (i for i in range(search_from, n)
                 if not np.isnan(cover_arr[i]) and cover_arr[i] < item_ss
                 and _detail[i] is not None and _detail[i]["reason"] != "beyond_horizon"), None
            )
            if trigger_idx is None:
                break

            # Ideal, fully lead-time-compliant placement: LT months before the
            # item is truly needed, arriving exactly on trigger_idx. If that
            # placement month has already passed (place_idx gets clamped to
            # first_oh_idx), the order still goes out as early as possible —
            # it just arrives late.
            place_idx    = max(trigger_idx - lt_months_int, first_oh_idx)
            arrival_idx  = min(place_idx + lt_months_int, n - 1)
            lt_shortfall = max(0, arrival_idx - trigger_idx)

            # A supplier's LT is a commitment counted from the moment you
            # actually place the order — "today" (first_oh_idx), not from
            # this order's own placement month. So even an order that will
            # arrive perfectly on time (lt_shortfall == 0) still doesn't get
            # the supplier's full standard notice if it has to be placed
            # sooner than "today + LT" — flag that too, distinctly from
            # outright lateness.
            notice_shortfall = max(0, (first_oh_idx + lt_months_int) - place_idx)
            is_exception = lt_shortfall > 0 or notice_shortfall > 0

            # Local demand rate at arrival — same "avg forecast, next 7mo"
            # convention COVER_MONTHS itself uses, so sizing stays consistent
            # with how coverage is measured.
            local_window = np.nan_to_num(forecast_arr[arrival_idx:arrival_idx+7], nan=0.0)
            local_rate = float(local_window.mean()) if local_window.size else 0.0
            if local_rate <= 0:
                # Nothing to size an order against here — move past this month
                # and keep looking for a later, actionable depletion.
                search_from = trigger_idx + 1
                continue

            inv_at_arrival = inv_arr[arrival_idx] if not np.isnan(inv_arr[arrival_idx]) else 0.0
            needed_inv = local_rate * item_target
            order_qty  = max(needed_inv - inv_at_arrival, 0)
            if item_sl < 9999:
                # Usable shelf life once the stock actually arrives — it ages
                # in transit too, so LT eats into the SL window before the
                # item is even on the shelf.
                usable_sl = max(item_sl - item_lt, 0.0)
                order_qty = min(order_qty, local_rate * usable_sl)
            order_qty = round(order_qty, 0)
            if order_qty <= 0:
                # Already enough covered by then (e.g. a real PO lands here),
                # or shelf life can't absorb any order at this lead time —
                # nothing to add; move past it rather than re-finding the same spot.
                search_from = arrival_idx + 1
                continue

            sim_po[arrival_idx] += order_qty
            if is_exception:
                po_exc[place_idx] += order_qty
                needed_by[place_idx] = {
                    "trigger_month": month_cols[trigger_idx],
                    "arrival_month": month_cols[arrival_idx],
                    "lt_shortfall": lt_shortfall,
                    "notice_shortfall": notice_shortfall,
                }
            else:
                po_rec[place_idx] += order_qty
            search_from = arrival_idx + 1

        # "Cover Updated" reflects only PO Rec. quantities, not PO Exception —
        # sim_po (used above) includes both so trigger-finding across passes
        # still sees an exception order's real, if late, eventual arrival, but
        # the displayed projection should only move when a normal PO Rec. is
        # added, and by exactly that amount.
        po_base = np.nan_to_num(po_raw, nan=0.0) if po_raw is not None else np.zeros(n)
        _, cover_final, _, _ = compute_item_inv_cover(forecast_arr, po_base + po_rec, onhand_raw, month_cols)

        base    = {"item": item, "description": desc_map.get(item, "")}
        po_row  = {**base, "ORDER_TYPE_FINAL": "PO_RECOMMENDATION"}
        exc_row = {**base, "ORDER_TYPE_FINAL": "PO_EXCEPTION"}
        upd_row = {**base, "ORDER_TYPE_FINAL": "COVER_MONTHS_UPDATED"}

        has_exceptions = False
        for i, m in enumerate(month_cols):
            po_row[m]  = po_rec[i] if po_rec[i] > 0 else np.nan
            exc_row[m] = po_exc[i] if po_exc[i] > 0 else np.nan
            if po_exc[i] > 0:
                has_exceptions = True
                needed_by_map[(item, m)] = needed_by.get(
                    i, {"trigger_month": m, "arrival_month": m, "lt_shortfall": 0, "notice_shortfall": 0}
                )
            upd_row[m] = cover_final[i] if not np.isnan(cover_final[i]) else np.nan

        new_rows.append(po_row)
        if has_exceptions:
            new_rows.append(exc_row)
        new_rows.append(upd_row)

    if not new_rows:
        return pivot_df, needed_by_map

    new_df = pd.DataFrame(new_rows)
    return pd.concat([pivot_df, new_df], ignore_index=True), needed_by_map


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

# Streamlit auto-generates a "#sku-procurement-tracking" anchor link on the
# SKU Tracking page's own title (hover the title to see the 🔗 icon) — a very
# natural link to copy and share. But a URL fragment is never sent to the
# server (browsers don't include it in the request at all), and that header
# only exists in the DOM once app_mode is already "SKU Tracking" — so on a
# fresh load (default: Simulation) the anchor points at nothing and silently
# does nothing. Fixed client-side: if that hash is present, redirect once to
# the real deep link (?page=sku_tracking, handled below) which *is* visible
# to Python. window.location inside this iframe reports its own "about:srcdoc"
# address, not the real page's — window.parent.location is the actual address
# bar, reachable here because Streamlit's iframe sandbox includes
# allow-same-origin.
import streamlit.components.v1 as _comp_hash
_comp_hash.html(
    """<script>
(function(){
  try {
    var loc = window.parent.location;
    if (loc.hash === "#sku-procurement-tracking") {
      var params = new URLSearchParams(loc.search);
      if (params.get("page") !== "sku_tracking") {
        params.set("page", "sku_tracking");
        loc.replace(loc.pathname + "?" + params.toString() + loc.hash);
      }
    }
  } catch (e) {}
})();
</script>""",
    height=0,
)

# ══════════════════════════════════════════════════════════════
# STAGED FILTER HELPER — every filter/selection group in the app
# uses this: widgets update a "pending" value live, but nothing is
# actually applied (nothing re-filters/re-renders) until the user
# clicks Apply, which shows a spinner while it takes effect.
# ══════════════════════════════════════════════════════════════

def _norm(v):
    """Normalize a widget value for equality comparison (order-independent for lists)."""
    if isinstance(v, list):
        return tuple(sorted(v, key=str))
    return v


def _apply_bar(key_prefix, pending, defaults, label="Apply", help_changed="Selection changed — click Apply to update."):
    """
    Compares `pending` (tuple of current widget values) against the last-applied value
    for `key_prefix`. Renders an Apply button (disabled when nothing changed) with a
    spinner on click, and returns the currently-applied tuple for the caller to filter
    with. On first run, `defaults` seeds the applied value so the page isn't empty
    before the user ever touches the filters.
    """
    applied_key = f"_applied__{key_prefix}"
    if applied_key not in st.session_state:
        st.session_state[applied_key] = tuple(_norm(v) for v in defaults)

    pending_norm = tuple(_norm(v) for v in pending)
    changed = pending_norm != st.session_state[applied_key]

    bcol, ccol = st.columns([1, 5])
    with bcol:
        if st.button(f"✅ {label}", key=f"_apply_btn__{key_prefix}", disabled=not changed, use_container_width=True):
            with st.spinner("🔄 Applying…"):
                st.session_state[applied_key] = pending_norm
                st.rerun()
    with ccol:
        if changed:
            st.caption(f"⚠️ {help_changed}")

    return st.session_state[applied_key]

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

for key in ["full_bom", "pivot_df", "original_pivot", "sim_result", "data_loaded", "load_error", "changed_items", "current_parent", "sub_df", "sub_items_map", "app_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.app_mode is None:
    # Lets a bookmarked/shared URL like ".../?page=sku_tracking" open straight
    # into SKU Tracking instead of always landing on Simulation first. Only
    # consulted on a session's very first load — the sidebar's own mode
    # switch (below) still works normally afterward even if the URL still
    # shows the query param.
    _qp_page = str(st.query_params.get("page", "")).strip().lower()
    if _qp_page in ("sku_tracking", "sku", "tracking"):
        st.session_state.app_mode = "SKU Tracking"
    elif _qp_page in ("tpl", "globus"):
        st.session_state.app_mode = "Globus TPL"
    elif _qp_page in ("rm", "rm_analysis"):
        st.session_state.app_mode = "RM Analysis"
    else:
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
            pivot_df, _needed_by = compute_po_recommendations(pivot_df, _month_cols_init, master_map=_master_map)

            st.session_state.full_bom = full_bom
            st.session_state.pivot_df = pivot_df
            st.session_state.po_needed_by = _needed_by
            st.session_state.original_pivot = pivot_df.copy()
            st.session_state.current_parent = _default_parent
            st.session_state.data_loaded = True
            st.session_state.load_error = None

        except Exception as e:
            st.session_state.load_error = str(e)
            st.session_state.data_loaded = False

def sku_tracking_filter_bar(tracking_df: pd.DataFrame, key_prefix: str, uid: str = "") -> pd.DataFrame:
    """
    The standard SKU Procurement Tracking filter bar — search box + status +
    priority + request type + planner + procurement rep + supplier + owner +
    exact-SKU, with the app-wide staged Apply/spinner pattern (_apply_bar)
    plus a one-click Clear. Status/priority/request type include any custom
    values actually present in the data (not just the curated defaults), so
    a value typed as "new" in the table's row editor is always filterable
    here too — same pattern as the planner/rep/supplier/owner lists below,
    which are already fully dynamic.

    When `uid` is given, this browser's last-applied selection is loaded once
    per session to seed the widgets, and any newly-applied selection (or a
    Clear) is saved back — so a returning visit reopens with the same filters.
    """
    def _opts_union(fixed_opts, series):
        existing = sorted({v.strip() for v in series.astype(str) if v.strip() and v.strip() not in fixed_opts})
        return list(fixed_opts) + existing

    _loaded_key = f"_loaded_filters__{key_prefix}"
    if _loaded_key not in st.session_state:
        st.session_state[_loaded_key] = load_user_filters(uid) if uid else {}
    _loaded = st.session_state[_loaded_key]

    fr1, fr2, fr3, fr4 = st.columns(4)
    with fr1:
        search_pending = st.text_input(
            "🔎 Search SKU / description", value=_loaded.get("search", ""), key=f"{key_prefix}_search_pending",
            placeholder="Type any part of the SKU or description…",
        )
    with fr2:
        status_opts = _opts_union(SKU_STATUS_OPTIONS, tracking_df["status"])
        _d_status = [v for v in _loaded.get("status", []) if v in status_opts]
        status_pending = st.multiselect("Filter by status", status_opts, default=_d_status, key=f"{key_prefix}_status_pending")
    with fr3:
        priority_opts = _opts_union(SKU_PRIORITY_OPTIONS, tracking_df["priority"])
        _d_priority = [v for v in _loaded.get("priority", []) if v in priority_opts]
        priority_pending = st.multiselect("Filter by תעדוף", priority_opts, default=_d_priority, key=f"{key_prefix}_priority_pending")
    with fr4:
        reqtype_opts = _opts_union(SKU_REQUEST_TYPES, tracking_df["request_type"])
        _d_reqtype = [v for v in _loaded.get("request_type", []) if v in reqtype_opts]
        reqtype_pending = st.multiselect("Filter by סוג הבקשה", reqtype_opts, default=_d_reqtype, key=f"{key_prefix}_reqtype_pending")

    fr5, fr6, fr7, fr8 = st.columns(4)
    with fr5:
        planner_opts = sorted([p for p in tracking_df["planner_code"].unique().tolist() if p])
        _d_planner = [v for v in _loaded.get("planner", []) if v in planner_opts]
        planner_pending = st.multiselect("Filter by planner code", planner_opts, default=_d_planner, key=f"{key_prefix}_planner_pending")
    with fr6:
        rep_opts = sorted([r for r in tracking_df["procurement_rep"].unique().tolist() if r])
        _d_rep = [v for v in _loaded.get("rep", []) if v in rep_opts]
        rep_pending = st.multiselect("Filter by נציג רכש", rep_opts, default=_d_rep, key=f"{key_prefix}_rep_pending")
    with fr7:
        supplier_opts = sorted([s for s in tracking_df["supplier"].unique().tolist() if s])
        _d_supplier = [v for v in _loaded.get("supplier", []) if v in supplier_opts]
        supplier_pending = st.multiselect("Filter by supplier", supplier_opts, default=_d_supplier, key=f"{key_prefix}_supplier_pending")
    with fr8:
        owner_opts = sorted([o for o in tracking_df["owner"].unique().tolist() if o])
        _d_owner = [v for v in _loaded.get("owner", []) if v in owner_opts]
        owner_pending = st.multiselect("Filter by owner", owner_opts, default=_d_owner, key=f"{key_prefix}_owner_pending")

    fr9, _fr10 = st.columns([1, 3])
    with fr9:
        item_opts = sorted(tracking_df["item"].unique().tolist())
        _d_item = [v for v in _loaded.get("item", []) if v in item_opts]
        item_pending = st.multiselect("Filter by SKU (exact)", item_opts, default=_d_item, key=f"{key_prefix}_item_pending")

    (search, status_filter, priority_filter, reqtype_filter, planner_filter,
     rep_filter, supplier_filter, owner_filter, item_filter) = _apply_bar(
        f"{key_prefix}_filters",
        [search_pending, status_pending, priority_pending, reqtype_pending, planner_pending,
         rep_pending, supplier_pending, owner_pending, item_pending],
        [_loaded.get("search", ""), _d_status, _d_priority, _d_reqtype, _d_planner,
         _d_rep, _d_supplier, _d_owner, _d_item],
    )

    if uid:
        _persist_key = f"_persisted__{key_prefix}_filters"
        _current = {
            "search": search, "status": status_filter, "priority": priority_filter,
            "request_type": reqtype_filter, "planner": planner_filter, "rep": rep_filter,
            "supplier": supplier_filter, "owner": owner_filter, "item": item_filter,
        }
        if _persist_key not in st.session_state:
            st.session_state[_persist_key] = _current  # first render — matches what was just loaded, nothing to save yet
        elif st.session_state[_persist_key] != _current:
            st.session_state[_persist_key] = _current
            save_user_filters(uid, _current)

    if st.button("🔄 Clear filters", key=f"{key_prefix}_clear"):
        for k in (f"{key_prefix}_search_pending", f"{key_prefix}_status_pending", f"{key_prefix}_priority_pending",
                  f"{key_prefix}_reqtype_pending", f"{key_prefix}_planner_pending", f"{key_prefix}_rep_pending",
                  f"{key_prefix}_supplier_pending", f"{key_prefix}_owner_pending", f"{key_prefix}_item_pending"):
            st.session_state.pop(k, None)
        st.session_state.pop(f"_applied__{key_prefix}_filters", None)
        st.session_state.pop(_loaded_key, None)
        if uid:
            st.session_state.pop(f"_persisted__{key_prefix}_filters", None)
            save_user_filters(uid, {})
        st.rerun()

    view_df = tracking_df.copy()
    if search:
        s = search.strip().lower()
        # regex=False: the box is a literal substring search. Descriptions are
        # full of "(", ")", "+", "." etc. — as a regex those either silently
        # change what matches (dropping real rows) or raise on an unbalanced
        # paren and blank the whole page.
        view_df = view_df[
            view_df["item"].astype(str).str.lower().str.contains(s, na=False, regex=False) |
            view_df["description"].astype(str).str.lower().str.contains(s, na=False, regex=False)
        ]
    if status_filter:    view_df = view_df[view_df["status"].isin(status_filter)]
    if priority_filter:  view_df = view_df[view_df["priority"].isin(priority_filter)]
    if reqtype_filter:   view_df = view_df[view_df["request_type"].isin(reqtype_filter)]
    if planner_filter:   view_df = view_df[view_df["planner_code"].isin(planner_filter)]
    if rep_filter:        view_df = view_df[view_df["procurement_rep"].isin(rep_filter)]
    if supplier_filter:  view_df = view_df[view_df["supplier"].isin(supplier_filter)]
    if owner_filter:     view_df = view_df[view_df["owner"].isin(owner_filter)]
    if item_filter:      view_df = view_df[view_df["item"].isin(item_filter)]
    return view_df


def render_sku_tracking_table(rows: list, col_defs: list, item_suggest: dict,
                               item_options: list, status_opts: list, priority_opts: list,
                               reqtype_opts: list, planner_opts: list, rep_opts: list,
                               supplier_opts: list, owner_opts: list,
                               api_base: str, default_status: str) -> str:
    """
    One unified SKU tracking table: view, filter (rows are pre-filtered by the
    caller), create, update, delete — all in place. Hand-built HTML/JS calling
    the data API directly via fetch(), not st.data_editor, for two reasons:
    (1) the editable grid renders cells on a canvas and does not reliably
    honor manually-typed line breaks — a real <textarea> does; (2) Streamlit's
    components.v1.html iframe has no callback channel back into the Python
    process, so writes have to go straight to the API instead of through
    Streamlit.
    """
    payload = {
        "rows": rows, "cols": col_defs, "itemSuggest": item_suggest, "itemOptions": item_options,
        "statusOpts": status_opts, "priorityOpts": priority_opts, "reqtypeOpts": reqtype_opts,
        "plannerOpts": planner_opts, "repOpts": rep_opts, "supplierOpts": supplier_opts,
        "ownerOpts": owner_opts, "apiBase": api_base, "defaultStatus": default_status,
    }
    # Escaping "</" prevents a "</script>" inside free-typed notes from ending the
    # script block early; \u2028/\u2029 are valid JSON but illegal unescaped in a
    # JS string literal in older engines.
    data_json = (json.dumps(payload, ensure_ascii=False)
                 .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))

    template = r"""<!DOCTYPE html><html dir="rtl" lang="he"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden;direction:rtl}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;background:#fff;direction:rtl;text-align:right}
#outer{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;flex-direction:column}
#toolbar{display:flex;align-items:center;gap:10px;padding:8px 10px;border-bottom:1px solid #e2e8f0;flex:0 0 auto}
#wrap{position:relative;flex:1 1 auto;overflow:auto;-webkit-overflow-scrolling:touch}
#wrap::-webkit-scrollbar{height:10px;width:10px}
#wrap::-webkit-scrollbar-track{background:#f1f1f1;border-radius:4px}
#wrap::-webkit-scrollbar-thumb{background:#b0b8c8;border-radius:4px;border:2px solid #f1f1f1}
#wrap::-webkit-scrollbar-thumb:hover{background:#7a8499}
table{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}
th{position:sticky;top:0;background:#f1f5f9;color:#334155;font-size:11px;font-weight:600;
   text-transform:uppercase;letter-spacing:.4px;padding:8px 10px;text-align:right;
   border-bottom:2px solid #99a;white-space:nowrap;z-index:2}
td{padding:6px 8px;border-bottom:1px solid #eef1f5;text-align:right;vertical-align:top;direction:rtl;
   font-variant-numeric:tabular-nums;word-break:break-word}
tr:hover td{background:#f8fafc}
tr.sk-editing td{background:#eff6ff}
/* RTL: the frozen leading columns (item, description, comments) pin to the RIGHT edge, actions to the LEFT */
th:nth-child(1),td:nth-child(1){position:sticky;right:0;width:110px;min-width:110px;max-width:110px;background:#fff;z-index:1}
th:nth-child(2),td:nth-child(2){position:sticky;right:110px;width:190px;min-width:190px;max-width:190px;background:#fff;z-index:1}
th:nth-child(3),td:nth-child(3){position:sticky;right:300px;width:300px;min-width:300px;max-width:300px;background:#fff;z-index:1}
th:nth-child(1),th:nth-child(2),th:nth-child(3){background:#f1f5f9;z-index:3}
td:nth-child(3){box-shadow:inset 2px 0 0 #cbd5e1}
tr:hover td:nth-child(1),tr:hover td:nth-child(2),tr:hover td:nth-child(3){background:#f8fafc}
tr.sk-editing td:nth-child(1),tr.sk-editing td:nth-child(2),tr.sk-editing td:nth-child(3){background:#eff6ff}
th:nth-child(12),td:nth-child(12){width:340px;min-width:340px}
th:last-child,td:last-child{position:sticky;left:0;width:96px;min-width:96px;background:#fff;z-index:1;box-shadow:inset -2px 0 0 #cbd5e1}
th:last-child{background:#f1f5f9;z-index:3}
tr:hover td:last-child{background:#f8fafc}
tr.sk-editing td:last-child{background:#eff6ff}
.sk-ok{background:#dcfce7;color:#14532d;font-weight:600}
.sk-mid{background:#fef9c3;color:#713f12;font-weight:600}
.sk-bad{background:#fee2e2;color:#991b1b;font-weight:600}
.sk-note{white-space:normal;line-height:1.6;direction:rtl;text-align:right}
.sk-empty{color:#94a3b8;text-align:center;white-space:normal}
input,textarea,select{width:100%;font:inherit;padding:4px 6px;border:1px solid #cbd5e1;border-radius:4px;background:#fff;text-align:right;direction:rtl}
input[type=date]{text-align:right;direction:rtl}
input:disabled{background:#f1f5f9;color:#64748b}
textarea{resize:vertical;min-height:56px;line-height:1.4}
.sk-custom-input{margin-top:4px}
.sk-hint{font-size:10px;color:#b45309;margin-top:2px}
.sk-actions{white-space:nowrap;text-align:center}
.sk-btn{border:1px solid #cbd5e1;background:#fff;border-radius:4px;padding:3px 7px;cursor:pointer;font-size:12px;margin:0 2px}
.sk-btn:hover{background:#f1f5f9}
.sk-btn:disabled{opacity:.5;cursor:default}
.sk-btn-primary{border-color:#2563eb;background:#2563eb;color:#fff}
.sk-btn-primary:hover{background:#1d4ed8}
.sk-btn-danger:hover{background:#fee2e2;border-color:#fca5a5}
#sk-error{display:none;color:#991b1b;background:#fee2e2;border:1px solid #fca5a5;border-radius:4px;padding:5px 10px;font-size:12px;flex:1 1 auto}
</style></head><body>
<div id="outer">
  <div id="toolbar">
    <button id="addBtn" class="sk-btn sk-btn-primary" onclick="startAdd()">➕ הוסף רשומה חדשה</button>
    <div id="sk-error"></div>
  </div>
  <div id="wrap">
    <table>
      <thead><tr id="thead-row"></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>
<datalist id="dl-item"></datalist>
<script>
const DATA = __DATA_JSON__;
let ROWS = DATA.rows;
const COLS = DATA.cols;
const ITEM_SUGGEST = DATA.itemSuggest;
const API_BASE = DATA.apiBase;
const DEFAULT_STATUS = DATA.defaultStatus;

// Fields with a known-value list get a real <select> (always shows every
// option, unlike <input list=datalist> which — as reported — filters its
// suggestions down to whatever text is already typed in the box, hiding
// everything else once a value is pre-filled) plus an explicit "add new
// value" choice that reveals a free-text box.
const OPTIONS_FOR = {
  supplier: DATA.supplierOpts, procurement_rep: DATA.repOpts, planner_code: DATA.plannerOpts,
  request_type: DATA.reqtypeOpts, priority: DATA.priorityOpts, status: DATA.statusOpts, owner: DATA.ownerOpts,
};
const CUSTOM_VALUE = "__ADD_NEW__";
const NOTE_COLS = new Set(["update_notes", "comments_procurement"]);
const DATE_COLS = new Set(["request_date", "next_action_date"]);

let editingId = null;
let busy = false;
let errorMsg = "";

function esc(s){
  return String(s == null ? "" : s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
function isoDate(v){
  return /^\d{4}-\d{2}-\d{2}$/.test(String(v||"")) ? v : "";
}
// Read-mode display: ISO (2025-11-19) -> Israeli / Excel style DD.MM.YYYY.
function fmtDate(v){
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(v == null ? "" : v));
  return m ? (m[3] + "." + m[2] + "." + m[1]) : String(v == null ? "" : v);
}
function todayISO(){
  const d = new Date();
  return d.getFullYear() + "-" + String(d.getMonth()+1).padStart(2,"0") + "-" + String(d.getDate()).padStart(2,"0");
}
function statusClass(v){
  const s = String(v||"");
  if (s.indexOf("הושלם") !== -1) return "sk-ok";
  if (s.indexOf("בטיפול") !== -1) return "sk-mid";
  if (s.indexOf("טרם טופל") !== -1) return "sk-bad";
  return "";
}
function priorityClass(v){
  const s = String(v||"");
  if (s.indexOf("דחוף") !== -1) return "sk-bad";
  if (s.indexOf("בינוני") !== -1) return "sk-mid";
  return "";
}

function buildDatalist(id, options){
  document.getElementById(id).innerHTML = options.map(function(o){ return "<option value=\"" + esc(o) + "\"></option>"; }).join("");
}
buildDatalist("dl-item", DATA.itemOptions);

document.getElementById("thead-row").innerHTML =
  COLS.map(function(c){ return "<th>" + esc(c.label) + "</th>"; }).join("") + "<th>פעולות</th>";

function readCell(col, row){
  const text = String(row[col.key] == null ? "" : row[col.key]);
  if (NOTE_COLS.has(col.key)) {
    // Entries are stored newline-separated — render each on its own line with an
    // explicit <br> so every update starts at the beginning of a new line.
    return text.trim()
      ? ('<td class="sk-note" dir="rtl">' + esc(text).replace(/\r\n?|\n/g, "<br>") + '</td>')
      : '<td class="sk-note sk-empty">–</td>';
  }
  if (DATE_COLS.has(col.key)) {
    return '<td dir="rtl">' + (esc(fmtDate(text)) || "–") + '</td>';
  }
  if (col.key === "status") return '<td class="' + statusClass(text) + '">' + (esc(text) || "–") + '</td>';
  if (col.key === "priority") return '<td class="' + priorityClass(text) + '">' + (esc(text) || "–") + '</td>';
  return '<td>' + (esc(text) || "–") + '</td>';
}

function editCell(col, row, rowId){
  const raw = row[col.key] == null ? "" : row[col.key];
  const val = esc(String(raw));
  const id = "f-" + col.key + "-" + rowId;
  if (col.key === "item") {
    if (rowId === "__new__") {
      return '<td><input id="' + id + '" list="dl-item" value="' + val + '" placeholder="מק&quot;ט (SKU)…" oninput="onItemInput(event)">'
           + '<div id="sk-item-hint" class="sk-hint"></div></td>';
    }
    return '<td><input id="' + id + '" value="' + val + '" disabled></td>';
  }
  if (NOTE_COLS.has(col.key)) {
    return '<td><textarea id="' + id + '" dir="rtl">' + val + '</textarea></td>';
  }
  if (DATE_COLS.has(col.key)) {
    return '<td><input type="date" id="' + id + '" value="' + isoDate(raw) + '" dir="rtl"></td>';
  }
  const opts = OPTIONS_FOR[col.key];
  if (opts) {
    const isKnown = raw === "" || opts.indexOf(raw) !== -1;
    let optionsHtml = '<option value="">– ריק –</option>';
    const seen = new Set();
    for (const o of opts) {
      if (!o || seen.has(o)) continue;
      seen.add(o);
      optionsHtml += '<option value="' + esc(o) + '"' + (o === raw ? " selected" : "") + '>' + esc(o) + '</option>';
    }
    if (!isKnown) {
      // A legacy/custom value not in the curated list — show it as its own
      // selected option instead of silently losing it.
      optionsHtml += '<option value="' + val + '" selected>' + val + '</option>';
    }
    optionsHtml += '<option value="' + CUSTOM_VALUE + '">➕ הוסף ערך חדש…</option>';
    return '<td>'
      + '<select id="' + id + '" onchange="onComboChange(\'' + id + '\')">' + optionsHtml + '</select>'
      + '<input id="' + id + '__custom" class="sk-custom-input" style="display:none" placeholder="הקלידו ערך חדש…">'
      + '</td>';
  }
  return '<td><input id="' + id + '" value="' + val + '"></td>';
}

function onComboChange(id){
  const sel = document.getElementById(id);
  const custom = document.getElementById(id + "__custom");
  if (!sel || !custom) return;
  if (sel.value === CUSTOM_VALUE) {
    custom.style.display = "block";
    custom.value = "";
    custom.focus();
  } else {
    custom.style.display = "none";
    custom.value = "";
  }
}

function setComboValue(id, value){
  const sel = document.getElementById(id);
  if (!sel || !value) return;
  let matched = false;
  for (const opt of sel.options) {
    if (opt.value === value) { matched = true; break; }
  }
  if (!matched) {
    const o = document.createElement("option");
    o.value = value; o.textContent = value;
    sel.insertBefore(o, sel.lastElementChild); // keep "➕ הוסף ערך חדש…" last
  }
  sel.value = value;
  onComboChange(id);
}

function onItemInput(e){
  const v = e.target.value.trim();
  const sug = ITEM_SUGGEST[v];
  const descEl = document.getElementById("f-description-__new__");
  const plannerSel = document.getElementById("f-planner_code-__new__");
  const supplierSel = document.getElementById("f-supplier-__new__");
  const hintEl = document.getElementById("sk-item-hint");
  if (sug) {
    if (descEl && !descEl.value && sug.description) descEl.value = sug.description;
    if (plannerSel && !plannerSel.value && sug.planner_code) setComboValue("f-planner_code-__new__", sug.planner_code);
    if (supplierSel && !supplierSel.value && sug.supplier) setComboValue("f-supplier-__new__", sug.supplier);
    if (hintEl) hintEl.textContent = "";
  } else if (v && DATA.itemOptions.indexOf(v) === -1) {
    if (hintEl) hintEl.textContent = 'מק"ט חדש — נא להשלים תיאור/ספק ידנית.';
  } else if (hintEl) {
    hintEl.textContent = "";
  }
}

function blankRow(){
  return {id:"__new__", item:"", description:"", supplier:"", procurement_rep:"",
    planner_code:"", request_type:"", request_date: todayISO(), next_action_date:"",
    priority:"", status: DEFAULT_STATUS, update_notes:"", comments_procurement:"", owner:""};
}

function actionsCell(rowId, mode){
  if (mode === "edit") {
    return '<td class="sk-actions">'
      + '<button class="sk-btn sk-btn-primary" onclick="doSave(\'' + rowId + '\')" ' + (busy?"disabled":"") + '>💾</button>'
      + '<button class="sk-btn" onclick="cancelEdit()" ' + (busy?"disabled":"") + '>✖️</button></td>';
  }
  return '<td class="sk-actions">'
    + '<button class="sk-btn" onclick="startEdit(\'' + rowId + '\')" ' + (busy?"disabled":"") + '>✏️</button>'
    + '<button class="sk-btn sk-btn-danger" onclick="doDelete(\'' + rowId + '\')" ' + (busy?"disabled":"") + '>🗑️</button></td>';
}

function renderTable(){
  let html = "";
  if (editingId === "__new__") {
    const row = blankRow();
    html += '<tr class="sk-editing sk-new">' + COLS.map(function(c){ return editCell(c, row, "__new__"); }).join("") + actionsCell("__new__","edit") + '</tr>';
  }
  for (const row of ROWS) {
    if (editingId === row.id) {
      html += '<tr class="sk-editing">' + COLS.map(function(c){ return editCell(c, row, row.id); }).join("") + actionsCell(row.id, "edit") + '</tr>';
    } else {
      html += '<tr>' + COLS.map(function(c){ return readCell(c, row); }).join("") + actionsCell(row.id, "read") + '</tr>';
    }
  }
  document.getElementById("tbody").innerHTML = html;
  document.getElementById("addBtn").disabled = (editingId !== null) || busy;
  const errEl = document.getElementById("sk-error");
  errEl.textContent = errorMsg;
  errEl.style.display = errorMsg ? "block" : "none";
}

function startAdd(){
  if (busy || editingId !== null) return;
  editingId = "__new__"; errorMsg = ""; renderTable();
  const el = document.getElementById("f-item-__new__");
  if (el) el.focus();
}
function startEdit(id){
  if (busy || editingId !== null) return;
  editingId = id; errorMsg = ""; renderTable();
}
function cancelEdit(){
  if (busy) return;
  editingId = null; errorMsg = ""; renderTable();
}

function collectFields(rowId){
  const out = {};
  for (const c of COLS) {
    if (c.key === "item" && rowId !== "__new__") continue;
    const el = document.getElementById("f-" + c.key + "-" + rowId);
    if (!el) continue;
    let v = el.value;
    if (OPTIONS_FOR[c.key] && v === CUSTOM_VALUE) {
      const customEl = document.getElementById("f-" + c.key + "-" + rowId + "__custom");
      v = customEl ? customEl.value : "";
    }
    out[c.key] = NOTE_COLS.has(c.key) ? v : v.trim();
  }
  return out;
}

async function doSave(rowId){
  if (busy) return;
  const fields = collectFields(rowId);
  if (rowId === "__new__" && !String(fields.item||"").trim()) {
    errorMsg = 'נא להזין מק"ט (SKU) לפני השמירה.';
    renderTable();
    return;
  }
  busy = true; errorMsg = ""; renderTable();
  try {
    if (rowId === "__new__") {
      const resp = await fetch(API_BASE + "/sku-tracking", {
        method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(fields),
      });
      if (!resp.ok) throw new Error(String(resp.status));
      const created = await resp.json();
      ROWS = [created].concat(ROWS);
    } else {
      const resp = await fetch(API_BASE + "/sku-tracking/" + encodeURIComponent(rowId), {
        method: "PATCH", headers: {"Content-Type":"application/json"}, body: JSON.stringify(fields),
      });
      if (!resp.ok) throw new Error(String(resp.status));
      const idx = ROWS.findIndex(function(r){ return r.id === rowId; });
      if (idx !== -1) ROWS[idx] = Object.assign({}, ROWS[idx], fields);
    }
    editingId = null;
  } catch (err) {
    errorMsg = "❌ השמירה נכשלה (" + (err.message || err) + "). ודאו שהשרת זמין ונסו שוב.";
  } finally {
    busy = false; renderTable();
  }
}

async function doDelete(rowId){
  if (busy) return;
  const row = ROWS.find(function(r){ return r.id === rowId; });
  const label = row ? row.item : rowId;
  if (!confirm("למחוק את הרשומה של " + label + "? הפעולה בלתי הפיכה.")) return;
  busy = true; errorMsg = ""; renderTable();
  try {
    const resp = await fetch(API_BASE + "/sku-tracking/" + encodeURIComponent(rowId), { method: "DELETE" });
    if (!resp.ok) throw new Error(String(resp.status));
    ROWS = ROWS.filter(function(r){ return r.id !== rowId; });
  } catch (err) {
    errorMsg = "❌ המחיקה נכשלה (" + (err.message || err) + "). ודאו שהשרת זמין ונסו שוב.";
  } finally {
    busy = false; renderTable();
  }
}

renderTable();
</script>
</body></html>"""
    return template.replace("__DATA_JSON__", data_json)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

if st.session_state.app_mode in ("SKU Tracking", "Globus TPL", "RM Analysis"):
    if st.button("⬅️ Back to MRP Simulation"):
        with st.spinner("🔄 Loading MRP Simulation…"):
            st.session_state.app_mode = "Simulation"
            st.session_state.app_mode_pending = "Simulation"
            st.rerun()
    if st.session_state.app_mode == "SKU Tracking":
        st.title("🔎 SKU Procurement Tracking")
        st.caption("Track open procurement issues per SKU — status, priority, next follow-up date, supplier and running notes.")
    elif st.session_state.app_mode == "Globus TPL":
        st.title("🏬 Globus TPL – Storage Savings")
        st.caption("פוטנציאל חיסכון באחסון במחסן גלובוס — איחוד משטחים לפי אצוות ספק, Safety Stock, מלאי עודף ומלאי מת.")
    else:
        st.title("🧪 RM Analysis – Raw Materials by Parent Product")
        st.caption("חומרי גלם ייחודיים / מרכזיים למק״טי אב, כמויות ומחירים לדרישה, ומצב מלאי מול הייצור הקיים.")
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
    with st.spinner("🔄 Loading SKU Tracking…"):
        tracking_df = load_sku_tracking()

    # Known SKUs from master data / pivot — powers the item dropdown and the
    # description/planner auto-fill on save (nothing here calls the DB again).
    _mm_track = getattr(st.session_state, "master_map", {}) or {}
    _pivot_track = getattr(st.session_state, "pivot_df", None)
    if _mm_track:
        track_item_options = sorted(_mm_track.keys())
    elif _pivot_track is not None:
        track_item_options = sorted(_pivot_track["item"].dropna().unique().tolist())
    else:
        track_item_options = []
    if not track_item_options:
        st.info("💡 Simulation data hasn't been loaded yet, so the **item** column will use free-text entry "
                 "and description/planner won't auto-fill. Switch to **Simulation** mode once to enable this.")

    # "comments_procurement" sits 3rd (right after description) and is frozen —
    # the template's sticky CSS pins the first three columns.
    display_cols = ["item","description","comments_procurement","supplier","procurement_rep",
                     "planner_code","request_type","request_date","next_action_date","priority",
                     "status","update_notes","owner"]

    # Real values already used historically (imported/entered) are unioned into the
    # curated option lists, so any legitimate past value stays selectable even if it
    # isn't one of the small set of "blessed" defaults in src/sku_tracking.py. Shared
    # by the quick-add form below and the grid further down.
    def _opts_with_existing(fixed_opts, series):
        existing = sorted({v.strip() for v in series.astype(str) if v.strip() and v.strip() not in fixed_opts})
        return list(fixed_opts) + existing

    _status_opts   = _opts_with_existing(SKU_STATUS_OPTIONS, tracking_df["status"])
    _priority_opts = _opts_with_existing(SKU_PRIORITY_OPTIONS, tracking_df["priority"])
    _reqtype_opts  = _opts_with_existing(SKU_REQUEST_TYPES, tracking_df["request_type"])

    # Planner / procurement rep / supplier have no fixed enum (any free text is
    # legitimate per the business) — these are the searchable "pick existing or type
    # a new one" lists for both the quick-add form and the grid's help hints.
    _planner_existing  = sorted({p.strip() for p in tracking_df["planner_code"].astype(str) if p.strip()})
    _rep_existing      = sorted({r.strip() for r in tracking_df["procurement_rep"].astype(str) if r.strip()})
    _supplier_existing = sorted({s.strip() for s in tracking_df["supplier"].astype(str) if s.strip()})
    _owner_existing    = sorted({o.strip() for o in tracking_df["owner"].astype(str) if o.strip()})

    n_open     = int((tracking_df["status"] == "🔴 טרם טופל").sum())
    n_progress = int((tracking_df["status"] == "🟡 בטיפול").sum())
    n_done     = int((tracking_df["status"] == "🟢 הושלם").sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("🔴 טרם טופל", n_open)
    k2.metric("🟡 בטיפול", n_progress)
    k3.metric("🟢 הושלם", n_done)

    # A per-browser identity, purely so the filter bar below can remember this
    # person's last selection — no login, no name, just an anonymous cookie.
    # st.context.cookies is read-only, so a value is picked in Python first
    # (reused from the cookie if one already exists) and a tiny invisible
    # component writes it back to the browser; components.v1.html's iframe
    # sandbox includes allow-same-origin, so document.cookie set inside it
    # lands on the real page's origin, not an isolated one.
    try:
        _uid_cookie = st.context.cookies.get("mrp_uid")
    except Exception:
        _uid_cookie = None
    if "sku_uid" not in st.session_state:
        st.session_state.sku_uid = _uid_cookie or uuid.uuid4().hex
    sku_uid = st.session_state.sku_uid
    import streamlit.components.v1 as _comp_uid
    _comp_uid.html(
        '<script>if (!document.cookie.includes("mrp_uid=")) '
        'document.cookie = "mrp_uid=' + sku_uid + '; path=/; max-age=31536000; samesite=lax";</script>',
        height=0,
    )

    edit_view_df = sku_tracking_filter_bar(tracking_df, "tt", sku_uid)

    is_filtered = len(edit_view_df) < len(tracking_df)
    if is_filtered:
        st.caption(f"Showing {len(edit_view_df)} of {len(tracking_df)} rows (filtered).")

    # ── One unified table: view + filter + create + update + delete ───
    # The table's own JS talks to the data API directly (via the IIS reverse
    # proxy on :8503 — the API itself stays on 127.0.0.1 for security), since
    # components.v1.html's iframe has no callback channel back into Streamlit.
    col_labels = ["מק\"ט","תיאור","הערות","ספק","נציג רכש","פלנר","סוג הבקשה","תאריך פנייה",
                  "מועד הבא לטיפול","תעדוף","סטטוס רכש","עדכון סטטוס","אחראי"]
    col_defs = [{"key": k, "label": lbl} for k, lbl in zip(display_cols, col_labels)]

    # Rows handed to the table — dates normalized to ISO so <input type="date">
    # reliably shows them (older CSV-migrated rows aren't guaranteed ISO already).
    _df_for_js = edit_view_df.copy()
    for _dcol in ("request_date", "next_action_date"):
        _df_for_js[_dcol] = pd.to_datetime(_df_for_js[_dcol], errors="coerce").dt.strftime("%Y-%m-%d")
    rows_for_js = _df_for_js.fillna("").astype(str).to_dict(orient="records")

    # Same auto-fill logic the old quick-add form used (description from pivot,
    # planner from master data, supplier from the most recent tracking row for
    # that item) — precomputed once here instead of round-tripping to the API,
    # since it's already resident in the Streamlit session at render time.
    _desc_by_item = {}
    if _pivot_track is not None and "item" in _pivot_track.columns and "description" in _pivot_track.columns:
        _dd = _pivot_track[["item", "description"]].dropna(subset=["description"])
        _dd = _dd[_dd["description"].astype(str).str.strip() != ""]
        _desc_by_item = _dd.drop_duplicates(subset="item", keep="first").set_index("item")["description"].astype(str).to_dict()

    _planner_by_item = {k: str((v or {}).get("planner_code") or "") for k, v in _mm_track.items()} if _mm_track else {}

    _supplier_by_item = {}
    _hist_all = tracking_df[tracking_df["supplier"].astype(str).str.strip() != ""]
    if not _hist_all.empty:
        for _it, _grp in _hist_all.sort_values("last_updated").groupby("item"):
            _supplier_by_item[_it] = str(_grp["supplier"].iloc[-1])

    item_suggest = {}
    for _it in set(list(_desc_by_item) + [k for k, v in _planner_by_item.items() if v] + list(_supplier_by_item)):
        _s = {}
        if _it in _desc_by_item: _s["description"] = _desc_by_item[_it]
        if _planner_by_item.get(_it): _s["planner_code"] = _planner_by_item[_it]
        if _it in _supplier_by_item: _s["supplier"] = _supplier_by_item[_it]
        if _s: item_suggest[_it] = _s

    # The API proxy lives on the same host the browser is already talking to
    # Streamlit on, just a different port — read that host from the request
    # itself rather than hardcoding a server IP/hostname.
    try:
        _host_hdr = st.context.headers.get("Host", "") or ""
    except Exception:
        _host_hdr = ""
    _api_host = _host_hdr.split(":")[0].strip() if _host_hdr else "127.0.0.1"
    api_base = f"http://{_api_host}:8503"
    default_status = "🔴 טרם טופל" if "🔴 טרם טופל" in _status_opts else (_status_opts[0] if _status_opts else "")

    table_h = min(600, max(220, len(edit_view_df) * 60 + 140))
    import streamlit.components.v1 as _comp_sk
    _comp_sk.html(
        render_sku_tracking_table(
            rows_for_js, col_defs, item_suggest, track_item_options,
            _status_opts, _priority_opts, _reqtype_opts,
            _planner_existing, _rep_existing, _supplier_existing, _owner_existing,
            api_base, default_status,
        ),
        height=table_h, scrolling=True,
    )
    st.caption("✏️ עריכה, 🗑️ מחיקה ו-➕ הוספת רשומה חדשה נשמרות ישירות מכאן — עדכון / הערות- רכש שומרים "
               "ירידת שורה אמיתית (Enter / Alt+Enter). מספרי הסיכום למעלה והפילטרים מתעדכנים רק בלחיצה על רענון.")
    if st.button("🔄 רענון (סנכרון KPI ופילטרים אחרי עריכה)"):
        st.rerun()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        tracking_df.to_excel(w, index=False)
    st.download_button("📥 Download tracking table", data=buf.getvalue(), file_name="sku_tracking.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.stop()

if st.session_state.app_mode == "Globus TPL":
    from src.tpl_ui import render_tpl
    render_tpl()
    st.stop()

if st.session_state.app_mode == "RM Analysis":
    from src.rm_analysis_ui import render_rm_analysis
    render_rm_analysis()
    st.stop()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Application Settings")
    _mode_options = ["Simulation", "SKU Tracking", "Globus TPL", "RM Analysis"]
    pending_mode = st.selectbox(
        "App Mode",
        _mode_options,
        index=_mode_options.index(st.session_state.app_mode or "Simulation"),
        key="app_mode_pending",
    )
    mode_changed = pending_mode != (st.session_state.app_mode or "Simulation")
    if st.button("✅ Apply", key="apply_app_mode", disabled=not mode_changed, use_container_width=True):
        with st.spinner(f"🔄 Loading {pending_mode}…"):
            st.session_state.app_mode = pending_mode
            st.rerun()
    if mode_changed:
        st.caption(f"⚠️ Click **Apply** to switch to **{pending_mode}**.")
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
            result["pivot"], _needed_by_sim = compute_po_recommendations(result["pivot"], _sim_month_cols, master_map=_mm_sim2)
            result["month"] = cfg.month
            st.session_state.sim_result = result
            st.session_state.po_needed_by = _needed_by_sim
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
def _build_cover_tooltips(pivot_df: pd.DataFrame, month_cols: tuple, po_needed_by: frozenset = None) -> dict:
    """
    Precompute a hover-only breakdown for each (item, month, row_type) cell of
    COVER_MONTHS, COVER_MONTHS_UPDATED and PO_EXCEPTION. Keys are
    (item, month, ORDER_TYPE_FINAL) since these rows are computed differently
    and must not share one tooltip.

    COVER_MONTHS reuses compute_item_inv_cover() — the exact same function
    _compute_inv_cover() calls — so this can never drift from what's displayed.

    po_needed_by: frozenset of ((item, place_month), trigger_month) pairs from
    compute_po_recommendations() — the month a PO_EXCEPTION cell was truly
    needed, vs. the (later) month it could actually be placed/arrives.
    """
    month_cols = list(month_cols)
    if pivot_df is None or pivot_df.empty or not month_cols:
        return {}
    _needed_by = {k: dict(v) for k, v in po_needed_by} if po_needed_by else {}
    order_type = pivot_df["ORDER_TYPE_FINAL"].astype(str)
    is_fcst   = order_type.str.contains("forecast", case=False, na=False)
    is_po     = order_type.isin(["Purchase order"])
    is_onhand = order_type.str.contains("on hand|onhand", case=False, na=False) | (order_type == "3.On Hand")
    is_cov    = order_type == "COVER_MONTHS"
    is_covu   = order_type == "COVER_MONTHS_UPDATED"
    is_po_rec = order_type == "PO_RECOMMENDATION"
    is_po_exc = order_type == "PO_EXCEPTION"

    fcst_by_item   = _sim_sum_by_item(pivot_df, is_fcst,   month_cols)
    po_by_item     = _sim_sum_by_item(pivot_df, is_po,     month_cols)
    onhand_by_item = _sim_sum_by_item(pivot_df, is_onhand, month_cols)
    cov_by_item    = _sim_sum_by_item(pivot_df, is_cov,    month_cols)
    covu_by_item   = _sim_sum_by_item(pivot_df, is_covu,   month_cols)
    po_rec_by_item = _sim_sum_by_item(pivot_df, is_po_rec, month_cols)
    po_exc_by_item = _sim_sum_by_item(pivot_df, is_po_exc, month_cols)

    all_items = set(fcst_by_item) | set(po_by_item) | set(onhand_by_item)
    tooltips = {}

    for item in all_items:
        inv_arr, cover_arr, first_oh_idx, detail = compute_item_inv_cover(
            fcst_by_item.get(item), po_by_item.get(item), onhand_by_item.get(item), month_cols,
        )
        if first_oh_idx is None:
            continue

        for i in range(first_oh_idx, len(month_cols)):
            d = detail[i]
            if d is None:
                continue
            m = month_cols[i]
            if d["reason"] == "no_forward_demand" or pd.isna(d["cover"]):
                tooltips[(item, m, "COVER_MONTHS")] = (
                    f"COVER = N/A\n"
                    f"INV({m}) = {d['inv']:,.2f} (ending inventory, after this month's demand)\n"
                    f"No forecast demand anywhere ahead of {m} to run out against, so coverage "
                    f"is undefined here — not shown as a fabricated number."
                )
            elif d["reason"] == "beyond_horizon":
                nonzero_steps = [s for s in d["steps"] if s[1] > 0][:10]
                steps_txt = "\n".join(f"  {mo}: demand {dm:,.2f} → remaining {rem:,.2f}" for mo, dm, rem in nonzero_steps)
                tooltips[(item, m, "COVER_MONTHS")] = (
                    f"COVER ≥ {d['cover']:,.0f} mo (at least)\n"
                    f"INV({m}) = {d['inv']:,.2f} — walked forward through every remaining month "
                    f"in the available forecast data ({int(d['cover'])} months) and demand never "
                    f"used it all up. True coverage is longer than the visible forecast horizon, "
                    f"so this is a lower bound, not an exact figure:\n{steps_txt}"
                )
            else:
                nonzero_steps = [s for s in d["steps"] if s[1] > 0][:10]
                steps_txt = "\n".join(f"  {mo}: demand {dm:,.2f} → remaining {rem:,.2f}" for mo, dm, rem in nonzero_steps)
                more = (f"\n  … +{len(d['steps']) - len(nonzero_steps)} more month(s) walked"
                        if len(d["steps"]) > len(nonzero_steps) else "")
                tooltips[(item, m, "COVER_MONTHS")] = (
                    f"COVER = {d['cover']:,.2f} mo\n"
                    f"INV({m}) = {d['inv']:,.2f} — walked forward from the month AFTER {m} "
                    f"(this month's demand is already reflected in INV, so it isn't counted "
                    f"again here), accumulating demand until it ran out:\n"
                    f"{steps_txt}{more}"
                )

        # ── COVER_MONTHS_UPDATED — only differs from COVER_MONTHS when a PO
        # recommendation or exception was triggered at or before this month.
        covu_arr = covu_by_item.get(item)
        if covu_arr is None:
            continue
        po_rec_arr = po_rec_by_item.get(item)
        po_exc_arr = po_exc_by_item.get(item)
        triggers = []
        for i, m in enumerate(month_cols):
            if po_rec_arr is not None and pd.notna(po_rec_arr[i]) and po_rec_arr[i] > 0:
                triggers.append((i, po_rec_arr[i], False))
            if po_exc_arr is not None and pd.notna(po_exc_arr[i]) and po_exc_arr[i] > 0:
                triggers.append((i, po_exc_arr[i], True))
        triggers.sort(key=lambda t: t[0])

        for i, m in enumerate(month_cols):
            covu_val = covu_arr[i]
            if pd.isna(covu_val):
                continue
            cov_val = cover_arr[i] if i < len(cover_arr) else np.nan
            if not pd.isna(cov_val) and np.isclose(covu_val, cov_val, atol=0.01):
                tooltips[(item, m, "COVER_MONTHS_UPDATED")] = (
                    f"Cover Updated = Cover (mo) = {covu_val:,.2f} mo\n"
                    f"No PO recommendation or exception affects this month, "
                    f"so it's identical to the base coverage."
                )
            else:
                prior = [t for t in triggers if t[0] <= i]
                cov_txt = f"{cov_val:,.2f}" if not pd.isna(cov_val) else "N/A"
                if prior:
                    t_idx, t_qty, t_exc = prior[-1]
                    kind = "PO Exception (can't meet lead time)" if t_exc else "PO Recommendation"
                    tooltips[(item, m, "COVER_MONTHS_UPDATED")] = (
                        f"Cover Updated = {covu_val:,.2f} mo  (Cover (mo) alone = {cov_txt})\n"
                        f"Adjusted because a {kind} of {t_qty:,.0f} units was triggered at "
                        f"{month_cols[t_idx]} — this projects inventory forward with that "
                        f"order added, decaying at the item's demand rate."
                    )
                else:
                    tooltips[(item, m, "COVER_MONTHS_UPDATED")] = (
                        f"Cover Updated = {covu_val:,.2f} mo  (Cover (mo) alone = {cov_txt})"
                    )

    # PO_EXCEPTION cells: the order couldn't be placed with full lead time —
    # show the month it was truly needed vs. the (later) month it actually
    # arrives, and/or that it couldn't get the supplier's full standard
    # notice counted from today. trigger/arrival months come straight from
    # compute_po_recommendations(), not re-derived here, since place_month
    # and trigger_month are often the same month in a lateness exception
    # (both clamped to "now") — comparing those two would misreport the
    # lateness as zero when the real gap is trigger→arrival.
    for (_item, _m), _info in _needed_by.items():
        _trigger_month = _info["trigger_month"]
        _arrival_month = _info["arrival_month"]
        _lt_shortfall = _info.get("lt_shortfall", 0)
        _notice_shortfall = _info.get("notice_shortfall", 0)
        parts = []
        if _lt_shortfall > 0:
            parts.append(
                f"Coverage runs out {_trigger_month} — lead time doesn't allow placing this "
                f"order in time, so the earliest it can arrive is {_arrival_month}, "
                f"{_lt_shortfall} month(s) late."
            )
        if _notice_shortfall > 0:
            parts.append(
                f"This order has to be placed {_notice_shortfall} month(s) sooner than the "
                f"supplier's standard lead-time notice (counted from today) would normally "
                f"allow — even though it still arrives on time, it's short notice."
            )
        tooltips[(_item, _m, "PO_EXCEPTION")] = "\n".join(parts)
    return tooltips


@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def render_pivot_html(df, month_cols, changed_items=None, master_map_key=None, lt_markers=None, sub_items=None, cover_tooltips=None):
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
    # cover_tooltips: {(item, month): "hover breakdown text"} for COVER_MONTHS cells
    if isinstance(cover_tooltips, frozenset): cover_tooltips = dict(cover_tooltips)
    cover_tooltips = cover_tooltips or {}
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
            is_cover_type = otype in ("COVER_MONTHS", "COVER_MONTHS_UPDATED")
            for c in cols:
                val = row.get(c)
                try:
                    fval = float(val)
                    if pd.isna(fval):
                        cells.append('<td class="v">–</td>')
                        continue
                    content = f"{fval:,.1f}"
                    if is_cover_type:
                        tip = cover_tooltips.get((item, c, otype), "")
                        if tip:
                            content = f'<span class="tt" data-tip="{tip}">{content}</span>'
                    if otype == "COVER_MONTHS" and fval < float(_ss_map.get(item, 7)):
                        cells.append(f'<td class="v alert-cov">{content}</td>')
                    elif otype == "COVER_MONTHS_UPDATED" and fval < float(_ss_map.get(item, 7)):
                        cells.append(f'<td class="v alert-covu">{content}</td>')
                    elif otype == "PO_RECOMMENDATION" and fval > 0:
                        cells.append(f'<td class="v alert-po">{fval:,.0f}</td>')
                    elif otype == "PO_EXCEPTION" and fval > 0:
                        exc_tip = cover_tooltips.get((item, c, "PO_EXCEPTION"), "")
                        exc_content = f'<span class="tt" data-tip="{exc_tip}">⚠️ {fval:,.0f}</span>' if exc_tip else f'⚠️ {fval:,.0f}'
                        cells.append(f'<td class="v alert-exc">{exc_content}</td>')
                    else:
                        cells.append(f'<td class="v">{content}</td>')
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
/* Custom hover tooltip for Cover (mo) cells — instant, styled, no native-title delay */
.tt{{position:relative;cursor:help;border-bottom:1px dotted #94a3b8}}
.tt:hover::after{{
  content:attr(data-tip);position:absolute;bottom:125%;left:50%;transform:translateX(-50%);
  background:#1e293b;color:#fff;padding:8px 10px;border-radius:6px;font-size:11px;line-height:1.5;
  white-space:pre-line;width:max-content;max-width:280px;text-align:left;z-index:1000;
  box-shadow:0 4px 14px rgba(0,0,0,.3);pointer-events:none;
}}
.tt:hover::before{{
  content:"";position:absolute;bottom:115%;left:50%;transform:translateX(-50%);
  border:5px solid transparent;border-top-color:#1e293b;z-index:1000;pointer-events:none;
}}

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

tab_sim, tab_changes, tab_data, tab_help = st.tabs(["🔬 Simulation Results", "📋 Forecast Changes", "📊 Data Preview", "❓ Help"])

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
            # On Hand is a snapshot, normally populated in exactly one month
            # column per item — take the first non-null value found, same
            # convention used elsewhere in the app (e.g. the Risk Dashboard's
            # per-item on-hand lookup).
            _oh_rows = updated_pivot[updated_pivot["ORDER_TYPE_FINAL"].str.contains("on hand|3.on hand", case=False, na=False)]
            _oh_month_cols = sorted([c for c in updated_pivot.columns if str(c).startswith("202")])

            def _on_hand_for(item):
                r = _oh_rows[_oh_rows["item"] == item]
                if r.empty:
                    return 0.0
                vals = pd.to_numeric(r[_oh_month_cols].iloc[0], errors="coerce").dropna()
                return float(vals.iloc[0]) if not vals.empty else 0.0

            summary_display = summary_df.copy()
            summary_display.insert(
                1, "description",
                summary_display["raw_material"].map(desc_map).fillna("")
            )
            summary_display.insert(
                2, "on_hand",
                summary_display["raw_material"].map(_on_hand_for)
            )
            st.dataframe(
                summary_display.style.format({"qty_change": "{:,.3f}", "on_hand": "{:,.1f}"}),
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
            filter_planner_pending = st.multiselect(
                "🧭 Filter by Planner Code", options=pc_options,
                default=[], placeholder="All planner codes…",
                help="e.g. RM1 / RM2 — narrows the SKUs available in the item filter below",
                key="pivot_planner_pending",
            )
        # Narrows the item-filter's OPTIONS live (before Apply) so picking is easy;
        # the actual table filtering below only updates once Apply is clicked.
        planner_items_pending = (
            {it for it in all_items_list
             if str(_mm_pivot_filter.get(str(it), {}).get("planner_code", "Unknown")) in filter_planner_pending}
            if filter_planner_pending else set(all_items_list)
        )
        with fc1:
            filter_items_pending = st.multiselect(
                "Filter by item", options=sorted(planner_items_pending),
                default=[], placeholder="Show all items…",
                key="pivot_items_pending",
            )
        with fc2:
            filter_from_onhand_pending = st.checkbox(
                "📅 From On Hand month", value=True,
                help=f"First On Hand month: {first_onhand_month}" if first_onhand_month else "Not detected",
                key="pivot_onhand_pending",
            )
        with fc3:
            filter_changed_only_pending = st.checkbox(
                "🟡 Changed items only", value=False,
                help="Show only SKUs updated by the simulation",
                key="pivot_changed_pending",
            )
        with fc4:
            filter_low_cover_pending = st.checkbox(
                "🔴 COVER < 7 only", value=False,
                help="Show only SKUs with at least one COVER month below 7",
                key="pivot_lowcover_pending",
            )

        filter_planner, filter_items, filter_from_onhand, filter_changed_only, filter_low_cover = _apply_bar(
            "pivot_filters",
            [filter_planner_pending, filter_items_pending, filter_from_onhand_pending,
             filter_changed_only_pending, filter_low_cover_pending],
            [[], [], True, False, False],
            label="Apply filters",
        )

        # ── Apply filters (pure pandas, no render yet) ───────────────
        # Determine which items to show, using the APPLIED planner selection
        planner_items = (
            {it for it in all_items_list
             if str(_mm_pivot_filter.get(str(it), {}).get("planner_code", "Unknown")) in filter_planner}
            if filter_planner else set(all_items_list)
        )
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
        _po_needed_by = getattr(st.session_state, "po_needed_by", {}) or {}
        # Values are dicts (unhashable) — flatten each to a hashable tuple of
        # sorted items so the whole thing can be a frozenset for caching.
        _po_needed_by_key = frozenset((k, tuple(sorted(v.items()))) for k, v in _po_needed_by.items())
        _cover_tips = _build_cover_tooltips(updated_pivot, tuple(all_month_cols), _po_needed_by_key)
        _comp.html(
            render_pivot_html(page_view, tuple(sorted(month_display)), frozenset(changed_items or set()), master_map_key=_ss_key,
                              sub_items=frozenset((k,v) for k,v in (getattr(st.session_state,"sub_items_map",{}) or {}).items()),
                              cover_tooltips=frozenset(_cover_tips.items())),
            height=h, scrolling=True,
        )
        st.caption("💡 Hover any **Cover (mo)** value to see how it was calculated.")

        legend_parts = ["🔴 COVER < SS months", "🟠 PO Recommendation", "🔴⚠️ PO Exception (can't meet lead time — hover for the month it's really needed)", "🔵 Coverage Updated (post-PO)"]
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
            sel_chart_item_pending = st.selectbox(
                "Select SKU to chart", options=chart_candidates, key="pivot_fcst_act_item_pending",
            )
            (sel_chart_item,) = _apply_bar(
                "pivot_fcst_act_chart", [sel_chart_item_pending], [chart_candidates[0]],
                label="Show chart",
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
                        # Sum across rows (an item can have more than one Forecast/Actual row)
                        # instead of taking iloc[0], which silently dropped data for such SKUs.
                        f_vals = (fcst_row[chart_months].apply(pd.to_numeric, errors="coerce").sum(axis=0, min_count=1)
                                  if not fcst_row.empty else pd.Series([np.nan]*len(chart_months), index=chart_months))
                        a_vals = (act_row[chart_months].apply(pd.to_numeric, errors="coerce").sum(axis=0, min_count=1)
                                  if not act_row.empty else pd.Series([np.nan]*len(chart_months), index=chart_months))

                        fig_fa = _go.Figure()
                        fig_fa.add_trace(_go.Bar(x=chart_months, y=a_vals.values, name="Actual", marker_color="#3b82f6"))
                        fig_fa.add_trace(_go.Scatter(x=chart_months, y=f_vals.values, name="Forecast",
                                                      mode="lines+markers", line=dict(color="#f97316", width=2),
                                                      connectgaps=True))
                        st.caption("ℹ️ Forecast is often entered for a few months at a time — the line connects across "
                                   "months with no entry so the trend stays readable; markers show months with real data.")
                        fig_fa.update_layout(
                            title=f"{sel_chart_item} — {desc_s}",
                            height=380, xaxis_title="Month", yaxis_title="Qty",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(t=60),
                        )
                        st.plotly_chart(fig_fa, use_container_width=True)
                except ImportError:
                    st.warning("Plotly not available — install `plotly` to see this chart.")

        # ── Next Purchase Orders — first recommended PO per SKU ─────────
        st.divider()
        st.subheader("📦 Next Purchase Orders — First Recommended PO per SKU")
        st.caption("The earliest month each SKU needs a PO placed, per the PO_RECOMMENDATION logic "
                   "(triggers when coverage drops below safety stock, fills to target coverage).")

        po_mask = updated_pivot["ORDER_TYPE_FINAL"] == "PO_RECOMMENDATION"
        po_by_item_next = _sim_sum_by_item(updated_pivot, po_mask, all_month_cols)
        _mm_po = getattr(st.session_state, "master_map", {}) or {}

        # Substitute items' On Hand stock is already merged into the main item's
        # On Hand row upstream (see _apply_substitutes), so it's already reflected
        # in COVER_MONTHS / this PO recommendation — surface which SKUs benefited
        # from that so it's visible, not just assumed.
        _sub_df_po = getattr(st.session_state, "sub_df", None)
        _main_to_subs = {}
        if _sub_df_po is not None and not _sub_df_po.empty:
            # dedupe — the source mapping table can list the same (Main, Substitute) pair more than once
            _main_to_subs = _sub_df_po.groupby("Main_Item")["Substitute_Item"].apply(
                lambda s: sorted(set(s))
            ).to_dict()

        next_po_rows = []
        for _item, _arr in po_by_item_next.items():
            for _i, _m in enumerate(all_month_cols):
                _v = _arr[_i]
                if pd.notna(_v) and _v > 0:
                    _desc = updated_pivot.loc[updated_pivot["item"] == _item, "description"].dropna()
                    _desc = _desc.iloc[0] if not _desc.empty else ""
                    _subs = _main_to_subs.get(_item, [])
                    next_po_rows.append({
                        "item": _item,
                        "description": _desc,
                        "planner_code": _mm_po.get(str(_item), {}).get("planner_code", ""),
                        "next_po_month": _m,
                        "quantity": round(float(_v), 0),
                        "substitute_stock_included": ", ".join(_subs) if _subs else "",
                    })
                    break  # first (earliest) PO month only

        next_po_df = pd.DataFrame(next_po_rows)
        if next_po_df.empty:
            st.success("✅ No upcoming purchase orders recommended.")
        else:
            next_po_df = next_po_df.sort_values("next_po_month").reset_index(drop=True)

            n_with_sub = int((next_po_df["substitute_stock_included"] != "").sum())
            npk1, npk2, npk3 = st.columns(3)
            npk1.metric("📦 SKUs with an upcoming PO", len(next_po_df))
            npk2.metric("📊 Total recommended quantity", f"{next_po_df['quantity'].sum():,.0f}")
            npk3.metric("🔄 With substitute stock included", n_with_sub)
            if n_with_sub:
                st.caption("🔄 Substitute SKUs' On Hand stock is already merged into the main item's coverage "
                           "before this recommendation is calculated — see **substitute_stock_included** below.")

            planner_opts_po = sorted([p for p in next_po_df["planner_code"].unique().tolist() if p])
            if planner_opts_po:
                po_planner_pending = st.multiselect("Filter by planner code", planner_opts_po, default=[], key="next_po_planner_pending")
                (po_planner_filter,) = _apply_bar("next_po_planner_filter", [po_planner_pending], [[]])
                if po_planner_filter:
                    next_po_df = next_po_df[next_po_df["planner_code"].isin(po_planner_filter)]

            st.dataframe(
                next_po_df.style.format({"quantity": "{:,.0f}"}),
                use_container_width=True, height=min(500, max(150, len(next_po_df) * 38 + 40)),
            )

            buf_po = io.BytesIO()
            with pd.ExcelWriter(buf_po, engine="openpyxl") as w:
                next_po_df.to_excel(w, index=False)
            st.download_button("📥 Download Next PO List", data=buf_po.getvalue(), file_name="next_po_recommendations.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

with tab_data:
    st.subheader("Data Preview")

    # ── Master Data viewer ──────────────────────────────────────
    if hasattr(st.session_state, "master_df") and st.session_state.master_df is not None:
        with st.expander("📋 Master Data (mrp_current.csv) — LT / SS / SL / Max Inv", expanded=False):
            md = st.session_state.master_df.reset_index()
            pc_filter_pending = st.multiselect("Filter by Planner Code",
                sorted(md["planner_code"].dropna().unique().tolist()),
                default=["RM1","RM2"], key="md_pc_filter_pending")
            (pc_filter,) = _apply_bar("master_data_planner_filter", [pc_filter_pending], [["RM1","RM2"]])
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
            sel_bom_pending = st.selectbox("Filter BOM by product", ["All"] + bom_items, key="bom_prod_filter_pending")
            (sel_bom,) = _apply_bar("bom_prod_filter", [sel_bom_pending], ["All"])
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
                sel_pending = st.multiselect("Filter items", items, default=items[:5], key="preview_filter_pending")
            with pc2:
                from_oh_pending = st.checkbox(
                    "📅 From first On Hand month", value=True, key="preview_from_oh_pending",
                    help=f"First On Hand month: {first_oh}" if first_oh else "Not detected",
                )
            sel, from_oh = _apply_bar("data_preview_filters", [sel_pending, from_oh_pending], [items[:5], True])

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
| 🔴 Red cell | COVER below the item's Safety Stock (original or updated coverage) |
| 🟠 Orange cell | PO Recommendation — placed with full lead time, arrives exactly when needed |
| 🔴⚠️ Red cell with ⚠️ | PO Exception — lead time doesn't allow placing this order in time; hover the cell for the month it was truly needed |
| 🔵 Blue row | Coverage Months Updated — projected coverage after PO injections |
| 🟡 Yellow row | Item updated by simulation |

### PO Recommendation Logic

The system continuously monitors coverage across the planning horizon, per item,
using that item's own master-data values (📋 Master Data row: LT / SS / SL / MAX) —
not a single fixed threshold for every SKU.

| Parameter | Meaning |
|-----------|---------|
| **Trigger (Safety Stock, SS)** | A PO is recommended only once projected coverage drops below the item's own SS (months) |
| **Target (Max Inventory, MAX)** | The order is sized to bring coverage up to MAX months at the point it arrives — never more |
| **Lead Time (LT)** | The order is placed LT months before it's needed so it arrives just in time. If there isn't enough runway left to do that, the order still goes out as early as possible — it just arrives late, and the cell is flagged 🔴⚠️ **PO Exception** with a hover tooltip showing the month it was truly needed |
| **Shelf Life (SL)** | Order quantity is capped at (SL − LT) months of demand — the usable shelf life once transit time is accounted for — so stock isn't ordered faster than it can be sold before expiring |
| **Cycle** | Repeats — whenever updated coverage drops below SS again, a new PO is generated |

The **Cover Updated** row tracks projected coverage *after* all recommended POs are applied, enabling you to verify the replenishment plan and spot any subsequent dips that trigger additional orders.

### Reload Data
Click **🔄 Reload Data** in the sidebar to refresh from the database.
    """)

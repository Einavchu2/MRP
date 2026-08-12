"""
simulation.py  –  BOM explosion engine + INV / COVER calculation
PATH-AWARE version using CONVERSION_RATE for correct package UOM quantities.

Based on: import_pandas_as_pd.py (auto PATH-aware + package conversion engine)
"""

import pandas as pd
import numpy as np
import math
from dataclasses import dataclass, field

try:
    from src.master_data import load_master, get_master_map
    _MASTER_LOADED = True
except Exception:
    _MASTER_LOADED = False


@dataclass
class SimulationConfig:
    parent_product:     str
    month:              str
    production_change:  float
    action:             str           # 'ADD' | 'REMOVE'
    search_mode:        str           # 'DIRECT' | 'DEEP' | 'PARTIAL'
    products_to_exclude: list = field(default_factory=list)


# ─── helpers (extracted from notebook) ──────────────────────────────────────

def _normalize_code(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def _round_package_qty(package_qty):
    if pd.isna(package_qty):
        return np.nan
    if package_qty > 0:
        return math.ceil(package_qty)
    if package_qty < 0:
        return -math.ceil(abs(package_qty))
    return 0


# ─── main entry point ────────────────────────────────────────────────────────

def run_simulation(
    full_bom: pd.DataFrame,
    pivot_df: pd.DataFrame,
    cfg: SimulationConfig,
) -> dict:
    """
    Explode BOM using PATH-aware engine, update forecast, recalculate INV+COVER.
    Returns dict with keys: pivot, results_df, summary_df, bom_issues_df, month
    """

    import warnings
    warnings.filterwarnings("ignore")

    # ── inputs (mirror notebook variables) ───────────────────────────────────
    parent_product   = _normalize_code(cfg.parent_product)
    month            = cfg.month
    production_change = float(cfg.production_change)
    action           = cfg.action
    search_mode      = cfg.search_mode
    PRODUCTS_TO_EXCLUDE = [_normalize_code(p) for p in cfg.products_to_exclude]
    QTY_FIELD        = "ING_QTY"
    DEMAND_IN_RUNS   = True
    AUTO_PATH_BY_PARENT = True
    UPDATE_FORECAST_IN_PACKAGE_UOM = True

    # ── 1. clean BOM ─────────────────────────────────────────────────────────
    bom   = full_bom.copy()
    pivot = pivot_df.copy()

    bom.columns   = bom.columns.str.strip()
    pivot.columns = pivot.columns.str.strip()

    required_bom_cols = [
        "PRODUCT", "INGREDIENT", "PRODUCT_UOM", "ING_UOM",
        "PRUDUCT_QTY", "ING_QTY", "PATH", "CONVERSION_RATE"
    ]
    missing_cols = [c for c in required_bom_cols if c not in bom.columns]
    if missing_cols:
        raise ValueError(f"Missing required BOM columns: {missing_cols}")

    for opt_col, default in [
        ("PRIMARY_UOM_CODE", bom["ING_UOM"]),
        ("UOM_CODE",         bom["ING_UOM"]),
        ("BOM_CONV",         bom["ING_QTY"]),
    ]:
        if opt_col not in bom.columns:
            bom[opt_col] = default

    for opt_col, default in [("CONV_STATUS","OK"),("FORMULA_DESC1",""),("INGREDIENT_DESCRIPTION","")]:
        if opt_col not in bom.columns:
            bom[opt_col] = default

    for c in ["PRODUCT", "INGREDIENT"]:
        bom[c] = bom[c].apply(_normalize_code)

    for c in ["PRODUCT_UOM","ING_UOM","PRIMARY_UOM_CODE","UOM_CODE",
              "CONV_STATUS","FORMULA_DESC1","INGREDIENT_DESCRIPTION"]:
        bom[c] = bom[c].astype(str).str.strip()

    pivot["item"]            = pivot["item"].apply(_normalize_code)
    pivot["ORDER_TYPE_FINAL"] = pivot["ORDER_TYPE_FINAL"].astype(str).str.strip()

    for c in ["PRUDUCT_QTY","ING_QTY","BOM_CONV","CONVERSION_RATE"]:
        bom[c] = pd.to_numeric(bom[c], errors="coerce").fillna(0)

    bom["PATH"] = bom["PATH"].astype(str).str.strip().str.replace("//","/",regex=False)

    def path_to_nodes(path):
        if pd.isna(path):
            return []
        return [_normalize_code(x) for x in str(path).strip("/").split("/") if _normalize_code(x)]

    bom["PATH_NODES"] = bom["PATH"].apply(path_to_nodes)

    # ── 2. lookups ───────────────────────────────────────────────────────────
    all_products    = set(bom["PRODUCT"].dropna().unique())
    all_ingredients = set(bom["INGREDIENT"].dropna().unique())
    raw_materials   = all_ingredients - all_products
    intermediates   = all_ingredients & all_products

    base_qty        = bom.groupby("PRODUCT")["PRUDUCT_QTY"].first().to_dict()
    prod_uom        = bom.groupby("PRODUCT")["PRODUCT_UOM"].first().to_dict()
    prim_uom        = bom.groupby("INGREDIENT")["PRIMARY_UOM_CODE"].first().to_dict()
    recipe_uom      = bom.groupby("INGREDIENT")["ING_UOM"].first().to_dict()
    product_desc    = bom.groupby("PRODUCT")["FORMULA_DESC1"].first().to_dict()
    ingredient_desc = bom.groupby("INGREDIENT")["INGREDIENT_DESCRIPTION"].first().to_dict()
    conv_rate       = bom.groupby("INGREDIENT")["CONVERSION_RATE"].first().to_dict()

    def get_conversion_rate(item):
        item = _normalize_code(item)
        r = conv_rate.get(item, 1)
        if pd.isna(r) or r == 0:
            return 1
        return float(r)

    def get_package_size(item):
        r = get_conversion_rate(item)
        return np.nan if r == 0 else 1 / r

    def convert_qty_to_package(qty, item):
        return qty * get_conversion_rate(item)

    # ── 3. BOM issue detection ────────────────────────────────────────────────
    _edge_count = bom.groupby(["PRODUCT","INGREDIENT","PATH"]).size()
    DUP_EDGES   = set(_edge_count[_edge_count>1].reset_index()[["PRODUCT","INGREDIENT","PATH"]].itertuples(index=False,name=None))
    ZERO_BASE   = {p for p,b in base_qty.items() if not b or pd.isna(b)}
    NO_CONV     = set(bom.loc[bom["CONV_STATUS"].astype(str).str.upper()!="OK","INGREDIENT"])
    _prim_mode  = bom.groupby("INGREDIENT")["PRIMARY_UOM_CODE"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
    UOM_MISMATCH = {x for x in intermediates if _prim_mode.get(x) != prod_uom.get(x)}

    def check_bom_issue(path, slash_path=""):
        nodes = path.split(" -> ") if isinstance(path,str) else list(path)
        tags  = []
        for p,c in zip(nodes[:-1],nodes[1:]):
            if slash_path and (p,c,slash_path) in DUP_EDGES:
                tags.append(f"DUPLICATE_EDGE_PATH:{p}->{c}")
            b = base_qty.get(p)
            if (p in ZERO_BASE) or (not b) or pd.isna(b):
                tags.append(f"MISSING_BASE:{p}")
        if nodes:
            leaf = nodes[-1]
            if leaf in NO_CONV:     tags.append("NO_CONVERSION")
            if leaf in UOM_MISMATCH: tags.append(f"UOM_MISMATCH:{leaf}")
        return ";".join(dict.fromkeys(tags))

    _issue_rows = []
    for p,c,path in DUP_EDGES:
        rws = bom[(bom["PRODUCT"]==p)&(bom["INGREDIENT"]==c)&(bom["PATH"]==path)]
        _issue_rows.append({"ISSUE":"DUPLICATE_EDGE_PATH","PRODUCT":p,"INGREDIENT":c,
            "DETAIL":f"{len(rws)} dup rows, PATH={path}"})
    for p in ZERO_BASE:
        _issue_rows.append({"ISSUE":"MISSING_BASE","PRODUCT":p,"INGREDIENT":"","DETAIL":"PRUDUCT_QTY=0"})
    for c in NO_CONV:
        _issue_rows.append({"ISSUE":"NO_CONVERSION","PRODUCT":"","INGREDIENT":c,"DETAIL":"CONV_STATUS!=OK"})
    for x in UOM_MISMATCH:
        _issue_rows.append({"ISSUE":"UOM_MISMATCH","PRODUCT":"","INGREDIENT":x,
            "DETAIL":f"primary={_prim_mode.get(x)} vs product_uom={prod_uom.get(x)}"})
    bom_issues_df = pd.DataFrame(_issue_rows, columns=["ISSUE","PRODUCT","INGREDIENT","DETAIL"])

    # ── 4. PATH-aware search engine ──────────────────────────────────────────
    results = []

    def row_starts_with_parent(row):
        nodes = row["PATH_NODES"]
        return isinstance(nodes,list) and len(nodes)>0 and nodes[0]==parent_product

    def row_matches_current_branch(row, current_nodes, ingredient):
        nodes = row["PATH_NODES"]
        if not isinstance(nodes,list) or len(nodes)==0:
            return False
        return nodes == current_nodes + [ingredient]

    def search_bom(current_product, demand, level=0, path_nodes=None, chain=None):
        nonlocal results
        current_product = _normalize_code(current_product)
        if path_nodes is None: path_nodes = [current_product]
        if chain is None:      chain = set()
        if current_product in chain:
            return
        chain = chain | {current_product}

        current_rows = bom[bom["PRODUCT"]==current_product].copy()
        if current_rows.empty: return

        if AUTO_PATH_BY_PARENT:
            current_rows = current_rows[current_rows["PATH_NODES"].apply(
                lambda nodes: isinstance(nodes,list) and len(nodes)>0 and nodes[0]==parent_product
            )].copy()
        if current_rows.empty: return

        base = base_qty.get(current_product, np.nan)
        if not base or pd.isna(base):
            base = 0

        for _, row in current_rows.iterrows():
            ingredient = _normalize_code(row["INGREDIENT"])
            qty        = row[QTY_FIELD]
            next_nodes = path_nodes + [ingredient]

            if AUTO_PATH_BY_PARENT and not row_starts_with_parent(row): continue
            if not row_matches_current_branch(row, path_nodes, ingredient): continue

            rate = (qty / base) if base else 0.0
            req  = demand * rate

            current_path = " -> ".join(next_nodes)
            slash_path   = "/" + "/".join(next_nodes)

            is_raw      = ingredient in raw_materials
            is_excluded = search_mode=="PARTIAL" and ingredient in PRODUCTS_TO_EXCLUDE

            edge_record = {
                "parent_product":      parent_product,
                "found_under_product": current_product,
                "ingredient":          ingredient,
                "ingredient_desc":     ingredient_desc.get(ingredient,""),
                "component_type":      "RAW" if is_raw else "EXCLUDED_STOP" if is_excluded else "INTERMEDIATE",
                "bom_qty":             qty,
                "parent_base_qty":     base,
                "rate_per_unit":       rate,
                "parent_req":          demand,
                "calculated_req":      req,
                "production_change":   production_change,
                "qty_change":          np.nan,
                "qty_change_conv":     np.nan,
                "qty_change_conv_rounded": np.nan,
                "conversion_rate":     get_conversion_rate(ingredient),
                "package_size":        get_package_size(ingredient),
                "uom":                 str(row.get("ING_UOM", recipe_uom.get(ingredient,""))),
                "primary_uom":         prim_uom.get(ingredient,""),
                "package_uom":         prim_uom.get(ingredient,""),
                "level":               level+1,
                "path":                current_path,
                "slash_path":          slash_path,
                "search_mode":         search_mode,
                "CHECK_BOM_ISSUE":     check_bom_issue(current_path, slash_path),
            }

            if is_raw or is_excluded:
                qty_change      = req if action=="ADD" else -req
                qty_change_conv = convert_qty_to_package(qty_change, ingredient)
                edge_record["qty_change"]              = qty_change
                edge_record["qty_change_conv"]         = qty_change_conv
                edge_record["qty_change_conv_rounded"] = _round_package_qty(qty_change_conv)
                edge_record["raw_material"]            = ingredient
                results.append(edge_record)
                continue

            if search_mode == "DIRECT":
                continue

            if ingredient in all_products:
                trace = edge_record.copy()
                trace["raw_material"] = ""
                results.append(trace)
                search_bom(ingredient, req, level+1, next_nodes, chain)

    # ── 5. run ───────────────────────────────────────────────────────────────
    top_base = base_qty.get(parent_product, 1) or 1
    effective_demand = production_change * top_base if DEMAND_IN_RUNS else production_change

    search_bom(parent_product, effective_demand, 0, [parent_product])

    results_df = pd.DataFrame(results)

    # ── 6. aggregate summary ─────────────────────────────────────────────────
    if not results_df.empty and "raw_material" in results_df.columns:
        raw_detail_df = results_df[results_df["raw_material"].astype(str).str.strip()!=""].copy()
    else:
        raw_detail_df = pd.DataFrame()

    if not raw_detail_df.empty:
        summary_df = (
            raw_detail_df
            .groupby(["raw_material","ingredient_desc","uom","package_uom","conversion_rate","package_size"], as_index=False)
            .agg(qty_change=("qty_change","sum"), qty_change_conv=("qty_change_conv","sum"),
                 path_count=("slash_path","nunique"))
        )
        summary_df["qty_change_conv_rounded"] = summary_df["qty_change_conv"].apply(_round_package_qty)
        # rename ingredient_desc → raw_material_desc for display
        summary_df = summary_df.rename(columns={"ingredient_desc":"raw_material_desc"})
    else:
        summary_df = pd.DataFrame(columns=["raw_material","raw_material_desc","qty_change","qty_change_conv"])

    # ── 7. update forecast in pivot ──────────────────────────────────────────
    if month not in pivot.columns:
        pivot[month] = 0

    for _, row in summary_df.iterrows():
        material = _normalize_code(row["raw_material"])
        qty_to_add = pd.to_numeric(
            row["qty_change_conv"] if UPDATE_FORECAST_IN_PACKAGE_UOM else row["qty_change"],
            errors="coerce") or 0

        fcst_mask = ((pivot["item"]==material) &
                     pivot["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False))
        if fcst_mask.sum()==0:
            pivot = pd.concat([pivot, pd.DataFrame([{"item":material,"ORDER_TYPE_FINAL":"1.Forecast",month:qty_to_add}])], ignore_index=True)
        else:
            current_val = pd.to_numeric(pivot.loc[fcst_mask, month], errors="coerce").fillna(0)
            pivot.loc[fcst_mask, month] = current_val + qty_to_add

    # ── 8. recalculate INV + COVER ───────────────────────────────────────────
    pivot = _compute_inv_cover(pivot)

    return {
        "pivot":         pivot,
        "results_df":    results_df,
        "summary_df":    summary_df,
        "bom_issues_df": bom_issues_df,
        "month":         cfg.month,
    }


# ─── INV / COVER (unchanged from previous version) ──────────────────────────

_DEFAULT_SS     = 7
_DEFAULT_TARGET = 15
_DEFAULT_LT     = 3
_DEFAULT_SL     = 24


def _window_mean(arr: np.ndarray) -> float:
    """Mirrors pandas' `series[window].fillna(0).mean()`: NaN only for an empty window."""
    return float(arr.mean()) if arr.size > 0 else np.nan


def _sum_by_item(pivot: pd.DataFrame, mask: pd.Series, month_cols: list) -> dict:
    """Numeric-sum the given rows per item, once, instead of per-material full-table scans."""
    sub = pivot.loc[mask, ["item"] + month_cols]
    if sub.empty:
        return {}
    sub = sub.copy()
    sub[month_cols] = sub[month_cols].apply(pd.to_numeric, errors="coerce")
    grouped = sub.groupby("item")[month_cols].sum(min_count=1)
    return {item: row.to_numpy(dtype=float) for item, row in grouped.iterrows()}


def _compute_inv_cover(pivot: pd.DataFrame, master_map: dict = None) -> pd.DataFrame:
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    n_months   = len(month_cols)

    if master_map is None and _MASTER_LOADED:
        try:
            master_map = get_master_map(load_master())
        except Exception:
            master_map = {}
    master_map = master_map or {}

    pivot = pivot[~pivot["ORDER_TYPE_FINAL"].isin(
        ["INV","COVER_MONTHS","PO_RECOMMENDATION","COVER_MONTHS_UPDATED","PO_EXCEPTION"]
    )].copy()

    new_rows = []

    # ── Pre-group rows by item ONCE (was: 3 full-table rescans per material) ──
    order_type = pivot["ORDER_TYPE_FINAL"]
    is_fcst    = order_type.str.contains("forecast", case=False, na=False)
    is_po      = order_type.isin(["Purchase order"])
    is_onhand  = (order_type.str.contains("on hand|onhand", case=False, na=False) |
                  (order_type == "3.On Hand"))

    fcst_by_item   = _sum_by_item(pivot, is_fcst,   month_cols)
    po_by_item     = _sum_by_item(pivot, is_po,     month_cols)
    onhand_by_item = _sum_by_item(pivot, is_onhand, month_cols)

    for material in pivot["item"].dropna().unique():
        m_data     = master_map.get(str(material), {})
        ss         = float(m_data.get("safety_stock",  _DEFAULT_SS))
        target_cov = float(m_data.get("max_inventory", _DEFAULT_TARGET))
        lead_time  = float(m_data.get("lead_time",     _DEFAULT_LT))
        shelf_life = float(m_data.get("shelf_life",    _DEFAULT_SL))

        forecast_raw = fcst_by_item.get(material)
        po_raw       = po_by_item.get(material)
        onhand_raw   = onhand_by_item.get(material)

        if forecast_raw is None and po_raw is None and onhand_raw is None:
            continue

        inv_row    = {"item":material,"ORDER_TYPE_FINAL":"INV"}
        cover_row  = {"item":material,"ORDER_TYPE_FINAL":"COVER_MONTHS"}
        po_rec_row = {"item":material,"ORDER_TYPE_FINAL":"PO_RECOMMENDATION"}

        if forecast_raw is None:
            forecast_raw = np.full(n_months, np.nan)
        # clip(lower=0).ffill().fillna(0) — tiny (n_months long), pandas is fine here
        forecast_arr = pd.Series(forecast_raw).clip(lower=0).ffill().fillna(0).to_numpy(dtype=float)
        po_arr       = np.nan_to_num(po_raw,     nan=0.0) if po_raw     is not None else np.zeros(n_months)
        onhand_arr   = np.nan_to_num(onhand_raw, nan=0.0) if onhand_raw is not None else np.zeros(n_months)

        first_oh_idx = next((i for i in range(n_months) if onhand_arr[i] > 0), None)
        if first_oh_idx is None: continue
        first_oh = month_cols[first_oh_idx]

        started = False
        running_inventory = 0.0
        inv_arr   = np.full(n_months, np.nan)
        cover_arr = np.full(n_months, np.nan)

        for i, m in enumerate(month_cols):
            if not started:
                if m==first_oh: started=True
                else:
                    inv_row[m]=cover_row[m]=po_rec_row[m]=np.nan
                    continue
            running_inventory = running_inventory + po_arr[i] - forecast_arr[i]
            if m==first_oh: running_inventory = onhand_arr[i] + po_arr[i] - forecast_arr[i]
            inv_row[m] = running_inventory
            inv_arr[i] = running_inventory
            future_avg = _window_mean(forecast_arr[i:i+7])
            cover = running_inventory/future_avg if future_avg>0 else np.nan
            cover_row[m] = cover
            cover_arr[i] = cover

        running_inv_sim = inv_arr.copy()
        running_cov_sim = cover_arr.copy()
        for m in month_cols: po_rec_row[m] = np.nan

        i = 0
        while i < n_months:
            m = month_cols[i]
            inv   = running_inv_sim[i]
            cover = running_cov_sim[i]
            if np.isnan(inv) or np.isnan(cover):
                i+=1; continue
            if cover < ss:
                future_avg    = _window_mean(forecast_arr[i:i+int(target_cov)])
                target_inv    = target_cov * future_avg
                sl_cap        = shelf_life * future_avg if future_avg>0 else np.inf
                target_inv    = min(target_inv, sl_cap)
                po_qty        = max(target_inv - inv, 0)
                if po_qty > 0:
                    lt_idx   = max(0, i-int(round(lead_time)))
                    po_month = month_cols[lt_idx]
                    if pd.isna(po_rec_row.get(po_month, np.nan)):
                        po_rec_row[po_month] = po_qty
                    new_inv = inv + po_qty
                    for j in range(i, n_months):
                        new_inv = (inv + po_qty - forecast_arr[j]) if j==i else (new_inv + po_arr[j] - forecast_arr[j])
                        running_inv_sim[j] = new_inv
                        avg_f = _window_mean(forecast_arr[j:j+7])
                        running_cov_sim[j] = new_inv/avg_f if avg_f>0 else np.nan
            i+=1

        new_rows.extend([inv_row, cover_row, po_rec_row])

    if new_rows:
        pivot = pd.concat([pivot, pd.DataFrame(new_rows)], ignore_index=True)

    return pivot


def compute_po_recommendations(pivot_df: pd.DataFrame, month_cols: list, master_map: dict = None) -> pd.DataFrame:
    """
    PO Recommendation logic:
    1. LEAD TIME constraint: PO cannot be placed before first_active_month + LT.
       LT is measured from the first month that appears in the pivot.
    2. EXCEPTION flag: If cover < 80% of SS at the trigger point → PO_EXCEPTION.
    3. COVER_MONTHS_UPDATED: Forward-propagated coverage after all PO injections.
    """
    master_map = master_map or {}
    if not month_cols:
        return pivot_df

    keep_mask = ~pivot_df["ORDER_TYPE_FINAL"].isin(
        ["PO_RECOMMENDATION", "PO_EXCEPTION", "COVER_MONTHS_UPDATED"]
    )
    pivot_df = pivot_df[keep_mask].copy()

    new_rows: list[dict] = []
    first_pivot_month = month_cols[0]

    for item, grp in pivot_df.groupby("item", sort=False):
        description = grp["description"].dropna().iloc[0] if not grp["description"].dropna().empty else ""

        _mp         = master_map.get(str(item), {})
        item_ss     = float(_mp.get("safety_stock",  _DEFAULT_SS) or _DEFAULT_SS)
        item_target = float(_mp.get("max_inventory", _DEFAULT_TARGET) or _DEFAULT_TARGET)
        item_lt     = float(_mp.get("lead_time",     _DEFAULT_LT) or _DEFAULT_LT)
        item_sl     = float(_mp.get("shelf_life",    _DEFAULT_SL) or _DEFAULT_SL)

        exception_threshold = item_ss * 0.80
        lt_months_int = int(round(item_lt))

        oh_rows_item = grp[grp["ORDER_TYPE_FINAL"].str.contains("on hand|3.on", case=False, na=False)]
        first_oh_idx = 0
        for _mi, _m in enumerate(month_cols):
            if not oh_rows_item.empty and _m in oh_rows_item.columns:
                _v = pd.to_numeric(oh_rows_item[_m].iloc[0], errors="coerce")
                if pd.notna(_v) and _v > 0:
                    first_oh_idx = _mi
                    break

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

        demand_rate = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(inv_vals[i]) and not np.isnan(cover_vals[i]) and cover_vals[i] > 0:
                demand_rate[i] = inv_vals[i] / cover_vals[i]
        last_rate = np.nan
        for i in range(n):
            if not np.isnan(demand_rate[i]):
                last_rate = demand_rate[i]
            elif not np.isnan(last_rate):
                demand_rate[i] = last_rate
        first_rate = next((demand_rate[i] for i in range(n) if not np.isnan(demand_rate[i])), 0.0)
        for i in range(n):
            if np.isnan(demand_rate[i]):
                demand_rate[i] = first_rate

        po_rec        = np.zeros(n)
        po_is_except  = np.zeros(n, dtype=bool)
        updated_cover = cover_vals.copy()

        proj_inv = inv_vals.copy()
        for i in range(1, n):
            if np.isnan(proj_inv[i]) and not np.isnan(proj_inv[i - 1]):
                decay = demand_rate[i - 1] if not np.isnan(demand_rate[i - 1]) else 0
                proj_inv[i] = max(proj_inv[i - 1] - decay, 0)

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
                    is_exception = cur_cover < exception_threshold
                    if is_exception:
                        place_idx = i
                    else:
                        place_idx = max(i, first_po_idx)
                        place_idx = min(place_idx, len(month_cols) - 1)

                    po_rec[place_idx] = round(order_qty, 0)
                    po_is_except[place_idx] = is_exception

                    cumulative_po = order_qty
                    for j in range(place_idx, n):
                        base_inv = original_inv[j] if not np.isnan(original_inv[j]) else (
                            max((original_inv[j-1] if j>0 and not np.isnan(original_inv[j-1]) else 0)
                                - (demand_rate[j-1] if not np.isnan(demand_rate[j-1]) else 0), 0)
                        )
                        proj_inv[j]      = base_inv + cumulative_po
                        updated_cover[j] = (proj_inv[j] / demand_rate[j]) if demand_rate[j] > 0 else np.nan
                        cumulative_po = max(cumulative_po - demand_rate[j], 0)

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

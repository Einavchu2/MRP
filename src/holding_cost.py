"""
holding_cost.py  –  Inventory Holding Cost analysis.

Identifies materials that are expensive to KEEP in inventory over time —
ranked on average inventory value held across the analysis window, not on
today's value alone.

Data (via the API's /inventory-history, all org EM items, any planner code —
narrowed to RM1 / RM2 / Other by filter_history() before the analysis):
  - current on-hand        BTG_OnHand_V, all subinventories
  - inventory history      RECONSTRUCTED: today's on-hand rolled backwards
                           through every (deduplicated) Material_Transactions
                           row. Validated 2026-10-29 against the last
                           BTG_Cover_Monts_Weekly_History snapshot
                           (2025-06-25): 94% of items within 1%, 97% within 5%.
  - unit cost              Oracle item cost, current period (ILS). Current
                           cost is applied to the whole window.
  - consumption            WIP / Account alias / Move Order issues — the same
                           definition the MRP pivot uses
  - target inventory       mrp_current.csv "MAX INVENTORY" (months of cover)
                           × average monthly consumption. Blank = no target,
                           never defaulted.
  - forward demand         ASCP Forecast + Planned/Work order demand

All thresholds, ABC cut-offs, score weights and the carrying-cost rate come
from holding_cost_config.json — nothing business-specific is hard-coded here.
"""
import json
import os

import numpy as np
import pandas as pd

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "holding_cost_config.json")

_DAYS_PER_MONTH = 365.25 / 12

FLAG_ORDER = [
    "HIGH HOLDING COST",
    "HIGH VALUE + HIGH DAYS ON HAND",
    "HIGH VALUE + LOW CONSUMPTION",
    "HIGH INVENTORY + NO RECENT CONSUMPTION",
    "EXCESS ABOVE TARGET",
    "SLOW MOVING",
    "POTENTIALLY OBSOLETE",
]


PLANNER_GROUPS = ["RM1", "RM2", "Other"]
_PLANNER_GROUP_CODES = {"B_RM1": "RM1", "B_RM2": "RM2"}


def load_config(path: str = _CONFIG_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def planner_groups(items: pd.DataFrame) -> pd.Series:
    """ITEM → "RM1" / "RM2" / "Other" (any other or missing planner code)."""
    md = items.drop_duplicates("ITEM").set_index("ITEM")["PLANNER_CODE"]
    return md.map(_PLANNER_GROUP_CODES).fillna("Other")


def filter_history(hist: dict, groups: list) -> dict:
    """
    Keep only items whose planner group is in `groups`. Done BEFORE
    compute_holding_cost so ranking, ABC and score are relative to the
    selected group. Items missing from MasterItemsBI count as "Other".
    """
    grp = planner_groups(hist["items"])
    return {k: df[df["ITEM"].map(grp).fillna("Other").isin(groups)] for k, df in hist.items()}


# ─────────────────────────────────────────────
# STEP 2a – reconstruct daily on-hand history
# ─────────────────────────────────────────────

def reconstruct_daily_onhand(onhand: pd.DataFrame, daily_net: pd.DataFrame,
                             days: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    End-of-day on-hand qty per item for the `days` days ending the day before
    `as_of` (the on-hand snapshot date). Returns items × dates.

    qty(d) = on_hand_now − Σ net transactions on days after d.
    Transactions dated on the snapshot day itself are ignored: the snapshot
    is taken early morning, so they mostly happened after it.
    """
    end = as_of.normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    oh = onhand.set_index("ITEM")["ONHAND_QTY"].astype(float)
    net = daily_net.copy()
    net["TRX_DAY"] = pd.to_datetime(net["TRX_DAY"])
    net = net[(net["TRX_DAY"] > dates[0]) & (net["TRX_DAY"] <= end)]
    net_m = (net.pivot_table(index="ITEM", columns="TRX_DAY", values="NET_QTY", aggfunc="sum")
                .reindex(columns=dates, fill_value=0).fillna(0))

    items = oh.index.union(net_m.index)
    net_m = net_m.reindex(items, fill_value=0)
    oh = oh.reindex(items).fillna(0)

    # Σ net on days strictly after each date: reverse cumulative sum, shifted one day.
    after = net_m.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1].shift(-1, axis=1).fillna(0)
    return after.rsub(oh, axis=0)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _pct_rank(s: pd.Series, zero_is_zero: bool = True) -> pd.Series:
    """0–1 percentile rank. Robust to the extreme skew of inventory values
    (min–max scaling would let one item compress everyone else to ~0).
    Values ≤ 0 score 0 when zero_is_zero — 'no excess' must not earn points."""
    r = s.rank(pct=True, method="average")
    if zero_is_zero:
        r = r.where(s > 0, 0.0)
    return r.fillna(0.0)


def _fmt_days(v: float) -> str:
    return "ללא צריכה" if not np.isfinite(v) else f"{v:,.0f} ימי מלאי"


# ─────────────────────────────────────────────
# STEPS 2–7 – metrics, ranking, ABC, flags, score
# ─────────────────────────────────────────────

def compute_holding_cost(hist: dict, master: pd.DataFrame, config: dict,
                         rate: float = None) -> dict:
    """
    hist   — dict of DataFrames from load_inventory_history()
    master — load_master(fill_defaults=False)
    rate   — annual carrying cost rate (fraction); overrides the config value.
             None → holding-cost amounts stay NaN; ranking uses average
             inventory value, which gives the identical order.

    Returns {"ranking": DataFrame, "excluded": DataFrame, "daily_value": DataFrame,
             "kpis": dict, "rate": float|None, "as_of": Timestamp, "days": int}
    """
    days = int(config.get("analysis_days", 365))
    if rate is None:
        rate = config.get("carrying_cost_rate_annual")
    abc_t = config["abc_thresholds"]
    fl = config["flags"]
    weights = config["score_weights"]

    onhand = hist["onhand"].copy()
    onhand["ONHAND_QTY"] = pd.to_numeric(onhand["ONHAND_QTY"], errors="coerce").fillna(0)
    as_of = pd.to_datetime(onhand["ONHAND_DATE"]).max() if not onhand.empty else pd.Timestamp.today()

    daily_net = hist["daily_net"].copy()
    daily_net["NET_QTY"] = pd.to_numeric(daily_net["NET_QTY"], errors="coerce").fillna(0)
    qty_hist = reconstruct_daily_onhand(onhand, daily_net, days, as_of)
    neg_days = (qty_hist < -1e-9).sum(axis=1)
    qty_hist = qty_hist.clip(lower=0)

    # ── Unit cost (one row per item expected; take the max if Oracle has several) ──
    cost = hist["cost"].copy()
    cost["ITEM_COST"] = pd.to_numeric(cost["ITEM_COST"], errors="coerce")
    unit_cost = cost.groupby("ITEM")["ITEM_COST"].max()

    items_md = hist["items"].drop_duplicates("ITEM").set_index("ITEM")

    # ── Consumption over the same window ──
    issue = hist["daily_issue"].copy()
    issue["TRX_DAY"] = pd.to_datetime(issue["TRX_DAY"])
    issue["ISSUE_QTY"] = pd.to_numeric(issue["ISSUE_QTY"], errors="coerce").fillna(0)
    issue = issue[(issue["TRX_DAY"] >= qty_hist.columns[0]) & (issue["TRX_DAY"] <= qty_hist.columns[-1])]
    cons_qty = issue.groupby("ITEM")["ISSUE_QTY"].sum()
    last_cons = issue[issue["ISSUE_QTY"] > 0].groupby("ITEM")["TRX_DAY"].max()

    fwd = hist["fwd_demand"].set_index("ITEM")["FWD_DEMAND_QTY"].astype(float) \
        if not hist["fwd_demand"].empty else pd.Series(dtype=float)

    df = pd.DataFrame(index=qty_hist.index)
    df.index.name = "item"
    df["description"] = items_md["DESCRIPTION"].reindex(df.index)
    df["category"] = items_md["CATEGORY"].reindex(df.index)
    df["planner_code"] = items_md["PLANNER_CODE"].reindex(df.index)
    df["planner_group"] = df["planner_code"].map(_PLANNER_GROUP_CODES).fillna("Other")
    df["uom"] = items_md["PRIMARY_UOM_CODE"].reindex(df.index)
    df["unit_cost"] = unit_cost.reindex(df.index)
    df["list_price"] = pd.to_numeric(items_md["LIST_PRICE"], errors="coerce").reindex(df.index) \
        if "LIST_PRICE" in items_md.columns else np.nan

    df["current_qty"] = onhand.set_index("ITEM")["ONHAND_QTY"].reindex(df.index).fillna(0)
    df["avg_qty"] = qty_hist.mean(axis=1)
    df["max_qty"] = qty_hist.max(axis=1)
    df["recon_negative_days"] = neg_days.reindex(df.index).fillna(0).astype(int)

    # Only items that actually held stock at some point in the window.
    df = df[(df["current_qty"] > 0) | (df["max_qty"] > 0)]

    excluded = df[df["unit_cost"].isna() | (df["unit_cost"] <= 0)].copy()
    excluded["reason"] = "אין עלות פריט ב-Oracle לתקופה הנוכחית — לא ניתן לתמחר"
    df = df.drop(excluded.index)

    df["current_value"] = df["current_qty"] * df["unit_cost"]
    df["avg_value"] = df["avg_qty"] * df["unit_cost"]
    df["max_value"] = df["max_qty"] * df["unit_cost"]

    df["consumption_qty"] = cons_qty.reindex(df.index).fillna(0)
    df["avg_daily_consumption"] = df["consumption_qty"] / days
    df["days_on_hand"] = np.where(df["avg_daily_consumption"] > 0,
                                  df["current_qty"] / df["avg_daily_consumption"].replace(0, np.nan),
                                  np.inf)
    # Inventory turns: annualised consumption value / average inventory value.
    df["turns"] = np.where(df["avg_value"] > 0,
                           df["consumption_qty"] * df["unit_cost"] * (365 / days) / df["avg_value"].replace(0, np.nan),
                           np.nan)
    df["last_consumption"] = last_cons.reindex(df.index)
    df["days_since_consumption"] = (as_of.normalize() - df["last_consumption"]).dt.days
    df["fwd_demand_qty"] = fwd.reindex(df.index).fillna(0)

    # ── Target / excess ──
    # An item listed twice with different MAX INVENTORY has no trustworthy target.
    mx = master["max_inventory"].groupby(level=0)
    conflicting = mx.nunique() > 1
    df["target_months"] = mx.first().where(~conflicting).reindex(df.index)
    df["target_conflict"] = conflicting.reindex(df.index).fillna(False).astype(bool)
    monthly_cons = df["avg_daily_consumption"] * _DAYS_PER_MONTH
    df["target_qty"] = df["target_months"] * monthly_cons
    has_target = df["target_qty"].notna()
    df["excess_qty"] = np.where(has_target, (df["current_qty"] - df["target_qty"]).clip(lower=0), np.nan)
    df["excess_value"] = df["excess_qty"] * df["unit_cost"]
    above = qty_hist.loc[df.index].gt(df["target_qty"].fillna(np.inf), axis=0)
    df["days_above_target"] = np.where(has_target, above.sum(axis=1), np.nan)
    df["months_above_target"] = df["days_above_target"] / _DAYS_PER_MONTH

    # ── STEP 3 – holding cost ──
    # Annual Carrying Cost = Average Inventory Value × annual rate.
    # Holding cost over the window = Σ daily value × rate / 365 = avg value × rate × days/365.
    df["annual_holding_cost"] = df["avg_value"] * rate if rate is not None else np.nan
    df["period_holding_cost"] = df["avg_value"] * rate * days / 365 if rate is not None else np.nan

    # ── STEP 4 – ranking ──  (rate is uniform, so avg_value gives the same order)
    df = df.sort_values("avg_value", ascending=False)
    total = df["avg_value"].sum()
    df["pct_of_total"] = df["avg_value"] / total if total > 0 else 0.0
    df["cum_pct"] = df["pct_of_total"].cumsum()
    df["rank"] = np.arange(1, len(df) + 1)

    # ── STEP 5 – ABC on holding cost ──
    # Class by cumulative share BEFORE the item, so the item that crosses the
    # 80% line still counts as A (it is part of what makes up the first 80%).
    prev_cum = df["cum_pct"] - df["pct_of_total"]
    df["abc"] = np.select([prev_cum < abc_t["A"], prev_cum < abc_t["B"]], ["A", "B"], "C")

    # ── STEP 6 – flags + reason ──
    valued = df["current_value"][df["current_value"] > 0]
    high_value_cut = valued.quantile(fl["high_value_percentile"]) if not valued.empty else np.inf
    high_value = df["current_value"] >= high_value_cut
    high_doh = df["days_on_hand"] >= fl["high_days_on_hand"]
    consumed = df["consumption_qty"] > 0
    slow = consumed & (df["turns"] < fl["slow_moving_turns"])
    no_recent = (df["current_qty"] > 0) & (
        ~consumed | (df["days_since_consumption"] >= fl["no_recent_consumption_days"]))
    obsolete = (df["current_qty"] > 0) & ~consumed & (df["fwd_demand_qty"] <= 0)

    # Reason / data-note texts are Hebrew (user-facing); flag codes stay English
    # because the UI filters on them.
    cur = config.get("currency", "")
    flags, reasons = [], []
    for it, r in df.iterrows():
        f, why = [], []
        if r["abc"] in fl["high_cost_classes"]:
            f.append("HIGH HOLDING COST")
            why.append(f"סיווג {r['abc']}: {r['pct_of_total']:.1%} מסך עלות ההחזקה "
                       f"(שווי מלאי ממוצע {r['avg_value']:,.0f} {cur})")
        if high_value[it] and high_doh[it]:
            f.append("HIGH VALUE + HIGH DAYS ON HAND")
            why.append(f"שווי נוכחי {r['current_value']:,.0f} {cur} (ב-{1 - fl['high_value_percentile']:.0%} "
                       f"העליונים) עם {_fmt_days(r['days_on_hand'])}")
        if high_value[it] and (not consumed[it] or slow[it]):
            f.append("HIGH VALUE + LOW CONSUMPTION")
            why.append(f"שווי גבוה, סבב מלאי {r['turns']:.2f} בשנה" if consumed[it]
                       else f"שווי גבוה, ללא צריכה ב-{days} הימים האחרונים")
        if high_value[it] and no_recent[it]:
            f.append("HIGH INVENTORY + NO RECENT CONSUMPTION")
            why.append(f"נצרך לאחרונה לפני {r['days_since_consumption']:.0f} ימים" if consumed[it]
                       else f"לא נצרך ב-{days} הימים האחרונים")
        if r["excess_value"] > 0:
            f.append("EXCESS ABOVE TARGET")
            why.append(f"{r['excess_qty']:,.1f} {r['uom'] or ''} מעל יעד מקסימום של {r['target_months']:g} חודשים "
                       f"({r['excess_value']:,.0f} {cur}), מעל היעד {r['days_above_target']:.0f} מתוך {days} ימים")
        if slow[it]:
            f.append("SLOW MOVING")
            why.append(f"סבב מלאי {r['turns']:.2f} בשנה, מתחת לסף {fl['slow_moving_turns']:g}")
        if obsolete[it]:
            f.append("POTENTIALLY OBSOLETE")
            why.append(f"ללא צריכה ב-{days} הימים האחרונים וגם ללא ביקוש עתידי ב-ASCP")
        flags.append(", ".join(f))
        reasons.append("; ".join(why))
    df["pattern_flags"] = flags
    df["reason"] = reasons

    notes = []
    for it, r in df.iterrows():
        n = []
        if r["target_conflict"]:
            n.append("שורות MAX INVENTORY סותרות ב-mrp_current.csv — עודף לא נבדק")
        elif pd.isna(r["target_months"]):
            n.append("אין MAX INVENTORY ב-mrp_current.csv — עודף לא נבדק")
        # Cost and list price usually agree within ~5%; a large gap usually
        # means they are quoted per different UOM (pack vs piece).
        ratio = r["unit_cost"] / r["list_price"] if r["list_price"] > 0 else np.nan
        if np.isfinite(ratio) and (ratio > 2 or ratio < 0.5):
            n.append(f"עלות הפריט {r['unit_cost']:,.2f} היא פי {ratio:.1f} ממחיר המחירון {r['list_price']:,.2f} "
                     f"— לבדוק יחידת מידה")
        if r["recon_negative_days"] > 0:
            n.append(f"הכמות המשוחזרת ירדה מתחת לאפס ב-{r['recon_negative_days']} ימים (תנועות חסרות?) — תוקנה ל-0")
        notes.append("; ".join(n))
    df["data_notes"] = notes

    # ── STEP 7 – Inventory Cost Score (0–100) ──
    comps = {
        "holding_cost": _pct_rank(df["avg_value"]),
        "days_on_hand": _pct_rank(df["days_on_hand"].where(df["current_qty"] > 0, 0)),
        "excess_inventory": _pct_rank(df["excess_value"].fillna(0)),
        "inventory_value": _pct_rank(df["current_value"]),
        # Low velocity = high score. No consumption ranks as slowest.
        "consumption_velocity": _pct_rank(-df["turns"].fillna(0), zero_is_zero=False)
                                .where(df["avg_value"] > 0, 0.0),
    }
    wsum = sum(float(weights.get(k, 0)) for k in comps)
    df["score"] = (sum(comps[k] * float(weights.get(k, 0)) for k in comps) / wsum * 100).round(1) \
        if wsum > 0 else np.nan

    # ── Daily value history (for charts) ──
    daily_value = qty_hist.loc[df.index].mul(df["unit_cost"], axis=0)

    slow_mask = slow | (df["current_qty"] > 0) & ~consumed
    kpis = {
        "total_inventory_value": df["current_value"].sum(),
        "avg_inventory_value": df["avg_value"].sum(),
        "annual_holding_cost": df["annual_holding_cost"].sum() if rate is not None else None,
        "excess_value": df["excess_value"].sum(),
        "slow_moving_value": df.loc[slow_mask, "current_value"].sum(),
        "high_cost_count": int(df["abc"].isin(fl["high_cost_classes"]).sum()),
        "n_items": len(df),
        "n_no_target": int(df["target_months"].isna().sum()),
        "n_excluded": len(excluded),
    }

    return {
        "ranking": df.reset_index(),
        "excluded": excluded.reset_index()[["item", "description", "current_qty", "reason"]],
        "daily_value": daily_value,
        "kpis": kpis,
        "rate": rate,
        "as_of": as_of,
        "days": days,
    }

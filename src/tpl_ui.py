"""
tpl_ui.py  –  Streamlit page for the Globus TPL storage-savings analysis.

Renders precomputed results from outputs/tpl/ (items.csv, consolidation.csv,
qa_sampling.csv, pallets_by_month.csv, meta.json, Globus_TPL_analysis.xlsx).
The ₪ figures and the Safety Stock recommendation are recomputed here from the
page's own settings (pallet rates, service level, realization share), so they
stay consistent with the Excel workbook's formulas.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tpl"
_BAR_COLOR = "#2a78d6"
_GRID = "rgba(128,128,128,0.18)"
_Z = {"95%": 1.645, "97.5%": 1.960, "98%": 2.054, "99%": 2.326}
_GROUPS = ["RM1", "RM2", "אריזה (PM)", "Min-Max", "אחר"]

# Hebrew text blocks are wrapped in st.container(key="tpl_rtl_…").
_RTL_CSS = """<style>
[class*="st-key-tpl_rtl"] [data-testid="stMarkdownContainer"],
[class*="st-key-tpl_rtl"] [data-testid="stCaptionContainer"],
[class*="st-key-tpl_rtl"] [data-testid="stAlertContentInfo"],
[class*="st-key-tpl_rtl"] [data-testid="stAlertContentWarning"] { direction: rtl; text-align: right; }
</style>"""


def _group(planner) -> str:
    p = str(planner or "")
    if p == "B_RM1":
        return "RM1"
    if p == "B_RM2":
        return "RM2"
    if p.endswith("MN-MX"):
        return "Min-Max"
    if p.startswith("B_PM"):
        return "אריזה (PM)"
    return "אחר"


@st.cache_data(ttl=600, show_spinner=False)
def _load(mtime: float) -> dict:
    rd = lambda f, **k: pd.read_csv(_DIR / f, encoding="utf-8-sig", **k)
    items = rd("items.csv", dtype={"item": str})
    items["group"] = items.planner.map(_group)
    for c in ("open_pos", "kind", "action", "ss_method"):
        items[c] = items[c].fillna("")
    return {
        "items": items,
        "cons": rd("consolidation.csv", dtype={"item": str, "sbatch": str, "lots": str}),
        "qa": rd("qa_sampling.csv", dtype={"item": str, "lot": str}),
        "qa_order": rd("qa_order.csv", dtype={"item": str}) if (_DIR / "qa_order.csv").exists() else pd.DataFrame(),
        "trend": rd("pallets_by_month.csv", dtype={"item": str}).set_index("item"),
        "meta": json.loads((_DIR / "meta.json").read_text(encoding="utf-8")),
    }


def _settings(meta: dict) -> dict:
    with st.expander("⚙️ הנחות החישוב (משפיעות על כל הסכומים בדף)", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        rc = c1.number_input("משטח מבוקר ₪/חודש", 0.0, 1000.0, float(meta["rate_ctrl"]), 1.0, key="tpl_rate_ctrl",
                             help="גיליון חיוב ספטמבר 2026, סעיף 10: ₪87 × 1.1631 (הצמדה)")
        ra = c2.number_input("משטח סביבתי ₪/חודש", 0.0, 1000.0, float(meta["rate_amb"]), 1.0, key="tpl_rate_amb",
                             help="גיליון חיוב ספטמבר 2026, סעיף 8: ₪56 × 1.1631")
        sl = c3.selectbox("רמת שירות ל-SS", list(_Z), index=2, key="tpl_sl")
        real = c4.slider("מימוש העודף בשנה", 0.0, 1.0, float(meta["realization"]), 0.05, key="tpl_real",
                         help="איזה חלק מהעודף ייעלם בפועל בשנה הקרובה (דחיית הזמנות). מלאי מת נספר תמיד ב-100%.")
        opt = c5.toggle("קיבולת היסטורית לאיחוד", value=False, key="tpl_opt",
                        help="כבוי = שמרני (אריזת הספק באצוות 12 החודשים האחרונים). דלוק = המקסימום שנצפה אי-פעם – דורש אישור גלובוס.")
        step = c6.slider("הורדת SS מקסימלית בשלב 1", 0.0, 1.0, float(meta.get("ss_step", 0.3)), 0.05, key="tpl_ss_step",
                         help="הורדה הדרגתית: בשלב הראשון לכל היותר אחוז זה מה-SS הנוכחי. היעד (King) – רק אחרי בדיקת צריכה ואספקות בפועל. 100% = ישר ליעד.")
    return {"rc": rc, "ra": ra, "z": _Z[sl], "sl": sl, "real": real, "opt": opt, "step": step}


def _compute(items: pd.DataFrame, s: dict) -> pd.DataFrame:
    df = items.copy()
    df["rate"] = np.where(df.cond == "מבוקר", s["rc"], s["ra"])
    df["cons"] = df.cons_o if s["opt"] else df.cons_c
    # Safety Stock: King for regular demand, current SS kept for intermittent demand
    king = s["z"] * np.sqrt(df.LT_m * df.cv ** 2 + df.sigma_lt ** 2)
    rec = np.ceil(np.clip(king, 1, None) * 2) / 2
    df["ss_target"] = np.where(df.ss_method == "King 98%", rec, df.SS_m)
    # staged: the first step cuts at most `step` of today's SS (rounded up to 0.5 month)
    floor = np.ceil(df.SS_m * (1 - s["step"]) * 2) / 2
    df["ss_rec"] = np.where(df.ss_target < df.SS_m, np.maximum(df.ss_target, floor), df.ss_target)
    delta = (df.SS_m - df.ss_rec).clip(lower=0)
    ok = (df.d > 0) & (df.oh_total > 0) & df.SS_m.notna()
    df["ss_pal_x"] = np.where(ok, np.minimum(delta * df.d * df.pal_now / df.oh_total, df.pal_now), 0.0)
    df["ss_pal_x"] = df.ss_pal_x.fillna(0)
    df["excess_pal"] = df.excess_pal.fillna(0)
    df["mon_cons"] = df.cons * df.rate
    df["mon_ss"] = df.ss_pal_x * df.rate
    df["mon_excess"] = df.excess_pal * df.rate * np.where(df.kind == "מת", 1.0, s["real"])
    df["mon_overlap"] = np.minimum(df.ss_pal_x, df.excess_pal) * df.rate
    df["mon_total"] = df.mon_cons + df.mon_ss + df.mon_excess - df.mon_overlap
    df["annual_total"] = df.mon_total * 12
    return df


def _kpis(df: pd.DataFrame, meta: dict):
    structural = (df.mon_cons.sum() + df.mon_ss.sum()) * 12
    excess = df.mon_excess.sum() * 12
    overlap = df.mon_overlap.sum() * 12
    k = st.columns(5)
    k[0].metric("משטחים מחויבים היום", f"{df.pal_now.sum():,.0f}")
    k[1].metric("עלות אחסון חודשית", f"₪{(df.pal_now * df.rate).sum():,.0f}")
    k[2].metric("חיסכון מבני שנתי", f"₪{structural:,.0f}", "איחוד + SS", delta_color="off")
    k[3].metric("חיסכון מעודף / מלאי מת", f"₪{excess - overlap:,.0f}", "בניכוי חפיפה עם SS", delta_color="off")
    k[4].metric("סה\"כ פוטנציאל שנתי", f"₪{structural + excess - overlap:,.0f}",
                f"קובץ קודם: ₪{meta['prev_file']['annual']:,.0f}", delta_color="off")
    k2 = st.columns(5)
    k2[0].metric("איחוד – משטחים", f"{df.cons.sum():,.0f}")
    k2[1].metric("הורדת SS – משטחים", f"{df.ss_pal_x.sum():,.1f}")
    k2[2].metric("מלאי מעל היעד – משטחים", f"{df.excess_pal.sum():,.0f}")
    k2[3].metric("מתוכו מלאי מת", f"{df.loc[df.kind == 'מת', 'excess_pal'].sum():,.0f}")
    k2[4].metric("מתוכו סיכון תפוגה", f"{df.loc[df.kind == 'סיכון תפוגה', 'excess_pal'].sum():,.0f}")


def _base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


_NF = st.column_config.NumberColumn
_ILS = _NF(format="₪%,.0f")


def _focus(df: pd.DataFrame):
    f = df[df.focus_rank.notna()].sort_values("mon_total", ascending=False)
    with st.container(key="tpl_rtl_focus"):
        st.caption("הפריטים עם הפוטנציאל הגבוה ביותר, ובנוסף 8 הפריטים מהקובץ הקודם (⭐). "
                   "חיסכון חודשי = (איחוד + הורדת SS + עודף × מימוש − חפיפה) × מחיר משטח.")
    show = f.assign(star=np.where(f.orig8, "⭐", ""))[
        ["star", "item", "desc", "group", "cond", "pal_now", "cons", "ss_pal_x", "excess_pal", "kind", "mon_total", "annual_total",
         "cover_now_m", "SS_m", "ss_rec", "LT_m", "SL_m", "open_pos", "action"]]
    st.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "star": st.column_config.TextColumn("", width=30), "item": "מק״ט", "desc": st.column_config.TextColumn("תיאור", width="medium"),
        "group": "קבוצה", "cond": "אחסון", "pal_now": _NF("משטחים", format="%.1f"), "cons": _NF("איחוד", format="%.0f"),
        "ss_pal_x": _NF("SS – משטחים", format="%.1f"), "excess_pal": _NF("עודף – משטחים", format="%.1f"), "kind": "מת / תפוגה",
        "mon_total": _NF("₪ לחודש", format="₪%,.0f"), "annual_total": _NF("₪ לשנה", format="₪%,.0f"),
        "cover_now_m": _NF("חודשי כיסוי", format="%.1f"), "SS_m": _NF("SS נוכחי", format="%.1f"), "ss_rec": _NF("SS שלב 1", format="%.1f"),
        "LT_m": _NF("LT", format="%.1f"), "SL_m": _NF("חיי מדף", format="%.0f"), "open_pos": st.column_config.TextColumn("PO פתוחים", width="medium"),
        "action": st.column_config.TextColumn("פעולה מומלצת", width="large"),
    })


def _item_detail(df: pd.DataFrame, trend: pd.DataFrame):
    import plotly.graph_objects as go
    opts = df.sort_values("mon_total", ascending=False)
    labels = {r.item: f"{r.item} — {r.desc}" for r in opts.itertuples()}
    it = st.selectbox("בחירת פריט", list(labels), format_func=labels.get, key="tpl_item")
    r = df.set_index("item").loc[it]
    c = st.columns(6)
    c[0].metric("משטחים היום", f"{r.pal_now:,.1f}")
    c[1].metric("ממוצע 11 חודשים", f"{r.pal_avg11:,.1f}")
    c[2].metric("חודשי כיסוי", "—" if pd.isna(r.cover_now_m) else f"{r.cover_now_m:,.1f}")
    c[3].metric("קרטונים למשטח (שמרני / היסטורי)", f"{r.cap_cons:g} / {r.cap_hist:g}")
    c[4].metric("יחידות Oracle לקרטון", f"{r.factor:g}")
    c[5].metric("₪ חיסכון חודשי", f"₪{r.mon_total:,.0f}")
    left, right = st.columns([3, 2])
    with left:
        if it in trend.index:
            t = trend.loc[it]
            fig = go.Figure(go.Bar(x=[m[5:] + "/" + m[2:4] for m in t.index], y=t.values, marker_color=_BAR_COLOR,
                                   hovertemplate="%{x}: %{y:.1f} משטחים<extra></extra>"))
            fig.update_layout(title="משטחים בגלובוס לפי חודש", yaxis_title="משטחים")
            st.plotly_chart(_base_layout(fig, 300), width="stretch")
    with right, st.container(key="tpl_rtl_item"):
        st.markdown(f"**קבוצה:** {r.group} ({r.planner}) · **אחסון:** {r.cond}")
        st.markdown(f"**צריכה חודשית (12 ח'):** {r.d12:,.1f} {r.uom} · **ביקוש ASCP:** {r.d_fwd:,.1f} {r.uom}")
        st.markdown(f"**LT:** {r.LT_m:g} ח' · **חיי מדף:** {r.SL_m:,.0f} ח' · **SS:** {r.SS_m if pd.notna(r.SS_m) else '—'} → "
                    f"**{r.ss_rec if pd.notna(r.ss_rec) else '—'}** ח' בשלב 1 ({r.ss_method or '—'})"
                    + (f" · יעד {r.ss_target:g} ח' אחרי בדיקה" if pd.notna(r.ss_target) and r.ss_target < r.ss_rec else ""))
        if r.open_pos:
            st.markdown(f"**PO פתוחים:** {r.open_pos}")
        st.markdown("**פעולה מומלצת:**")
        for a in str(r.action).split("; "):
            st.markdown(f"- {a}")


def _consolidation(cons: pd.DataFrame, df: pd.DataFrame, s: dict):
    c = cons.copy()
    rate = c.item.map(df.set_index("item").rate)
    c["target"] = np.ceil(c.qty / c.cap_cons)
    c["reduce_c"] = (c.pallets - c.target).clip(lower=0)
    c["reduce_h"] = (c.pallets - np.ceil(c.qty / c.cap_hist)).clip(lower=0)
    c["mon"] = (c.reduce_h if s["opt"] else c.reduce_c) * rate
    with st.container(key="tpl_rtl_cons"):
        st.caption("איחוד רק בתוך אותה אצוות ספק (Batch Supplier No) ורק במשטחים חד-פריטיים – אין ערבוב אצוות יצרן. "
                   "מארזים לכל משטח = הכמות בכל SSCC היום.")
    st.dataframe(c[["item", "desc", "sbatch", "lots", "src", "pallets", "qty", "qty_per_pallet", "cap_cons", "cap_hist", "reduce_c", "reduce_h", "mon"]],
                 hide_index=True, width="stretch", height=500, column_config={
        "item": "מק״ט", "desc": st.column_config.TextColumn("תיאור", width="medium"), "sbatch": "אצוות ספק", "lots": "אצווה פנימית",
        "src": "מקור מיפוי", "pallets": "משטחים", "qty": "מארזים", "qty_per_pallet": "מארזים לכל משטח",
        "cap_cons": "קיבולת שמרנית", "cap_hist": "קיבולת היסטורית", "reduce_c": "צמצום שמרני", "reduce_h": "צמצום היסטורי",
        "mon": _NF("₪ לחודש", format="₪%,.0f")})


def _ss(df: pd.DataFrame, s: dict):
    x = df[(df.d > 0) & df.SS_m.notna()].sort_values("mon_ss", ascending=False)
    with st.container(key="tpl_rtl_ss"):
        st.caption(f"נוסחת King ברמת שירות {s['sl']}: SS = z·√(LT·σd² + d²·σLT²), בחודשי כיסוי. "
                   "ביקוש לא סדיר (CV>1.5 או ≥12 חודשים ללא צריכה) – נשאר ה-SS הנוכחי. "
                   f"הורדה הדרגתית: SS מומלץ = שלב 1 (עד {s['step']:.0%} מה-SS הנוכחי); SS יעד – אחרי בדיקת צריכה ואספקות בפועל. "
                   "⚠ = SS + מנת הזמנה חורגים מחיי המדף בקבלה פחות 3 חודשים.")
    x = x.assign(sl_flag=np.where(x.ss_vs_sl.astype(str) == "True", "⚠", ""))
    st.dataframe(x[["item", "desc", "group", "LT_m", "SL_m", "sl_at_receipt_m", "d12", "d_fwd", "cv", "sigma_lt", "ss_method", "SS_m", "ss_rec",
                    "ss_target", "order_cover_m", "sl_flag", "ss_pal_x", "mon_ss"]], hide_index=True, width="stretch", height=520, column_config={
        "item": "מק״ט", "desc": st.column_config.TextColumn("תיאור", width="medium"), "group": "קבוצה",
        "LT_m": _NF("LT", format="%.1f"), "SL_m": _NF("חיי מדף", format="%.0f"), "sl_at_receipt_m": _NF("חיי מדף בקבלה", format="%.1f"),
        "d12": _NF("צריכה/ח' (12)", format="%,.1f"), "d_fwd": _NF("ביקוש ASCP/ח'", format="%,.1f"), "cv": _NF("CV", format="%.2f"),
        "sigma_lt": _NF("σLT", format="%.2f"), "ss_method": "שיטה", "SS_m": _NF("SS נוכחי", format="%.1f"), "ss_rec": _NF("SS שלב 1", format="%.1f"),
        "ss_target": _NF("SS יעד", format="%.1f"), "order_cover_m": _NF("מנת הזמנה (ח')", format="%.1f"), "sl_flag": "חיי מדף", "ss_pal_x": _NF("משטחים", format="%.1f"),
        "mon_ss": _NF("₪ לחודש", format="₪%,.0f")})


def _excess(df: pd.DataFrame):
    x = df[(df.excess_pal >= 0.5) | (df.kind != "")].sort_values("mon_excess", ascending=False)
    with st.container(key="tpl_rtl_excess"):
        st.caption("יעד מקסימלי = SS מומלץ × ביקוש + מנת הזמנה (לפריט ללא אצווה – עד 3 חודשים). "
                   "מת = ASCP מתכנן פחות מ-20% מהצריכה ההיסטורית. סיכון תפוגה = יש ביקוש, אך יותר מחצי מהמלאי צפוי לפוג לפני שימוש.")
    st.dataframe(x[["item", "desc", "group", "kind", "oh_total", "max_level", "excess_qty", "cover_now_m", "months_to_burn", "exp_risk_qty",
                    "pal_now", "excess_pal", "mon_excess", "value_tpl", "storage_to_value", "open_pos", "planned_12m", "action"]],
                 hide_index=True, width="stretch", height=520, column_config={
        "item": "מק״ט", "desc": st.column_config.TextColumn("תיאור", width="medium"), "group": "קבוצה", "kind": "מת / תפוגה",
        "oh_total": _NF("מלאי כולל", format="%,.0f"), "max_level": _NF("יעד מקסימלי", format="%,.0f"), "excess_qty": _NF("עודף", format="%,.0f"),
        "cover_now_m": _NF("חודשי כיסוי", format="%.1f"), "months_to_burn": _NF("חודשים לשחיקה", format="%.1f"),
        "exp_risk_qty": _NF("צפי פגות", format="%,.0f"), "pal_now": _NF("משטחים", format="%.1f"), "excess_pal": _NF("משטחי עודף", format="%.1f"),
        "mon_excess": _NF("₪ לחודש (×מימוש)", format="₪%,.0f"), "value_tpl": _NF("שווי ב-TPL", format="₪%,.0f"),
        "storage_to_value": _NF("אחסון שנתי ÷ שווי", format="percent"), "open_pos": st.column_config.TextColumn("PO פתוחים", width="medium"),
        "planned_12m": _NF("הזמנות מתוכננות", format="%,.0f"), "action": st.column_config.TextColumn("פעולה מומלצת", width="large")})


def _trend(df: pd.DataFrame, trend: pd.DataFrame):
    import plotly.graph_objects as go
    tot = trend.reindex(df.item).sum()
    fig = go.Figure(go.Bar(x=[m[5:] + "/" + m[2:4] for m in tot.index], y=tot.values, marker_color=_BAR_COLOR,
                           hovertemplate="%{x}: %{y:,.0f} משטחים<extra></extra>"))
    fig.update_layout(title="סה\"כ משטחים בגלובוס (פריטים בסינון)", yaxis_title="משטחים")
    st.plotly_chart(_base_layout(fig, 320), width="stretch")
    t = trend.reindex(df.item).dropna(how="all")
    t.columns = [m[5:] + "/" + m[2:4] for m in t.columns]
    t = t.join(df.set_index("item")[["desc", "excess_avg11_c", "fill_now"]])
    t = t.sort_values(t.columns[-4], ascending=False)
    st.dataframe(t, width="stretch", height=420, column_config={
        "desc": st.column_config.TextColumn("תיאור", width="medium"),
        "excess_avg11_c": _NF("משטחים חלקיים עודפים (ממוצע)", format="%.1f"), "fill_now": _NF("מילוי היום", format="percent"),
        **{c: _NF(c, format="%.1f") for c in t.columns[:-3]}})


def _samples(n):
    """QA rule: floor(sqrt N) + 1 packs from every new lot."""
    return np.floor(np.sqrt(n)) + 1


def _order_for_full(x):
    """Smallest lot (packs) that leaves exactly x packs after QA sampling."""
    return x + np.floor(np.sqrt(x + np.floor(np.sqrt(x)) + 1)) + 1


def _qa_order(qo: pd.DataFrame):
    st.markdown("#### כמות הזמנה מיושרת למשטחים מלאים אחרי הדגימה")
    with st.container(key="tpl_rtl_qa_order"):
        st.markdown("להזמין **k משטחים מלאים + ⌊√N⌋+1 מארזי דגימה**, ולבקש מהספק לשלוח את מארזי הדגימה **ביחידה נפרדת** "
                    "(עדיף ישירות ל-QC ולא לגלובוס – כל SSCC שנקלט בגלובוס מחויב חודש מלא). QC מחשבת √N+1 על כל ההזמנה, "
                    "והכמות המומלצת נותנת בדיוק את מארזי היחידה הנפרדת – המשטחים הראשיים נשארים מלאים, "
                    "ואין משטח דליל שאי אפשר להשלים מאצוות יצרן אחרת. הכמות שנדגמת ≈ (√(Q/u)+1)·u וגדלה עם גודל המארז u – "
                    "כשמורידים SS והאצוות קטנות, הדגימה תופסת חלק גדול יותר מהאצווה. "
                    "**N** וקיבולת = חציון האצוות 02–11/2026 והקיבולת השמרנית; יחידה = מארז/קרטון גלובוס.")
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("מחשבון: מארזים באצווה (N)", 1, 100000, 100, key="tpl_qa_n")
    cap = c2.number_input("מארזים למשטח", 1, 10000, 20, key="tpl_qa_cap")
    s_, k = int(_samples(n)), int(np.ceil((n - _samples(n)) / cap))
    up = int(_order_for_full(k * cap))
    down = int(_order_for_full((k - 1) * cap)) if k > 1 else None
    c3.metric("מארזי דגימה", s_, f"{s_ / n:.0%} מהאצווה", delta_color="off")
    with st.container(key="tpl_rtl_qa_calc"):
        st.markdown(f"אחרי הדגימה: **{n - s_}** מארזים על **{k}** משטחים (מילוי משטח אחרון {((n - s_) % cap or cap) / cap:.0%}). "
                    f"להזמנת משטחים מלאים: **{up}** = {k} × {cap} + {up - k * cap} מארזי דגימה בנפרד"
                    + (f", או **{down}** = {k - 1} × {cap} + {down - (k - 1) * cap} מארזי דגימה בנפרד." if down else "."))
    if qo.empty:
        return
    t = qo.copy()
    t["s"] = _samples(t.N)
    t["share"] = t.s / t.N
    t["pal"] = np.ceil((t.N - t.s) / t.cap)
    rem = (t.N - t.s) % t.cap
    t["fill"] = np.where(rem == 0, 1.0, rem / t.cap)
    t["down"] = np.where(t.pal > 1, _order_for_full((t.pal - 1) * t.cap), np.nan)
    t["up"] = _order_for_full(t.pal * t.cap)
    advice = np.where(t.share > 0.25, "דגימה >25% – לבקש מארז קטן יותר / מארזי דגימה נפרדים. ", "")
    order = np.where(t.fill >= 1, "מיושר – אין שינוי",
                     np.where(t.pal <= 1, "אצווה במשטח אחד – יישור לא רלוונטי",
                              np.where(t.fill < 0.5, "להזמין " + t.down.fillna(0).astype(int).astype(str) + " (למטה)",
                                       "להזמין " + t.up.astype(int).astype(str) + " (למעלה)")))
    t["rec"] = advice + order
    use_down = (t.pal > 1) & (t.fill < 0.5)
    k_ = np.where(use_down, t.pal - 1, t.pal)
    tot = np.where(use_down, t.down, t.up)
    t["ship"] = np.where(t.pal > 1, [f"{a:.0f} משטחים × {c:.0f} + {b - a * c:.0f} מארזי דגימה בנפרד (סה״כ {b:.0f})"
                                     for a, b, c in zip(k_, np.nan_to_num(tot), t.cap)], "")
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "item": "מק״ט", "desc": st.column_config.TextColumn("תיאור", width="medium"), "cond": "תנאי", "lots": "אצוות",
        "N": _NF("N לאצווה", format="%,.0f"), "cap": _NF("קיבולת", format="%,.0f"), "s": _NF("דגימה", format="%.0f"),
        "share": _NF("% נדגם", format="percent"), "pal": _NF("משטחים", format="%.0f"), "fill": _NF("מילוי אחרון", format="percent"),
        "down": _NF("הזמנה – למטה", format="%,.0f"), "up": _NF("הזמנה – למעלה", format="%,.0f"),
        "rec": st.column_config.TextColumn("המלצה", width="large"),
        "ship": st.column_config.TextColumn("הוראת משלוח לספק", width="large")})


def _qa(qa: pd.DataFrame, qo: pd.DataFrame):
    with st.container(key="tpl_rtl_qa"):
        st.markdown("כלל QA: מכל אצווה חדשה נלקחים **⌊√N⌋+1** מארזים. כמות שנקלטה ב-Oracle (N) מול הכמות שהופיעה בגלובוס בחודש הראשון: "
                    "כשיש פער, הוא שווה בדיוק ל-√N+1. הדגימה לא מוסיפה משטחים, אבל משאירה את המשטח האחרון של האצווה דליל.")
    if qa.empty:
        st.info("אין אצוות חדשות עם נתוני קליטה לפריטים בסינון.")
    else:
        c = st.columns(len(qa.match.unique()))
        for col, (k, v) in zip(c, qa.match.value_counts().items()):
            col.metric(k, int(v))
        st.dataframe(qa, hide_index=True, width="stretch", column_config={
            "item": "מק״ט", "lot": "אצווה", "month": "חודש הופעה", "N": _NF("N שנקלט", format="%,.0f"), "Q": _NF("בגלובוס", format="%,.0f"),
            "gap": _NF("פער", format="%,.0f"), "sqrtN": _NF("√N+1", format="%.0f"), "match": "סיווג", "P": "משטחים", "cap": "קיבולת"})
    _qa_order(qo)


def _methodology(meta: dict):
    with st.expander("ℹ️ מתודולוגיה, מקורות ופערי נתונים"), st.container(key="tpl_rtl_method"):
        st.markdown(f"""
- **מלאי גלובוס:** 11 דוחות "BTG Stock report" (01/2026–11/2026), שורה = SSCC = משטח. משטח משותף לכמה מק״טים מחולק ביניהם.
  מיקומי NH (מוצר מוגמר, בניין אחר) לא נכללים.
- **חיוב:** כל SSCC שבמלאי + כל קליטה בחודש מחויבים חודש מלא (ספטמבר: {meta['billing_sep']['ambient_pallets']} סביבתי +
  {meta['billing_sep']['controlled_pallets']} מבוקר).
- **יחידות:** גלובוס סופרת קרטונים, Oracle סופרת יחידות בסיס (למשל Ster.Bag: קרטון = 5 יחידות). מקדם ההמרה לכל פריט =
  חציון היחס בין מלאי ESMTPL החודשי ב-Oracle לבין מלאי גלובוס.
- **אצוות ספק:** SUPPLIER_LOT_NUMBER מ-Oracle, ובגיבוי Batch Supplier No מגליון מלאי הפתיחה. {meta['mapped_supplier_batch_share']:.0%} מהשורות מופו;
  השאר טופל כאצווה נפרדת (שמרני).
- **קיבולת משטח:** שמרנית = המשטח המלא ביותר באצוות שהתקבלו ב-12 החודשים האחרונים; היסטורית = המקסימום שנצפה. לא קיימים בגלובוס נתוני נפח/משקל –
  יש לאשר מולם לפני איחוד.
- **צריכה:** WIP Issue + Account alias / Move Order Issue ללא סיבה, בניכוי WIP Return (11/2024–10/2026) – כמו בדף ה-MRP.
  **ביקוש:** ASCP, 12 חודשים קדימה; פריטים ש-ASCP לא מתכנן (Min-Max) – לפי היסטוריה.
- **σLT:** סטיית התקן של איחור האספקה מול התאריך המובטח ב-PO.
- **חפיפה:** בפריטים עם עודף, הורדת ה-SS כבר כלולה בעודף – מנוכה מהסכום כדי לא לספור פעמיים.
- **לאימות:** קיבולת פיזית מול גלובוס · מלאי מת – אישור QA/כספים ובדיקת שימוש חלופי · פריטי PM ו-Min-Max – אישור המתכנן האחראי.

נתונים נכונים ל-{meta['snapshot']} · עודכן {meta['built']}.
""")


def render_tpl():
    st.markdown(_RTL_CSS, unsafe_allow_html=True)
    if not (_DIR / "items.csv").exists():
        st.error(f"לא נמצאו נתוני הניתוח בתיקייה {_DIR}")
        return
    data = _load((_DIR / "items.csv").stat().st_mtime)
    meta = data["meta"]
    s = _settings(meta)

    c1, c2, c3 = st.columns([3, 2, 2])
    groups = c1.pills("קבוצת פריטים", _GROUPS, selection_mode="multi", default=_GROUPS, key="tpl_groups",
                      help="RM1 = B_RM1, RM2 = B_RM2, אריזה = B_PM*, Min-Max = מתכנני *_MN-MX, אחר = כל השאר")
    kinds = c2.pills("סינון מצב", ["מת", "סיכון תפוגה"], selection_mode="multi", key="tpl_kinds")
    search = c3.text_input("חיפוש מק״ט / תיאור", key="tpl_search")
    if not groups:
        st.info("יש לבחור לפחות קבוצת פריטים אחת.")
        return

    df = _compute(data["items"], s)
    df = df[df.group.isin(groups)]
    if kinds:
        df = df[df.kind.isin(kinds)]
    if search:
        q = search.strip().lower()
        df = df[df.item.str.contains(q, regex=False) | df.desc.str.lower().str.contains(q, regex=False)]
    if df.empty:
        st.info("אין פריטים בסינון הנוכחי.")
        return

    _kpis(df, meta)
    with st.container(key="tpl_rtl_asof"):
        st.caption(f"מלאי גלובוס נכון ל-{meta['snapshot']} · {len(df)} פריטים בסינון · הסכומים מחושבים לפי ההנחות למעלה")

    tabs = st.tabs(["🎯 פריטי מיקוד", "🔍 פריט", "📦 איחוד לפי אצוות ספק", "🛡️ Safety Stock", "⚠️ עודף ומלאי מת",
                    "📈 מגמת משטחים", "🧪 דגימות QA"])
    with tabs[0]:
        _focus(df)
    with tabs[1]:
        _item_detail(df, data["trend"])
    with tabs[2]:
        _consolidation(data["cons"][data["cons"].item.isin(df.item)], df, s)
    with tabs[3]:
        _ss(df, s)
    with tabs[4]:
        _excess(df)
    with tabs[5]:
        _trend(df, data["trend"])
    with tabs[6]:
        qo = data["qa_order"]
        _qa(data["qa"][data["qa"].item.isin(df.item)], qo[qo.item.isin(df.item)] if not qo.empty else qo)

    xl = _DIR / "Globus_TPL_analysis.xlsx"
    if xl.exists():
        st.download_button("📥 הורדת חוברת הניתוח המלאה (Excel)", data=xl.read_bytes(), file_name="Globus_TPL_analysis.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="tpl_dl")
    _methodology(meta)

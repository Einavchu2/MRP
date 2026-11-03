"""
holding_cost_ui.py  –  Streamlit page for the Inventory Holding Cost analysis.
All calculations live in src/holding_cost.py; this only renders.
"""
import copy
import io

import numpy as np
import pandas as pd
import streamlit as st

from src.db import load_inventory_history
from src.holding_cost import FLAG_ORDER, PLANNER_GROUPS, compute_holding_cost, filter_history, load_config
from src.master_data import load_master

# Reference categorical palette, first three slots (validated all-pairs for
# scatter use). Colour follows the ABC class, never the rank.
_ABC_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
_BAR_COLOR = "#2a78d6"
_GRID = "rgba(128,128,128,0.18)"

# Hebrew meaning of each flag code (codes stay English — the table filters on them).
_FLAG_HE = {
    "HIGH HOLDING COST": "עלות החזקה גבוהה — בין הפריטים שמרכיבים את רוב עלות ההחזקה (סיווג A)",
    "HIGH VALUE + HIGH DAYS ON HAND": "שווי גבוה + ימי מלאי רבים",
    "HIGH VALUE + LOW CONSUMPTION": "שווי גבוה + צריכה נמוכה",
    "HIGH INVENTORY + NO RECENT CONSUMPTION": "מלאי גבוה + אין צריכה לאחרונה",
    "EXCESS ABOVE TARGET": "עודף מעל יעד MAX INVENTORY",
    "SLOW MOVING": "תנועה איטית — סבב מלאי נמוך מהסף",
    "POTENTIALLY OBSOLETE": "חשד למלאי מת — אין צריכה ואין ביקוש עתידי",
}

# Hebrew explanation blocks are wrapped in st.container(key="hc_rtl_…").
_RTL_CSS = """<style>
[class*="st-key-hc_rtl"] [data-testid="stMarkdownContainer"],
[class*="st-key-hc_rtl"] [data-testid="stCaptionContainer"] { direction: rtl; text-align: right; }
</style>"""


def _money(v, cur):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.0f} {cur}"


def _base_layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(namelength=-1), legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


def _settings(config: dict) -> tuple:
    """Rate + overrides (session-only). Returns (config_copy, rate)."""
    cfg = copy.deepcopy(config)
    cfg_rate = config.get("carrying_cost_rate_annual")

    c1, c2 = st.columns([1, 3])
    with c1:
        rate_pct = st.number_input(
            "Annual carrying cost rate (%)",
            min_value=0.0, max_value=100.0, step=0.5,
            value=None if cfg_rate is None else float(cfg_rate) * 100,
            placeholder="לא הוגדר — מהכספים",
            help="העלות השנתית של החזקת 1 ₪ מלאי: עלות הון + אחסון + ביטוח + הפרשה להתיישנות. "
                 "יש לקבל מהכספים (בקרה / גזברות). להגדרה קבועה: holding_cost_config.json → "
                 "carrying_cost_rate_annual.",
            key="hc_rate_pct",
        )
    rate = None if rate_pct is None else rate_pct / 100
    with c2, st.container(key="hc_rtl_rate"):
        if rate is None:
            st.warning(
                "**לא הוגדר שיעור עלות החזקה.** אין שיעור כזה ב-DWH או באפליקציה, ולכן לא מונח שום ערך. "
                "הדירוג, הפארטו, סיווג ה-ABC והציון עדיין תקפים — הם מחושבים לפי שווי המלאי הממוצע, "
                "שנותן בדיוק אותו סדר לכל שיעור אחיד — אבל סכומי עלות ההחזקה יישארו ריקים עד שיוזן שיעור. "
                "יש לקבל אותו מהכספים."
            )
        elif cfg_rate is None or abs(rate - cfg_rate) > 1e-12:
            st.info(f"משתמש ב-{rate_pct:g}% לסשן הנוכחי בלבד. כדי לקבע אותו כערך המאושר, יש להגדיר "
                    f"`carrying_cost_rate_annual: {rate:g}` בקובץ `holding_cost_config.json`.")

    with st.expander("⚙️ ספים ומשקלי ציון (שינוי לסשן הנוכחי בלבד — ברירות המחדל ב-holding_cost_config.json)"):
        with st.container(key="hc_rtl_weights_title"):
            st.markdown(f"**משקלי הציון — סטטוס: {config.get('score_weights_status', 'unknown')}**")
        wcols = st.columns(5)
        labels = {
            "holding_cost": "Holding cost", "days_on_hand": "Days on hand",
            "excess_inventory": "Excess inventory", "consumption_velocity": "Low consumption velocity",
            "inventory_value": "Current inventory value",
        }
        for col, (k, lbl) in zip(wcols, labels.items()):
            cfg["score_weights"][k] = col.number_input(
                lbl, min_value=0.0, max_value=100.0, step=5.0,
                value=float(config["score_weights"].get(k, 0)), key=f"hc_w_{k}")
        with st.container(key="hc_rtl_weights_note"):
            st.caption("עלות החזקה ושווי מלאי ממוצע נבדלים רק בשיעור האחיד, לכן המשקל של 'Holding cost' "
                       "כבר מכסה את השווי המשוקלל לאורך זמן; 'Current inventory value' מוסיף מעליו את "
                       "החשיפה של היום.")

        t1, t2, t3, t4, t5 = st.columns(5)
        cfg["abc_thresholds"]["A"] = t1.number_input(
            "A class up to cum. %", 50.0, 99.0, config["abc_thresholds"]["A"] * 100, 1.0, key="hc_abc_a") / 100
        cfg["abc_thresholds"]["B"] = t2.number_input(
            "B class up to cum. %", 51.0, 99.9, config["abc_thresholds"]["B"] * 100, 1.0, key="hc_abc_b") / 100
        cfg["flags"]["high_days_on_hand"] = t3.number_input(
            "High days on hand ≥", 30, 3650, int(config["flags"]["high_days_on_hand"]), 30, key="hc_doh")
        cfg["flags"]["slow_moving_turns"] = t4.number_input(
            "Slow moving: turns/yr <", 0.1, 12.0, float(config["flags"]["slow_moving_turns"]), 0.1, key="hc_turns")
        cfg["flags"]["no_recent_consumption_days"] = t5.number_input(
            "No recent consumption ≥ days", 14, 365, int(config["flags"]["no_recent_consumption_days"]), 7,
            key="hc_norecent")
        if cfg["abc_thresholds"]["B"] <= cfg["abc_thresholds"]["A"]:
            st.error("סף B חייב להיות גבוה מסף A — משתמש בברירות המחדל מקובץ ההגדרות.")
            cfg["abc_thresholds"] = copy.deepcopy(config["abc_thresholds"])
    return cfg, rate


def _kpis(res, cur):
    k = res["kpis"]
    c = st.columns(5)
    c[0].metric("Total Inventory Value", _money(k["total_inventory_value"], cur),
                help=f"מלאי נוכחי × עלות פריט ב-Oracle, עבור {k['n_items']} פריטים מתומחרים.")
    c[1].metric("Est. Annual Holding Cost", _money(k["annual_holding_cost"], cur),
                help=f"סכום שווי המלאי הממוצע ({res['days']} הימים האחרונים) × שיעור עלות ההחזקה השנתי."
                     if res["rate"] is not None else "יש להזין שיעור עלות החזקה כדי לחשב.")
    c[2].metric("Excess Inventory Value", _money(k["excess_value"], cur),
                help="סכום של max(מלאי נוכחי − יעד MAX INVENTORY, 0) × עלות. "
                     f"{k['n_no_target']} פריטים ללא יעד לא נכללים.")
    c[3].metric("Slow Moving Inventory Value", _money(k["slow_moving_value"], cur),
                help="השווי הנוכחי של פריטים שסבב המלאי שלהם מתחת לסף התנועה האיטית, "
                     "או שלא נצרכו כלל בתקופה.")
    c[4].metric("High-Cost Materials", f"{k['high_cost_count']}",
                help="פריטים בסיווג ABC של עלות גבוהה — המעטים שמרכיבים את רוב עלות ההחזקה.")


def _pareto(rk, basis_col, basis_lbl, cfg, max_items):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = rk.head(max_items)
    # Two panels sharing the material axis instead of a dual y-axis chart.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
    for cls, color in _ABC_COLORS.items():
        s = d[d["abc"] == cls]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s["item"], y=s[basis_col], name=f"Class {cls}", marker_color=color,
            customdata=np.stack([s["description"], s["pct_of_total"] * 100, s["cum_pct"] * 100], axis=-1),
            hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>" + basis_lbl +
                          ": %{y:,.0f}<br>Share: %{customdata[1]:.1f}%<br>Cumulative: %{customdata[2]:.1f}%"
                          "<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d["item"], y=d["cum_pct"] * 100, mode="lines", name="Cumulative %",
        line=dict(color="#52514e", width=2), showlegend=False,
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    for t, lbl in [(cfg["abc_thresholds"]["A"], "A"), (cfg["abc_thresholds"]["B"], "B")]:
        fig.add_hline(y=t * 100, line=dict(color="#8a8984", width=1, dash="dot"), row=2, col=1,
                      annotation_text=f"{lbl} ≤ {t:.0%}", annotation_position="top left",
                      annotation_xshift=24)
    fig.update_xaxes(type="category", showticklabels=len(d) <= 40, tickangle=-60)
    fig.update_yaxes(title_text=basis_lbl, row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 102], row=2, col=1)
    fig.update_layout(bargap=0.15)
    return _base_layout(fig, 560)


def _scatter(rk, size_col, size_lbl, cur, doh_threshold):
    import plotly.graph_objects as go

    # < 1 ILS of stock (rounding residue) would stretch the log axis by decades.
    d = rk[rk["current_value"] >= 1].copy()
    finite = d["days_on_hand"].replace(np.inf, np.nan)
    cap = (finite.max() if finite.notna().any() else 365) * 1.6
    d["doh_plot"] = d["days_on_hand"].replace(np.inf, cap)
    d["doh_txt"] = np.where(np.isinf(d["days_on_hand"]), "no consumption in window",
                            d["days_on_hand"].round(0).map("{:,.0f} days".format))
    smax = d[size_col].max() or 1
    fig = go.Figure()
    for cls, color in _ABC_COLORS.items():
        s = d[d["abc"] == cls]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["doh_plot"], y=s["current_value"], mode="markers", name=f"Class {cls}",
            marker=dict(size=s[size_col].fillna(0), sizemode="area", sizeref=2 * smax / 55 ** 2, sizemin=4,
                        color=color, opacity=0.75, line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([s["item"], s["description"], s["doh_txt"], s[size_col].fillna(0),
                                 s["pattern_flags"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Days on hand: %{customdata[2]}"
                          f"<br>Current value: %{{y:,.0f}} {cur}<br>{size_lbl}: %{{customdata[3]:,.0f}} {cur}"
                          "<br><i>%{customdata[4]}</i><extra></extra>",
        ))
    # Reference lines as plain traces in data units — add_vline on log axes misplaces them.
    y_lo, y_hi = d["current_value"].min() * 0.7, d["current_value"].max() * 2
    for x, txt in [(doh_threshold, f"{doh_threshold:,} days"), (cap, "no consumption")]:
        fig.add_trace(go.Scatter(
            x=[x, x], y=[y_lo, y_hi], mode="lines+text", text=["", txt], textposition="top left",
            line=dict(color="#8a8984", width=1, dash="dot"), hoverinfo="skip", showlegend=False,
            textfont=dict(color="#52514e", size=11),
        ))
    fig.update_xaxes(type="log", title_text="Days on hand (log scale)")
    fig.update_yaxes(type="log", title_text=f"Current inventory value ({cur}, log scale)")
    return _base_layout(fig, 520)


def _top20(rk, basis_col, basis_lbl, cur):
    import plotly.graph_objects as go

    d = rk.head(20).iloc[::-1]
    lbl = d["item"] + " · " + d["description"].fillna("").str.slice(0, 38)
    fig = go.Figure(go.Bar(
        x=d[basis_col], y=lbl, orientation="h", marker_color=_BAR_COLOR,
        customdata=np.stack([d["abc"], d["pct_of_total"] * 100, d["days_on_hand"].replace(np.inf, np.nan)], axis=-1),
        hovertemplate="<b>%{y}</b><br>" + basis_lbl + f": %{{x:,.0f}} {cur}<br>Class %{{customdata[0]}} · "
                      "%{customdata[1]:.1f}% of total<br>Days on hand: %{customdata[2]:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title_text=f"{basis_lbl} ({cur})")
    fig.update_layout(bargap=0.25)
    return _base_layout(fig, 600)


def _material_detail(res, rk, cur):
    import plotly.graph_objects as go

    opts = rk["item"].tolist()
    fmt = dict(zip(rk["item"], "#" + rk["rank"].astype(str) + " · " + rk["item"] + " · " +
                   rk["description"].fillna("").str.slice(0, 50)))
    item = st.selectbox("Material", opts, format_func=lambda i: fmt.get(i, i), key="hc_detail_item")
    r = rk.set_index("item").loc[item]
    series = res["daily_value"].loc[item]

    m = st.columns(5)
    m[0].metric("Average value", _money(r["avg_value"], cur))
    m[1].metric("Max value", _money(r["max_value"], cur))
    m[2].metric("Annual holding cost", _money(r["annual_holding_cost"], cur))
    m[3].metric("Days on hand", "No consumption" if np.isinf(r["days_on_hand"]) else f"{r['days_on_hand']:,.0f}")
    m[4].metric("Months above target",
                "No target" if pd.isna(r["months_above_target"]) else f"{r['months_above_target']:.1f}")

    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode="lines", name="Inventory value",
        line=dict(color=_BAR_COLOR, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
        hovertemplate=f"%{{x|%d %b %Y}}<br>Value: %{{y:,.0f}} {cur}<extra></extra>",
    ))
    if pd.notna(r["target_qty"]):
        fig.add_hline(y=r["target_qty"] * r["unit_cost"], line=dict(color="#d03b3b", width=1.5, dash="dash"),
                      annotation_text=f"MAX INVENTORY target ({r['target_months']:g} months)",
                      annotation_position="top left")
    fig.add_hline(y=r["avg_value"], line=dict(color="#52514e", width=1, dash="dot"),
                  annotation_text="average", annotation_position="bottom left")
    fig.update_yaxes(title_text=f"Inventory value ({cur})", rangemode="tozero")
    fig.update_layout(showlegend=False, hovermode="x unified")
    st.plotly_chart(_base_layout(fig, 360), width="stretch")
    with st.container(key="hc_rtl_detail"):
        if r["pattern_flags"]:
            flags = r["pattern_flags"].split(", ")
            st.markdown("**דגלים:**\n" + "\n".join(f"- {_FLAG_HE.get(f, f)} (`{f}`)" for f in flags))
            st.markdown("**הסבר:**\n" + "\n".join(f"- {w}" for w in r["reason"].split("; ")))
        else:
            st.markdown("**אין דגלים לפריט זה.**")
        if r["data_notes"]:
            st.caption(f"⚠️ הערות נתונים: {r['data_notes']}")


def _ranking_table(rk, cur, has_rate):
    f1, f2, f3 = st.columns([1, 2, 2])
    abc_sel = f1.multiselect("ABC class", ["A", "B", "C"], default=["A", "B", "C"], key="hc_f_abc")
    flag_sel = f2.multiselect("Has flag (any of)", FLAG_ORDER, key="hc_f_flags")
    txt = f3.text_input("Search material / description", key="hc_f_txt")

    d = rk[rk["abc"].isin(abc_sel)]
    if flag_sel:
        d = d[d["pattern_flags"].apply(lambda s: any(f in s for f in flag_sel))]
    if txt:
        t = txt.strip().lower()
        d = d[d["item"].str.lower().str.contains(t, regex=False) |
              d["description"].fillna("").str.lower().str.contains(t, regex=False)]

    cols = {
        "rank": "Rank", "item": "Material", "description": "Description",
        "planner_group": "Group", "planner_code": "Planner", "abc": "ABC",
        "score": "Cost Score", "avg_qty": "Avg Inv Qty", "uom": "UOM", "avg_value": f"Avg Inv Value ({cur})",
        "current_value": f"Current Inv Value ({cur})", "days_on_hand": "Days on Hand",
        "excess_value": f"Excess Inv Value ({cur})", "annual_holding_cost": f"Est. Annual Holding Cost ({cur})",
        "pct_of_total": "% of Total Holding Cost", "cum_pct": "Cumulative %",
        "max_value": f"Max Inv Value ({cur})", "months_above_target": "Months Above Target",
        "turns": "Turns / yr", "days_since_consumption": "Days Since Consumption",
        "pattern_flags": "Flags", "reason": "הסבר", "data_notes": "הערות נתונים",
    }
    out = d[list(cols)].copy()
    out["days_on_hand"] = out["days_on_hand"].replace(np.inf, np.nan)
    out["pct_of_total"] *= 100
    out["cum_pct"] *= 100
    out = out.rename(columns=cols)

    nf = st.column_config.NumberColumn
    st.dataframe(
        out, hide_index=True, width="stretch", height=520,
        column_config={
            cols["avg_qty"]: nf(format="%.1f"),
            cols["avg_value"]: nf(format="localized"),
            cols["current_value"]: nf(format="localized"),
            cols["max_value"]: nf(format="localized"),
            cols["excess_value"]: nf(format="localized"),
            cols["annual_holding_cost"]: nf(format="localized",
                                            help=None if has_rate else "יש להזין שיעור עלות החזקה"),
            cols["days_on_hand"]: nf(format="localized", help="ריק = לא הייתה צריכה בתקופה"),
            cols["pct_of_total"]: nf(format="%.2f%%"),
            cols["cum_pct"]: nf(format="%.1f%%"),
            cols["months_above_target"]: nf(format="%.1f"),
            cols["turns"]: nf(format="%.2f"),
            cols["score"]: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )
    with st.container(key="hc_rtl_count"):
        st.caption(f"מוצגים {len(out)} מתוך {len(rk)} פריטים.")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="Holding Cost Ranking")
    st.download_button("📥 Download ranking (Excel)", data=buf.getvalue(),
                       file_name="inventory_holding_cost.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _methodology(res, cfg, cur, groups):
    with st.expander("ℹ️ שיטת החישוב, מקורות נתונים ואיכות נתונים"), st.container(key="hc_rtl_method"):
        st.markdown(f"""
- **היקף:** ארגון EM, קבוצות פריטים **{', '.join(groups)}** (RM1 = B_RM1, RM2 = B_RM2, Other = כל קוד
  מתכנן אחר או חסר). הדירוג, ה-ABC והציון מחושבים יחסית לקבוצות שנבחרו. התקופה: **{res['days']} הימים
  האחרונים** עד {(res['as_of'].normalize() - pd.Timedelta(days=1)):%d/%m/%Y}.
- **היסטוריית המלאי משוחזרת:** המלאי של היום (BTG_OnHand_V, כל ה-subinventories) "מגולגל אחורה" דרך כל
  התנועות ב-Material_Transactions, אחרי הסרת שורות כפולות. נבדק מול ה-snapshot השבועי האחרון ב-DWH
  (25/06/2025): 94% מהפריטים בסטייה של עד 1%, 97% עד 5%.
- **עלות יחידה:** עלות פריט ב-Oracle לתקופה הנוכחית, ב-**{cur}**, ומוחלת על כל התקופה.
- **שווי מלאי ממוצע** = כמות מלאי יומית ממוצעת × עלות יחידה.
  **עלות החזקה שנתית** = שווי מלאי ממוצע × שיעור עלות ההחזקה השנתי.
- **ימי מלאי** = כמות נוכחית ÷ צריכה יומית ממוצעת (WIP Issue / Account alias issue / Move Order Issue).
- **סבב מלאי (Turns)** = שווי הצריכה השנתית ÷ שווי המלאי הממוצע.
- **יעד** = MAX INVENTORY (בחודשים) מ-mrp_current.csv × צריכה חודשית ממוצעת. ערך ריק או סותר *לא* מוחלף
  בברירת מחדל, ולכן לפריטים אלה העודף לא נבדק.
- **ABC:** לפי החלק המצטבר מעלות ההחזקה — A עד {cfg['abc_thresholds']['A']:.0%},
  B עד {cfg['abc_thresholds']['B']:.0%}, C השאר.
- **POTENTIALLY OBSOLETE** דורש את שני התנאים: אין צריכה בתקופה *וגם* אין ביקוש עתידי ב-ASCP.
- **ציון עלות (Cost Score)** = ממוצע משוקלל של דירוגי אחוזון (0–100). סטטוס המשקלים:
  {cfg.get('score_weights_status', '')}.
""")
        st.markdown("**משמעות הדגלים:**\n" + "\n".join(f"- `{f}` — {_FLAG_HE[f]}" for f in FLAG_ORDER))
        if not res["excluded"].empty:
            st.markdown("**פריטים שהוצאו — לא ניתן לתמחר:**")
            st.dataframe(res["excluded"], hide_index=True, width="stretch")
        notes = res["ranking"][res["ranking"]["data_notes"] != ""][["item", "description", "data_notes"]]
        if not notes.empty:
            st.markdown(f"**פריטים עם הערות נתונים ({len(notes)}):**")
            st.dataframe(notes, hide_index=True, width="stretch")


def render_holding_cost():
    try:
        config = load_config()
    except Exception as e:
        st.error(f"לא ניתן לקרוא את holding_cost_config.json: {e}")
        return
    st.markdown(_RTL_CSS, unsafe_allow_html=True)
    cur = config.get("currency", "")
    groups = st.pills(
        "Item group", PLANNER_GROUPS, selection_mode="multi", default=["RM1", "RM2"], key="hc_groups",
        help="RM1 = מתכנן B_RM1, RM2 = B_RM2, Other = כל קוד מתכנן אחר (או חסר) בארגון EM. "
             "הדירוג, סיווג ה-ABC והציון מחושבים בתוך הקבוצות שנבחרו.",
    )
    if not groups:
        st.info("יש לבחור לפחות קבוצת פריטים אחת.")
        return
    cfg, rate = _settings(config)

    try:
        hist = load_inventory_history(int(cfg.get("analysis_days", 365)))
    except Exception as e:
        st.error(f"❌ {e}")
        return
    hist = filter_history(hist, groups)
    res = compute_holding_cost(hist, load_master(fill_defaults=False), cfg, rate=rate)
    rk = res["ranking"]
    if rk.empty:
        st.info("לא נמצא מלאי מתומחר בקבוצות שנבחרו.")
        return

    has_rate = rate is not None
    basis_col = "annual_holding_cost" if has_rate else "avg_value"
    basis_lbl = "Annual holding cost" if has_rate else "Average inventory value"

    _kpis(res, cur)
    with st.container(key="hc_rtl_asof"):
        st.caption(f"מלאי נכון ל-{res['as_of']:%d/%m/%Y %H:%M} · תקופת היסטוריה {res['days']} ימים · ערכים ב-{cur}"
                   + ("" if has_rate else " · בסיס הדירוג: שווי מלאי ממוצע (שיעור עלות החזקה לא הוגדר)"))

    t1, t2, t3, t4, t5 = st.tabs(["📊 Pareto / ABC", "🎯 Value vs Days on Hand", "🏆 Top 20",
                                  "📈 By Material", "📋 Ranking table"])
    with t1:
        n = st.slider("Materials shown", 10, len(rk), min(60, len(rk)), 10, key="hc_pareto_n") \
            if len(rk) > 10 else len(rk)
        st.plotly_chart(_pareto(rk, basis_col, basis_lbl, cfg, n), width="stretch")
        counts = rk.groupby("abc").agg(materials=("item", "size"), share=("pct_of_total", "sum"))
        cc = st.columns(3)
        for col, cls in zip(cc, ["A", "B", "C"]):
            if cls in counts.index:
                col.metric(f"Class {cls}", f"{counts.loc[cls, 'materials']} materials",
                           f"{counts.loc[cls, 'share']:.1%} מעלות ההחזקה", delta_color="off")
    with t2:
        st.plotly_chart(_scatter(rk, basis_col, basis_lbl, cur, cfg["flags"]["high_days_on_hand"]),
                        width="stretch")
        with st.container(key="hc_rtl_scatter"):
            st.caption("למעלה מימין = שווי גבוה שמוחזק זמן רב. גודל הבועה = "
                       + ("עלות החזקה שנתית" if has_rate else "שווי מלאי ממוצע")
                       + ". פריטים ללא צריכה מופיעים על הקו הימני.")
    with t3:
        st.plotly_chart(_top20(rk, basis_col, basis_lbl, cur), width="stretch")
    with t4:
        _material_detail(res, rk, cur)
    with t5:
        _ranking_table(rk, cur, has_rate)

    _methodology(res, cfg, cur, groups)

"""
rm_analysis_ui.py  –  Streamlit page for the raw-material analysis of a set of
parent products (src/rm_analysis.py): which RMs are unique to them / mainly
theirs, what N batches of each require and cost, and the stock position after
the production that is already open.
"""
import re

import pandas as pd
import streamlit as st

from src import rm_analysis as ra
from src.db import load_rm_bom, load_rm_items, load_rm_plan

_DEFAULT = "6316201000, 6316202000, 6306203000"
_RTL_CSS = """<style>
[class*="st-key-rm_rtl"] [data-testid="stMarkdownContainer"],
[class*="st-key-rm_rtl"] [data-testid="stCaptionContainer"],
[class*="st-key-rm_rtl"] [data-testid="stAlertContentInfo"],
[class*="st-key-rm_rtl"] [data-testid="stAlertContentWarning"] { direction: rtl; text-align: right; }
</style>"""
_ICON = {"חסר": "🔴", "חסר תוך הטווח הקצר": "🟠", "מספיק לדרישה אחת בלבד": "🟡", "תקין": "🟢", "לא נדרש בדרישה הנוכחית": "⚪"}
_NF = st.column_config.NumberColumn
_TX = st.column_config.TextColumn


@st.cache_data(ttl=300, show_spinner="מחשב…")
def _run(targets: tuple, family: tuple, batches: tuple, fwd: int, short: int, main_share: float):
    bom = load_rm_bom()
    leaves = ra.structure(bom, targets, family)["leaves"]
    R = ra.analyze(bom, load_rm_plan(), load_rm_items(tuple(leaves)), list(targets), family, dict(batches),
                   fwd_months=fwd, short_months=short, main_share=main_share)
    return R, ra.product_info(bom)


def _inputs(bom: pd.DataFrame):
    pinfo = ra.product_info(bom)
    label = lambda p: f"{p} — {pinfo.FORMULA_DESC1.get(p, '')}"
    raw = st.text_input("מק״טי אב לניתוח (מופרדים בפסיק)", _DEFAULT, key="rm_targets")
    targets = list(dict.fromkeys(t for t in re.split(r"[\s,;]+", raw) if t))
    bad = [t for t in targets if t not in pinfo.index]
    if bad:
        st.error(f"מק״טים ללא עץ מוצר ב-Indented_Bills: {', '.join(bad)}")
    targets = [t for t in targets if t in pinfo.index]
    if not targets:
        return None
    cols = st.columns(len(targets))
    batches = {}
    for c, t in zip(cols, targets):
        batches[t] = c.number_input(
            f"{label(t)}", min_value=0.0, value=1.0, step=1.0, key=f"rm_b_{t}",
            help=f"מספר אצוות; אצווה אחת = {pinfo.PQ.get(t):g} {pinfo.PRODUCT_UOM.get(t)} (כמות המוצר בפורמולה)")
    with st.expander("⚙️ הגדרות מתקדמות"):
        roots = sorted(set(bom.PRODUCT) - set(bom.INGREDIENT) - set(targets))
        sug = ra.suggest_family(bom, targets)
        family = st.multiselect(
            "מוצרים קשורים מאותה משפחה (לא מחושבים בדרישה, אבל השימוש שלהם לא הופך חומר גלם ל'משותף')",
            roots, default=sug, format_func=label, key=f"rm_fam_{'_'.join(targets)}",
            help="הצעה אוטומטית: מוצרי קצה שהתיאור שלהם מתחיל באותה מילה כמו אחד המק״טים (למשל Aminopeptidase weekly seeding)")
        c1, c2, c3 = st.columns(3)
        fwd = c1.number_input("אופק תכנון ASCP (חודשים)", 1, 36, 12, key="rm_fwd")
        short = c2.number_input("טווח קצר למצב מלאי (חודשים)", 1, 12, 3, key="rm_short")
        main_share = c3.slider("סף 'מרכזי' – חלק היעד מהצריכה", 0.3, 0.9, ra.MAIN_SHARE, 0.05, key="rm_main")
    return targets, family, batches, int(fwd), int(short), float(main_share)


def _kpis(R: pd.DataFrame):
    k = st.columns(6)
    k[0].metric("חומרי גלם בעץ", f"{len(R)}")
    for i, c in enumerate(ra.CLASSES, 1):
        m = R.cls == c
        k[i].metric(c, f"{m.sum()}", f"₪{R.loc[m, 'cost'].sum():,.0f}", delta_color="off")
    k[4].metric("עלות הדרישה", f"₪{R.cost.sum():,.0f}")
    k[5].metric("חסרים", f"{(R.status == 'חסר').sum()}",
                f"{(R.status == 'מספיק לדרישה אחת בלבד').sum()} מספיקים לדרישה אחת בלבד", delta_color="off")


def _materials(R: pd.DataFrame, targets, pinfo):
    c1, c2 = st.columns([2, 3])
    cls = c1.pills("סיווג", ra.CLASSES, selection_mode="multi", default=ra.CLASSES, key="rm_cls")
    only = c2.toggle("רק פריטים שנדרשים בדרישה הנוכחית", True, key="rm_only")
    df = R[R.cls.isin(cls or ra.CLASSES)]
    if only:
        df = df[df.req > 0]
    bcols = {f"b_{t}": _NF(f"לאצווה {t}", format="%.3f", help=pinfo.FORMULA_DESC1.get(t, "")) for t in targets}
    show = df.reset_index()[["item", "INGREDIENT_DESCRIPTION", "ITEM_TYPE", "cls", "share_fwd", "share_3y", "others", "ING_UOM",
                             *bcols, "req", "pack", "packs", "price", "psrc", "cost", "vendor"]]
    show[["share_fwd", "share_3y"]] = show[["share_fwd", "share_3y"]] * 100
    st.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "item": "מק״ט", "INGREDIENT_DESCRIPTION": _TX("תיאור", width="medium"), "ITEM_TYPE": "סוג", "cls": "סיווג",
        "share_fwd": _NF("חלק היעד – קדימה", format="%.0f%%"), "share_3y": _NF("חלק היעד – 3 שנים", format="%.0f%%"),
        "others": _TX("צרכנים אחרים", width="medium"), "ING_UOM": "יחידה", **bcols,
        "req": _NF("כמות נדרשת", format="%,.2f"), "pack": _NF("יחידות במארז", format="%g"),
        "packs": _NF("מארזים", format="%,.2f"), "price": _NF("מחיר למארז", format="₪%,.2f"), "psrc": "מקור מחיר",
        "cost": _NF("עלות", format="₪%,.0f"), "vendor": _TX("ספק", width="medium"),
    })
    st.caption(f"{len(df)} פריטים · עלות בסינון ₪{df.cost.sum():,.0f}")


def _stock(R: pd.DataFrame, short: int):
    c1, c2 = st.columns([3, 2])
    sts = [s for s in _ICON if s in set(R.status)]
    pick = c1.pills("סטטוס", sts, selection_mode="multi", default=[s for s in sts if s != "לא נדרש בדרישה הנוכחית"],
                    format_func=lambda s: f"{_ICON[s]} {s}", key="rm_sts")
    df = R[R.status.isin(pick or sts)].copy()
    df["o"] = df.status.map({s: i for i, s in enumerate(_ICON)})
    df = df.sort_values(["o", "bal"]).reset_index()
    df["status"] = df.status.map(lambda s: f"{_ICON[s]} {s}")
    df["bal_packs"] = df.bal / df.pack
    sm = f"{short} ח'"
    show = df[["status", "item", "INGREDIENT_DESCRIPTION", "cls", "ING_UOM", "oh_ok", "oh_q", "oh_exp", "wo_oth", "wo_chain", "req",
               "bal", "bal_packs", "n_more", "pl_short_oth", "po_short", "bal_short", "po_next", "exp_min", "exp_short", "notes"]]
    st.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "status": _TX("סטטוס", width="medium"), "item": "מק״ט", "INGREDIENT_DESCRIPTION": _TX("תיאור", width="medium"),
        "cls": "סיווג", "ING_UOM": "יחידה",
        "oh_ok": _NF("מלאי מאושר בתוקף", format="%,.2f"), "oh_q": _NF("בהסגר", format="%,.2f"),
        "oh_exp": _NF("פג תוקף", format="%,.2f", help="לא נספר במלאי"),
        "wo_oth": _NF("WO פתוחים – אחרים", format="%,.2f", help="יתרת הזמנות עבודה פתוחות של מוצרים אחרים"),
        "wo_chain": _NF("WO פתוחים – היעד", format="%,.2f", help="יתרת הזמנות עבודה פתוחות של המק״טים הנבחרים (הייצור הנוכחי)"),
        "req": _NF("דרישה", format="%,.2f"),
        "bal": _NF("יתרה אחרי ייצור קיים ודרישה", format="%,.2f"), "bal_packs": _NF("יתרה במארזים", format="%,.2f"),
        "n_more": _NF("מספיק לכמה דרישות", format="%.1f"),
        "pl_short_oth": _NF(f"ביקוש מתוכנן אחרים – {sm}", format="%,.2f"), "po_short": _NF(f"PO עד {sm}", format="%,.2f"),
        "bal_short": _NF(f"יתרה צפויה {sm}", format="%,.2f"),
        "po_next": st.column_config.DateColumn("PO הבא", format="DD/MM/YYYY"),
        "exp_min": st.column_config.DateColumn("תפוגה קרובה", format="DD/MM/YYYY"),
        "exp_short": _NF(f"יפוג תוך {sm}", format="%,.2f"), "notes": _TX("הערה", width="large"),
    })
    with st.container(key="rm_rtl_stock"):
        st.caption("יתרה = מלאי מאושר בתוקף − הזמנות עבודה פתוחות (כולל של היעד) − הדרישה. "
                   f"יתרה צפויה {sm} = יתרה − ביקוש מתוכנן של מוצרים אחרים + PO פתוחים, עד {sm}. כל הכמויות ביחידת העץ.")


def _methodology():
    with st.expander("ℹ️ איך זה מחושב"), st.container(key="rm_rtl_method"):
        st.markdown("""
- **עץ מוצר:** Indented_Bills. חומר גלם = עלה בעץ של המק״טים הנבחרים. דרישה לאצווה = פיצוץ העץ דרך כל התמיסות והבופרים, עם עצירה במק״טים הנבחרים האחרים (כדי ששלב בשרשרת לא ייספר פעמיים). אצווה = כמות המוצר בפורמולה.
- **ייחודי:** אף מוצר קצה אחר לא מגיע אליו בלי לעבור דרך המק״טים הנבחרים או מוצרי המשפחה.
- **מרכזי:** לפחות 50% (ניתן לשינוי) מהדרישה הצפויה שלו הולכים ליעד. הדרישה הצפויה היא הזמנות עבודה ומתוכננות ב-ASCP באופק הנבחר, מגולגלות במורד העץ לפי חלקו של כל צרכן. אם אין תכנון, החישוב לפי צריכה בפועל ב-3 שנים (WIP_Last_3_Years).
- **מחיר:** שורת ה-PO האחרונה (לפי מספר PO, ללא מבוטלות), מומרת ל-₪ לפי שער ה-PO. אם אין PO, עלות תקן. גודל המארז נלקח מתיאור הפריט.
- **מלאי:** BTG_OnHand_V בארגון EM. לא כולל הסגר (Q) ואצוות שפג תוקפן. הביקוש מ-ASCP: הזמנות עבודה פתוחות, ביקוש מתוכנן ותחזית. הביקוש משויך ליעד אם ההזמנה המולידה שייכת למק״ט נבחר או לביניים שמשמש רק אותו. PO, דרישות רכש ומשלוחים בדרך נספרים כאספקה.
""")


def render_rm_analysis():
    st.markdown(_RTL_CSS, unsafe_allow_html=True)
    try:
        bom = load_rm_bom()
    except Exception as e:
        st.error(f"טעינת עץ המוצר נכשלה: {e}")
        return
    inp = _inputs(bom)
    if not inp:
        st.info("יש להזין לפחות מק״ט אב אחד עם עץ מוצר.")
        return
    # The table only recalculates on Apply; the first visit applies the defaults.
    pending = (tuple(inp[0]), tuple(inp[1]), tuple(sorted(inp[2].items())), *inp[3:])
    if "rm_applied" not in st.session_state:
        st.session_state.rm_applied = pending
    changed = pending != st.session_state.rm_applied
    c1, c2 = st.columns([1, 4])
    if c1.button("✅ Apply", key="rm_apply", type="primary", disabled=not changed, width="stretch"):
        st.session_state.rm_applied = pending
        st.rerun()
    if changed:
        with c2.container(key="rm_rtl_pending"):
            st.warning("יש שינויים שלא הוחלו – לחצו **Apply** כדי לרענן את הטבלה. התוצאות למטה עדיין לפי ההגדרות הקודמות.")
    t_, f_, b_, fwd, short, main_share = st.session_state.rm_applied
    targets, family, batches = list(t_), list(f_), dict(b_)
    try:
        R, pinfo = _run(t_, f_, b_, fwd, short, main_share)
    except Exception as e:
        st.error(f"הניתוח נכשל: {e}")
        return
    if R.empty:
        st.info("לא נמצאו חומרי גלם בעצים של המק״טים שנבחרו.")
        return

    _kpis(R)
    with st.container(key="rm_rtl_asof"):
        st.caption(f"נתוני Oracle / ASCP נכון ל-{pd.Timestamp.today():%d/%m/%Y} · הדרישה: "
                   + ", ".join(f"{batches[t]:g} × {t}" for t in targets))
    tabs = st.tabs(["📋 חומרי גלם, כמויות ומחירים", "📦 מצב מלאי מול הייצור הקיים"])
    with tabs[0]:
        _materials(R, targets, pinfo)
    with tabs[1]:
        _stock(R, short)

    xl = ra.build_workbook(R, targets, pinfo, batches, short)
    st.download_button("📥 הורדה ל-Excel (עם נוסחאות)", data=xl, file_name=f"RM_analysis_{'_'.join(targets)}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rm_dl")
    _methodology()

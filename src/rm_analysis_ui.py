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
    raw = st.text_input("מק״טי אב לניתוח (מופרדים בפסיק)", _DEFAULT, key="rm_targets",
                        help="מק״טים שיש להם עץ מוצר (פורמולה) ב-Oracle. כל חומרי הגלם בעצים שלהם ינותחו. "
                             "כשמזינים כמה שלבים של אותה שרשרת (Fermentation → Recovery → Bulk), כל שלב נספר רק פעם אחת.")
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
    need = R[R.req > 0]
    k = st.columns(5)
    k[0].metric("חומרי גלם בדרישה", f"{len(need)}", f"מתוך {len(R)} בעץ", delta_color="off",
                help="חומרי גלם שנדרשים לדרישה הנוכחית. השאר הם קודים חלופיים בעץ (כמות 0 בפורמולה הפעילה).")
    for i, c in enumerate(ra.CLASSES, 1):
        m = need.cls == c
        k[i].metric(c, f"{m.sum()} פריטים", f"₪{need.loc[m, 'cost'].sum():,.0f}", delta_color="off", help=_CLS_HELP[c])
    short = (need.status == "חסר").sum()
    k[4].metric("עלות הדרישה", f"₪{need.cost.sum():,.0f}",
                f"🔴 {short} חסרים · 🟡 {(need.status == 'מספיק לדרישה אחת בלבד').sum()} גבוליים", delta_color="off",
                help="סכום העלות של כל חומרי הגלם לדרישה (מארזים × מחיר למארז). "
                     "חסרים / גבוליים – לפי לשונית מצב המלאי.")


_CLS_HELP = {
    "ייחודי": "משמש רק את המק״טים שנבחרו (ומוצרי המשפחה שלהם) – אף מוצר אחר לא צורך אותו.",
    "מרכזי": "משמש גם מוצרים אחרים, אבל רוב הצריכה (מעל הסף, ברירת מחדל 50%) הולכת למק״טים שנבחרו.",
    "משותף": "משמש בעיקר מוצרים אחרים; המק״טים שנבחרו צורכים פחות ממחצית.",
}


def _materials(R: pd.DataFrame):
    c1, c2 = st.columns([2, 3])
    cls = c1.pills("סיווג", ra.CLASSES, selection_mode="multi", default=ra.CLASSES, key="rm_cls",
                   help="\n\n".join(f"**{k}** – {v}" for k, v in _CLS_HELP.items()))
    only = c2.toggle("רק פריטים שנדרשים", True, key="rm_only",
                     help="הסתרת קודים חלופיים בעץ שאינם בשימוש בפורמולה הפעילה (כמות נדרשת 0)")
    df = R[R.cls.isin(cls or ra.CLASSES)]
    if only:
        df = df[df.req > 0]
    df = df.reset_index()
    df["share"] = df.share_fwd.where(df.req12.fillna(0) > 0, df.share_3y) * 100
    show = df[["item", "INGREDIENT_DESCRIPTION", "cls", "share", "req", "ING_UOM", "packs", "price", "cost", "vendor"]]
    st.dataframe(show, hide_index=True, width="stretch", height=520, column_config={
        "item": _TX("מק״ט", help="מק״ט חומר הגלם ב-Oracle"),
        "INGREDIENT_DESCRIPTION": _TX("תיאור", width="large", help="תיאור הפריט; גודל המארז נלקח מכאן (למשל Pack12KG)"),
        "cls": _TX("סיווג", help=" · ".join(f"{k}: {v}" for k, v in _CLS_HELP.items())),
        "share": _NF("% לצריכת היעד", format="%.0f%%",
                     help="איזה חלק מהצריכה של חומר הגלם הולך למק״טים שנבחרו. לפי תכנון ASCP באופק התכנון; "
                          "אם אין תכנון – לפי צריכה בפועל ב-3 השנים האחרונות. 100% = ייחודי."),
        "req": _NF("כמות נדרשת", format="%,.2f",
                   help="כמות חומר הגלם לדרישה שהוזנה (מספר אצוות × כמות לאצווה לפי עץ המוצר, כולל דרך תמיסות ובופרים). "
                        "ביחידת העץ – ראו עמודת יחידה."),
        "ING_UOM": _TX("יחידה", help="יחידת המידה בעץ המוצר (KG / L / PCS…)"),
        "packs": _NF("מארזים", format="%,.2f",
                     help="כמות נדרשת ÷ גודל מארז. מספר עשרוני = חלק ממארז; ברכישה יש לעגל למעלה."),
        "price": _NF("מחיר למארז", format="₪%,.2f",
                     help="מחיר שורת ה-PO האחרונה (לפי מספר PO), מומר ל-₪ לפי שער ה-PO. אם אין PO – עלות תקן. "
                          "מספר ה-PO ומטבע המקור – בקובץ ה-Excel."),
        "cost": _NF("עלות", format="₪%,.0f", help="מארזים × מחיר למארז"),
        "vendor": _TX("ספק", width="medium", help="הספק ב-PO האחרון; אם אין PO – הספק המאושר (ASL)"),
    })
    st.caption(f"{len(df)} פריטים · עלות ₪{df.cost.sum():,.0f}")


def _stock(R: pd.DataFrame, short: int):
    sts = [s for s in _ICON if s in set(R.status)]
    pick = st.pills("סטטוס", sts, selection_mode="multi", default=[s for s in sts if s != "לא נדרש בדרישה הנוכחית"],
                    format_func=lambda s: f"{_ICON[s]} {s}", key="rm_sts", help=_STATUS_HELP)
    df = R[R.status.isin(pick or sts)].copy()
    df["o"] = df.status.map({s: i for i, s in enumerate(_ICON)})
    df = df.sort_values(["o", "bal"]).reset_index()
    df["status"] = df.status.map(lambda s: f"{_ICON[s]} {s}")
    df["reserved"] = df.wo_oth + df.wo_chain
    show = df[["status", "item", "INGREDIENT_DESCRIPTION", "ING_UOM", "oh_ok", "reserved", "req", "bal", "n_more",
               "bal_short", "po_next", "notes"]]
    st.dataframe(show, hide_index=True, width="stretch", height=520, column_config={
        "status": _TX("מצב", width="medium", help=_STATUS_HELP),
        "item": _TX("מק״ט", help="מק״ט חומר הגלם ב-Oracle"),
        "INGREDIENT_DESCRIPTION": _TX("תיאור", width="medium"),
        "ING_UOM": _TX("יחידה", help="כל הכמויות בטבלה ביחידת העץ (KG / L / PCS…), לא במארזים"),
        "oh_ok": _NF("מלאי זמין", format="%,.2f",
                     help="מלאי ב-Oracle (ארגון EM) בסטטוס מאושר ובתוקף. לא כולל הסגר ואצוות שפג תוקפן – הן מופיעות בהערות."),
        "reserved": _NF("שמור לייצור פתוח", format="%,.2f",
                        help="יתרה שטרם הונפקה בהזמנות עבודה פתוחות ב-ASCP – של כל המוצרים, "
                             "כולל הזמנות פתוחות של המק״טים שנבחרו (הקמפיין הנוכחי)."),
        "req": _NF("נדרש", format="%,.2f", help="הכמות לדרישה שהוזנה – בנוסף לייצור הפתוח"),
        "bal": _NF("יתרה", format="%,.2f",
                   help="מלאי זמין − שמור לייצור פתוח − נדרש. שלילי = חסר היום."),
        "n_more": _NF("מספיק ל-X דרישות", format="%.1f",
                      help="כמה דרישות כאלה המלאי מכסה אחרי הייצור הפתוח. 1.0–2.0 = מספיק לדרישה אחת בלבד."),
        "bal_short": _NF(f"יתרה צפויה ({short} ח')", format="%,.2f",
                         help=f"היתרה בעוד {short} חודשים: + PO פתוחים (כולל דרישות רכש ומשלוחים בדרך) "
                              f"− ביקוש מתוכנן ותחזית של מוצרים אחרים ב-ASCP. שלילי = צפוי חוסר."),
        "po_next": st.column_config.DateColumn("PO הבא", format="DD/MM/YYYY",
                                               help="תאריך האספקה הקרוב של PO / דרישת רכש פתוחה ב-ASCP"),
        "notes": _TX("הערות", width="large",
                     help="פג תוקף, הסגר, תפוגה קרובה, PO ורכש מתוכנן ב-ASCP, ייצור מתוכנן נוסף של היעד, "
                          "וקודים חלופיים בעץ עם מלאי. לחיצה כפולה על תא מציגה את הטקסט המלא."),
    })


_STATUS_HELP = ("🔴 **חסר** – אין מספיק מלאי לדרישה אחרי הייצור הפתוח.  \n"
                "🟠 **חסר תוך הטווח הקצר** – יש היום, אבל לפי ביקוש מתוכנן ו-PO צפוי חוסר.  \n"
                "🟡 **מספיק לדרישה אחת בלבד** – אחרי הדרישה נשאר פחות מדרישה נוספת.  \n"
                "🟢 **תקין** – מספיק לפחות לשתי דרישות.  \n"
                "⚪ **לא נדרש** – קוד חלופי בעץ, לא בשימוש בפורמולה הפעילה.")


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
    if c1.button("✅ Apply", key="rm_apply", type="primary", disabled=not changed, width="stretch",
                 help="חישוב הטבלאות מחדש לפי המק״טים, האצוות וההגדרות שהוזנו"):
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
        _materials(R)
    with tabs[1]:
        _stock(R, short)

    xl = ra.build_workbook(R, targets, pinfo, batches, short)
    st.download_button("📥 הורדה ל-Excel (עם נוסחאות)", data=xl, file_name=f"RM_analysis_{'_'.join(targets)}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rm_dl",
                       help="קובץ עם שני גיליונות ונוסחאות חיות: כל העמודות המלאות (כמות לאצווה לכל מק״ט, מקור מחיר, הסגר, תפוגה, PO), ותאים צהובים לשינוי מספר האצוות")
    _methodology()

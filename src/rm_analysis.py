"""
rm_analysis.py — raw-material analysis for a set of parent products (built for
the Aminopeptidase chain 6316201000 Fermentation / 6316202000 Recovery /
6306203000 Bulk, works for any set of formulas).

For every raw material (BOM leaf) in the parents' trees:
  * class      ייחודי  — reachable from no other top-level product once the
                         traversal is blocked at the targets (and at related
                         family products, e.g. "Aminopeptidase weekly seeding")
               מרכזי   — the targets take ≥ 50% of its forward 12-month
                         requirement (ASCP work + planned orders propagated
                         down the BOM); with no forward plan, of the last 3
                         years' actual WIP consumption
               משותף   — everything else
  * requirement per batch of each target (BOM explosion; one batch = the
    formula's product quantity, e.g. 30 G for the Bulk), blocked at the
    other targets so a chain step is never counted twice
  * price      latest PO line by PO_NO, converted to ₪ at the PO's rate;
               otherwise standard cost
  * stock      approved, non-expired on-hand − open work-order demand (all
               products, including the targets' own open orders) − the
               requirement; plus a short-horizon projection with open POs and
               other products' planned demand

Pure pandas: the page (src/rm_analysis_ui.py) fetches inputs via src/db.py.
build_workbook() writes the same two tables as Excel with live formulas.
"""
import io
import re

import numpy as np
import pandas as pd

MAIN_SHARE = 0.5
CLASSES = ["ייחודי", "מרכזי", "משותף"]
_FAMILY = "יעד"
_PACK_RE = re.compile(r"(?:1\s*Pack\s*=\s*|Pac(?:k)?\s*(?:of\s*)?)(\d+(?:\.\d+)?)\s*(KG|G|L|ML|M|PCS|Pcs)", re.I)


# ── BOM structure ───────────────────────────────────────────────

def _stem(desc) -> str:
    w = re.split(r"[\s(/,-]+", str(desc or "").strip().lower())
    return w[0] if w and len(w[0]) >= 4 else ""


def product_info(bom: pd.DataFrame) -> pd.DataFrame:
    return bom.drop_duplicates("PRODUCT").set_index("PRODUCT")[["FORMULA_DESC1", "PQ", "PRODUCT_UOM"]]


# Abbreviations used as the first word of related formulas' descriptions
# ("AP - Periodic column sampling" belongs to the Aminopeptidase family).
_ALIASES = {"aminopeptidase": {"ap"}}


def suggest_family(bom: pd.DataFrame, targets) -> list:
    """Other top-level products of the same family (description starts with
    the same word as a target's, or a known abbreviation of it), e.g. the
    weekly seeding that also uses the Aminopeptidase WCB."""
    desc = product_info(bom).FORMULA_DESC1
    words = {_stem(desc.get(t)) for t in targets} - {""}
    words |= {a for w in words for a in _ALIASES.get(w, ())}
    first = lambda d: (re.split(r"[\s(/,-]+", str(d or "").strip().lower()) or [""])[0]
    roots = set(bom.PRODUCT) - set(bom.INGREDIENT) - set(targets)
    return sorted(r for r in roots if first(desc.get(r)) in words)


def _reach(kids: dict, starts, block=frozenset()) -> set:
    seen, stack = set(), list(starts)
    while stack:
        for c in kids.get(stack.pop(), ()):
            if c not in block and c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _edges(bom: pd.DataFrame) -> pd.DataFrame:
    e = bom.drop_duplicates(["PRODUCT", "INGREDIENT"]).copy()
    e["per"] = e.ING_QTY / e.PQ.replace(0, np.nan)
    return e


def structure(bom: pd.DataFrame, targets, family=()) -> dict:
    """Leaves of the targets' trees, which of them are unique, and the
    intermediates used only by the targets (their demand counts as the
    targets')."""
    T, F = set(targets), set(family)
    kids = bom.groupby("PRODUCT").INGREDIENT.apply(set).to_dict()
    products = set(bom.PRODUCT)
    in_t = _reach(kids, T)
    roots = products - set(bom.INGREDIENT) - T - F
    other = _reach(kids, roots, block=frozenset(T | F))
    leaves = sorted(x for x in in_t if x not in products)
    return {
        "leaves": leaves,
        "unique": {x for x in leaves if x not in other},
        "chain": T | F | ((in_t & products) - other),
        "products": products,
    }


def per_batch(bom: pd.DataFrame, targets) -> pd.DataFrame:
    """Leaf quantity for one batch of each target (columns = targets)."""
    e = _edges(bom)
    kids = {p: list(zip(g.INGREDIENT, g.per)) for p, g in e.groupby("PRODUCT")}
    products, T = set(bom.PRODUCT), set(targets)
    pq = product_info(bom).PQ

    def explode(root, qty, acc, path):
        for c, per in kids.get(root, []):
            if c in T or pd.isna(per) or c in path:
                continue
            if c in products:
                explode(c, qty * per, acc, path | {c})
            else:
                acc[c] = acc.get(c, 0) + qty * per
        return acc

    return pd.DataFrame({t: pd.Series(explode(t, float(pq.get(t) or 1), {}, {t}), dtype=float) for t in targets}).fillna(0)


# ── share of demand going to the targets ────────────────────────

def shares(bom, supply, wip3y, chain, leaves, today, months=12) -> pd.DataFrame:
    """share_fwd: targets' part of each leaf's forward requirement (ASCP work +
    planned orders due within `months`, propagated through every intermediate
    in proportion to who consumes it); share_3y: same on actual WIP
    consumption. Also the other consumer families (top 'others' string)."""
    e = _edges(bom)
    s = supply.copy()
    s["due"] = pd.to_datetime(s.NEW_DUE_DATE, errors="coerce")
    sup = s[s.due < today + pd.DateOffset(months=months)].groupby("ITEM").QUANTITY.sum()
    e = e[e.PRODUCT.isin(sup.index)].copy()
    e["req"] = e.PRODUCT.map(sup) * e.per
    par = {x: list(zip(g.PRODUCT, g.req)) for x, g in e.groupby("INGREDIENT")}
    memo_s, memo_f = {}, {}

    def share(x, path=()):
        if x in chain:
            return 1.0
        if x in memo_s:
            return memo_s[x]
        ps = [(p, r) for p, r in par.get(x, []) if p not in path]
        tot = sum(r for _, r in ps)
        v = float(sum(r * share(p, path + (x,)) for p, r in ps) / tot) if tot > 0 else 0.0
        memo_s[x] = v
        return v

    def fam(x, path=()):
        """x's demand split by final consumer (top-level producer)."""
        if x in chain:
            return {_FAMILY: 1.0}
        if x in memo_f:
            return memo_f[x]
        ps = [(p, r) for p, r in par.get(x, []) if p not in path]
        tot = sum(r for _, r in ps)
        if not ps or tot <= 0:
            return {x: 1.0}
        out = {}
        for p, r in ps:
            for k, v in fam(p, path + (x,)).items():
                out[k] = out.get(k, 0) + v * r / tot
        memo_f[x] = out
        return out

    desc = product_info(bom).FORMULA_DESC1.to_dict()

    def fam_name(k):
        if k == _FAMILY:
            return k
        n = re.sub(r"\s+", " ", str(desc.get(k, k)))[:28]
        return n.split("(")[0].split(" API")[0].strip()

    rows = []
    for x in leaves:
        ps = par.get(x, [])
        req = sum(r for _, r in ps)
        chain_req = sum(r * share(p) for p, r in ps)
        agg = {}
        for p, r in ps:
            for k, v in fam(p).items():
                agg[fam_name(k)] = agg.get(fam_name(k), 0) + v * r
        oth = sorted(((k, v) for k, v in agg.items() if k != _FAMILY), key=lambda kv: -kv[1])
        rows.append({"item": x, "req12": req, "chain12": chain_req,
                     "others": "; ".join(f"{k} {v / req:.0%}" for k, v in oth)[:80] if req > 0 else ""})
    f = pd.DataFrame(rows).set_index("item")
    f["share_fwd"] = f.chain12 / f.req12.replace(0, np.nan)

    w = wip3y.copy()
    w = w[w.ITEM.isin(leaves)]
    w["cs"] = w.ACTUAL_QTY * w.FORMULA_NO.map(lambda p: share(p))
    h = w.groupby("ITEM").agg(act3y=("ACTUAL_QTY", "sum"), chain3y=("cs", "sum"))
    f = f.join(h)
    f["share_3y"] = f.chain3y / f.act3y.replace(0, np.nan)
    return f


# ── prices ──────────────────────────────────────────────────────

def _pack(desc, u, puom) -> float:
    """Units (in the BOM UOM) per purchased pack, parsed from the description."""
    if puom in ("PCS", None) or u == "PAC" or (isinstance(puom, str) and puom == u):
        return 1.0
    m = _PACK_RE.search(str(desc).replace(",", "."))
    if not m:
        return np.nan
    v, un, u = float(m.group(1)), m.group(2).upper(), str(u).upper()
    if un == u:
        return v
    if un == "KG" and u == "G":
        return v * 1000
    if un == "G" and u == "KG":
        return v / 1000
    return np.nan


def prices(items: dict) -> pd.DataFrame:
    po = items["po"].copy()
    po["price"] = pd.to_numeric(po.LN_U_PRICE, errors="coerce")
    po = po[(po.price > 0) & (po.LN_CANCEL_F.fillna("N") != "Y")].copy()
    po["p_ils"] = po.price * pd.to_numeric(po.RATE, errors="coerce").fillna(1)
    po["pono"] = pd.to_numeric(po.PO_NO, errors="coerce")
    po["ln"] = pd.to_numeric(po.LN, errors="coerce")
    po = po.sort_values(["pono", "ln"]).groupby("ITEM").tail(1).set_index("ITEM")
    c = items["cost"].copy()
    c["o"] = (c.ORGANIZATION_CODE != "EM").astype(int)
    std = c.dropna(subset=["ITEM_COST"]).sort_values(["o", "ORGANIZATION_CODE"]).groupby("ITEM").ITEM_COST.first()
    out = po[["p_ils", "PO_NO", "CURR", "VENDOR_NAME"]].join(std.rename("std_cost"), how="outer")
    puom = items["master"].drop_duplicates("ITEM").set_index("ITEM").PRIMARY_UOM_CODE
    return out.join(puom.rename("puom"), how="outer")


# ── the analysis table ──────────────────────────────────────────

def analyze(bom, plan, items, targets, family=(), batches=None, today=None,
            fwd_months=12, short_months=3, main_share=MAIN_SHARE) -> pd.DataFrame:
    """One row per raw material (index = item). `batches` = {target: n}."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    batches = {t: float((batches or {}).get(t, 1)) for t in targets}
    st_ = structure(bom, targets, family)
    leaves = st_["leaves"]
    pb = per_batch(bom, targets).reindex(leaves).fillna(0)
    info = bom.drop_duplicates("INGREDIENT").set_index("INGREDIENT").loc[
        leaves, ["INGREDIENT_DESCRIPTION", "ITEM_TYPE", "ING_UOM", "ASL_VENDOR_NAME"]]
    R = info.join(pb.add_prefix("b_"))
    R["req"] = sum(pb[t] * batches[t] for t in targets)

    sh = shares(bom, plan["supply"], plan["wip3y"], st_["chain"], leaves, today, fwd_months)
    R = R.join(sh)
    R["unique"] = R.index.isin(st_["unique"])
    main = np.where(R.req12.fillna(0) > 0, R.share_fwd.fillna(0) >= main_share, R.share_3y.fillna(0) >= main_share)
    R["cls"] = np.where(R.unique, "ייחודי", np.where(main, "מרכזי", "משותף"))

    P = prices(items).reindex(leaves)
    R = R.join(P)
    R["pack"] = [_pack(d, u, p if isinstance(p, str) else None)
                 for d, u, p in zip(R.INGREDIENT_DESCRIPTION, R.ING_UOM, R.puom)]
    R["price"] = R.p_ils.fillna(R.std_cost.where(R.std_cost > 1))
    R["psrc"] = np.where(R.p_ils.notna(), "PO " + R.PO_NO.fillna("").astype(str) + " (" + R.CURR.fillna("") + ")",
                         np.where(R.price.notna(), "עלות תקן", ""))
    R["vendor"] = R.VENDOR_NAME.fillna(R.ASL_VENDOR_NAME)
    R["packs"] = R.req / R.pack.replace(0, np.nan)
    R["cost"] = R.packs * R.price

    R = R.join(stock(items, plan["supply"], leaves, R, st_["chain"], today, fwd_months, short_months))
    R["notes"] = _notes(R, today)
    R["o"] = R.cls.map({c: i for i, c in enumerate(CLASSES)})
    R = R.sort_values(["o", "cost"], ascending=[True, False], na_position="last").drop(columns="o")
    R.index.name = "item"
    return R


def stock(items, supply, leaves, R, chain, today, fwd_months=12, short_months=3) -> pd.DataFrame:
    """Stock position per leaf, in BOM units."""
    h_s, h_l = today + pd.DateOffset(months=short_months), today + pd.DateOffset(months=fwd_months)
    f = pd.Series(np.where((R.puom.fillna("PAC") == "PAC") & (R.ING_UOM != "PAC"), R.pack.fillna(1), 1.0), index=R.index)
    S = pd.DataFrame(index=pd.Index(leaves))
    z = lambda s: s.reindex(S.index).fillna(0)

    oh = items["onhand"].copy()
    oh = oh[oh.ORGANIZATION_CODE == "EM"]
    oh["q"] = pd.to_numeric(oh.TOTAL_QOH, errors="coerce") * oh.ITEM.map(f)
    oh["exp"] = pd.to_datetime(oh.EXPIRATION_DATE, errors="coerce")
    live = oh.exp.isna() | (oh.exp >= today)
    ok = live & (oh.STATUS_CODE != "Q")
    S["oh_ok"] = z(oh[ok].groupby("ITEM").q.sum())
    S["oh_q"] = z(oh[live & (oh.STATUS_CODE == "Q")].groupby("ITEM").q.sum())
    S["oh_exp"] = z(oh[~live].groupby("ITEM").q.sum())
    S["exp_short"] = z(oh[ok & (oh.exp < h_s)].groupby("ITEM").q.sum())
    S["exp_fwd"] = z(oh[ok & (oh.exp < h_l)].groupby("ITEM").q.sum())
    S["exp_min"] = oh[ok & (oh.q > 0)].groupby("ITEM").exp.min().reindex(S.index)

    pmap = supply.drop_duplicates("Order_Number").set_index("Order_Number").ITEM
    A = items["ascp"].copy()
    A = A[A.ORGANIZATION == "EM"]
    A["q"] = pd.to_numeric(A.QUANTITY, errors="coerce") * A.ITEM.map(f)
    A["due"] = pd.to_datetime(A.NEW_DUE_DATE, errors="coerce")
    A["chain"] = A.Order_Number.map(pmap).isin(chain)
    d = A[A.ORDER_TYPE_TEXT.isin(["Work order demand", "Planned order demand", "Forecast"])]
    wo = d.ORDER_TYPE_TEXT == "Work order demand"
    dem = lambda m: z(-d[m].groupby("ITEM").q.sum())
    S["wo_oth"] = dem(wo & ~d.chain)
    S["wo_chain"] = dem(wo & d.chain)
    S["pl_short_oth"] = dem(~wo & ~d.chain & (d.due < h_s))
    S["pl_fwd_oth"] = dem(~wo & ~d.chain & (d.due < h_l))
    S["pl_fwd_chain"] = dem(~wo & d.chain & (d.due < h_l))
    s = A[A.ORDER_TYPE_TEXT.isin(["Purchase order", "Purchase requisition", "PO in receiving", "Intransit shipment"])]
    S["po_short"] = z(s[s.due < h_s].groupby("ITEM").q.sum())
    S["po_fwd"] = z(s[s.due < h_l].groupby("ITEM").q.sum())
    S["po_next"] = s.groupby("ITEM").due.min().reindex(S.index)
    S["plan_po_fwd"] = z(A[(A.ORDER_TYPE_TEXT == "Planned order") & (A.due < h_l)].groupby("ITEM").q.sum())

    S["bal"] = S.oh_ok - S.wo_oth - S.wo_chain - R.req
    S["bal_short"] = S.bal - S.pl_short_oth + S.po_short
    S["n_more"] = ((S.oh_ok - S.wo_oth - S.wo_chain) / R.req.replace(0, np.nan)).clip(lower=0)
    S["status"] = np.select(
        [R.req <= 0, S.bal < 0, S.bal_short < 0, S.bal < R.req],
        ["לא נדרש בדרישה הנוכחית", "חסר", "חסר תוך הטווח הקצר", "מספיק לדרישה אחת בלבד"], "תקין")
    return S


def _fmt(v, u=""):
    return f"{v:,.0f} {u}".strip() if abs(v) >= 10 else f"{v:,.2f} {u}".strip()


def _notes(R: pd.DataFrame, today) -> pd.Series:
    """Automatic remarks: expired, quarantined and soon-expiring stock, open/planned POs on shortages,
    stock tied to the targets' open orders, and in-tree alternative codes
    (same description stem, not used in the current requirement) that hold stock."""
    alt = R[(R.req <= 0) & ((R.oh_ok + R.oh_q) > 0)].copy()
    alt["stem"] = alt.INGREDIENT_DESCRIPTION.map(_stem)
    out = {}
    for i, r in R.iterrows():
        u, n = r.ING_UOM, []
        if r.oh_exp > 0:
            n.append(f"{_fmt(r.oh_exp, u)} פג תוקף – לא נספרו")
        if r.req > 0 and r.oh_q > 0:
            n.append(f"{_fmt(r.oh_q, u)} בהסגר – לא נספרו")
        if r.req > 0 and r.exp_short > 0:
            n.append(f"{_fmt(r.exp_short, u)} יפוגו בטווח הקצר (ראשון ב-{r.exp_min:%d/%m/%Y})")
        if r.req > 0 and r.status != "תקין":
            if r.wo_chain > 0 and r.oh_ok - r.wo_oth - r.wo_chain <= 1e-9:
                n.append("כל המלאי משויך להזמנות העבודה הפתוחות של היעד")
            if r.po_fwd > 0:
                n.append(f"PO פתוח {_fmt(r.po_fwd, u)}, הקרוב ב-{r.po_next:%d/%m/%Y}")
            else:
                n.append("אין PO פתוח")
            if r.plan_po_fwd > 0:
                n.append(f"ASCP מתכנן רכש {_fmt(r.plan_po_fwd, u)}")
            if r.pl_fwd_chain > 0:
                n.append(f"ASCP מתכנן עוד {_fmt(r.pl_fwd_chain, u)} ליעד")
            if r.oh_ok > 0 and r.exp_fwd >= r.oh_ok * 0.99 and pd.notna(r.exp_min):
                n.append(f"כל המלאי פג עד {r.exp_min:%m/%Y}")
            if r.status == "חסר":
                for j, a in alt[alt.stem == _stem(r.INGREDIENT_DESCRIPTION)].iterrows():
                    if j == i:
                        continue
                    t = f"קוד חלופי בעץ: {j} ({_fmt(a.oh_ok, a.ING_UOM)}"
                    if pd.notna(a.exp_min):
                        t += f", פג {a.exp_min:%d/%m/%Y}"
                    if a.oh_q > 0:
                        t += f", +{_fmt(a.oh_q, a.ING_UOM)} בהסגר"
                    n.append(t + ")")
        out[i] = "; ".join(n)
    return pd.Series(out)


# ── Excel export ────────────────────────────────────────────────

def build_workbook(R: pd.DataFrame, targets, pinfo: pd.DataFrame, batches: dict, short_months=3,
                   asof=None) -> bytes:
    """Two sheets with live formulas: 'חומרי גלם' (requirement, packs, cost —
    driven by yellow batch inputs) and 'מצב מלאי' (stock position linked to the
    first sheet's requirement)."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter as L

    FN = "Arial"
    BASE, BOLD = Font(name=FN, size=10), Font(name=FN, size=10, bold=True)
    IN = Font(name=FN, size=10, color="0000FF")
    H, HF = Font(name=FN, size=10, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F4E78")
    YEL = PatternFill("solid", fgColor="FFFF00")
    FILL = {"ייחודי": PatternFill("solid", fgColor="E2EFDA"), "מרכזי": PatternFill("solid", fgColor="DDEBF7"), "משותף": None}
    th = Side(style="thin", color="BFBFBF")
    BOX = Border(left=th, right=th, top=th, bottom=th)
    ILS, ILS2 = '₪#,##0;(₪#,##0);"-"', '₪#,##0.00;(₪#,##0.00);"-"'
    N2, PCT = '#,##0.00;(#,##0.00);"-"', '0%;(0%);"-"'
    WRAP = Alignment(wrap_text=True, vertical="top", horizontal="right")
    asof = asof or pd.Timestamp.today().strftime("%d/%m/%Y")
    nt = len(targets)

    def put(ws, r, c, v, fmt=None, font=BASE, fill=None):
        if isinstance(v, float) and np.isnan(v):
            v = None
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, pd.Timestamp):
            v = None if pd.isna(v) or v.year > 2090 else v.to_pydatetime()
        if v is pd.NaT:
            v = None
        x = ws.cell(r, c, v)
        x.font, x.border = font, BOX
        if fmt:
            x.number_format = fmt
        if fill:
            x.fill = fill
        return x

    def header(ws, row, cols, widths, height):
        for j, (c, wd) in enumerate(zip(cols, widths), 1):
            x = ws.cell(row, j, c)
            x.font, x.fill, x.border = H, HF, BOX
            x.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
            ws.column_dimensions[L(j)].width = wd
        ws.row_dimensions[row].height = height
        ws.freeze_panes = ws.cell(row + 1, 3)

    def notes(ws, row, texts, width):
        for k, t in enumerate(texts, row):
            c = ws.cell(k, 1, t)
            c.font, c.alignment = BASE, WRAP
            ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=width)
            ws.row_dimensions[k].height = 30

    wb = Workbook()
    ws = wb.active
    ws.title = "חומרי גלם"
    ws.sheet_view.rightToLeft = True
    ws["A1"] = "חומרי גלם בעץ – " + ", ".join(f"{t} {pinfo.FORMULA_DESC1.get(t, '')}" for t in targets)
    ws["A1"].font = Font(name=FN, size=14, bold=True, color="1F4E78")
    for k, t in enumerate(targets):
        r_ = 3 + k
        c_ = ws.cell(r_, 1, f"דרישה – {t} {pinfo.FORMULA_DESC1.get(t, '')}")
        c_.font, c_.alignment = BOLD, Alignment(horizontal="right")
        ws.merge_cells(start_row=r_, start_column=1, end_row=r_, end_column=2)
        x = ws.cell(r_, 3, batches.get(t, 1))
        x.font, x.fill, x.border, x.alignment = IN, YEL, BOX, Alignment(horizontal="center")
        pq, pu = pinfo.PQ.get(t), pinfo.PRODUCT_UOM.get(t)
        ws.cell(r_, 4, f"אצוות (1 אצווה = {pq:g} {pu})").font = BASE
    nr = 3 + nt + 1
    notes(ws, nr, [
        "ייחודי = מופיע רק בעץ של המק״טים הנבחרים (ומוצרי המשפחה). מרכזי = היעד צורך את רוב הכמות (לפי ASCP 12 ח'; ללא תכנון – לפי 3 שנים). "
        "משותף = משמש בעיקר מוצרים אחרים (צרכנים אחרים בעמודה G).",
        "כמויות = פיצוץ עץ המוצר (Indented_Bills) לאצווה של כל מק״ט, כולל דרך תמיסות/בופרים. כמות נדרשת = כמות לאצווה × מספר האצוות בתאים הצהובים. "
        f"מחיר = PO אחרון (לפי מספר PO), מומר ל-₪ לפי שער ה-PO; ללא PO – עלות תקן. נתונים נכון ל-{asof}.",
    ], 18 + nt)
    hr = nr + 3
    tcol = list(range(9, 9 + nt))                                  # per-batch columns
    cq, cpk, cnp, cpr, csrc, cup, ccost, cven = range(9 + nt, 17 + nt)
    cols = ["מק״ט", "תיאור", "סוג", "סיווג", "חלק היעד – קדימה (12 ח')", "חלק היעד – 3 שנים", "צרכנים אחרים", "יחידה"] + \
           [f"לאצווה {t}" for t in targets] + \
           ["כמות נדרשת", "יחידות במארז", "מארזים נדרשים", "מחיר למארז (₪)", "מקור מחיר", "מחיר ליחידה (₪)", "עלות נדרשת (₪)", "ספק"]
    header(ws, hr, cols, [12, 46, 8, 9, 10, 10, 34, 7] + [11] * nt + [11, 9, 10, 12, 20, 11, 12, 30], 45)
    rowof = {}
    for k, (i, r) in enumerate(R.iterrows(), hr + 1):
        rowof[i] = k
        fl = FILL[r.cls]
        vals = [i, r.INGREDIENT_DESCRIPTION, r.ITEM_TYPE, r.cls, r.share_fwd, r.share_3y, r.others, r.ING_UOM]
        for j, v in enumerate(vals, 1):
            put(ws, k, j, v, {5: PCT, 6: PCT}.get(j), BASE, fl)
        for j, t in zip(tcol, targets):
            put(ws, k, j, round(float(r[f"b_{t}"]), 4), N2, IN, fl)
        put(ws, k, cq, "=" + "+".join(f"{L(j)}{k}*$C${3 + n}" for n, j in enumerate(tcol)), N2, fill=fl)
        put(ws, k, cpk, r.pack, N2, IN, fl)
        put(ws, k, cnp, f'=IF({L(cpk)}{k}>0,{L(cq)}{k}/{L(cpk)}{k},"")', N2, fill=fl)
        put(ws, k, cpr, r.price, ILS2, IN, fl)
        put(ws, k, csrc, r.psrc, fill=fl)
        put(ws, k, cup, f'=IF(AND({L(cpk)}{k}>0,{L(cpr)}{k}<>""),{L(cpr)}{k}/{L(cpk)}{k},"")', ILS2, fill=fl)
        put(ws, k, ccost, f'=IF(AND({L(cnp)}{k}<>"",{L(cpr)}{k}<>""),{L(cnp)}{k}*{L(cpr)}{k},"")', ILS, fill=fl)
        put(ws, k, cven, r.vendor, fill=fl)
        ws.cell(k, 2).alignment = ws.cell(k, 7).alignment = Alignment(wrap_text=True)
    last = hr + len(R)
    for n, cl in enumerate(CLASSES + [None], last + 2):
        put(ws, n, 2, f"סה\"כ {cl}" if cl else "סה\"כ כל חומרי הגלם", font=BOLD)
        rng = f"{L(ccost)}{hr + 1}:{L(ccost)}{last}"
        put(ws, n, ccost, f'=SUMIF($D${hr + 1}:$D${last},"{cl}",{rng})' if cl else f"=SUM({rng})", ILS, BOLD)
    ws.auto_filter.ref = f"A{hr}:{L(len(cols))}{last}"

    # ── stock sheet ──
    M = f"'{ws.title}'"
    w2 = wb.create_sheet("מצב מלאי")
    w2.sheet_view.rightToLeft = True
    w2["A1"] = "מצב מלאי – אחרי הייצור הקיים והדרישה (לפי התאים הצהובים בגיליון חומרי גלם)"
    w2["A1"].font = Font(name=FN, size=14, bold=True, color="1F4E78")
    sm = f"{short_months} ח'"
    notes(w2, 2, [
        f"נתונים: מלאי Oracle (BTG_OnHand_V, ארגון EM) ו-ASCP, נכון ל-{asof}. כל הכמויות ביחידת העץ (KG / L / PCS…), לא במארזים. "
        "מלאי מאושר = כל סטטוס חוץ מהסגר (Q), ללא אצוות שפג תוקפן.",
        "ייצור קיים = הזמנות עבודה פתוחות ב-ASCP (היתרה שטרם הונפקה), כולל הזמנות פתוחות של היעד עצמו – הדרישה בעמודה K היא תוספת מעבר להן.",
        f"יתרה = F − I − J − K. יתרה צפויה {sm} = מוסיפה PO פתוחים ומורידה ביקוש מתוכנן (Planned/Forecast) של מוצרים אחרים עד {sm}.",
    ], 22)
    h2 = 6
    c2 = ["מק״ט", "תיאור", "סיווג", "יחידה", "יחידות במארז", "מלאי מאושר בתוקף", "בהסגר (Q)", "פג תוקף (לא נספר)",
          "הזמנות עבודה פתוחות – מוצרים אחרים", "הזמנות עבודה פתוחות – היעד", "דרישה (מגיליון 1)",
          "יתרה אחרי ייצור קיים ודרישה", "יתרה במארזים", "מספיק לכמה דרישות כאלה",
          f"ביקוש מתוכנן מוצרים אחרים – {sm}", f"PO פתוחים עד {sm}", f"יתרה צפויה בעוד {sm}", "PO פתוח הבא",
          "תפוגה קרובה", f"כמות שתפוג תוך {sm}", "סטטוס", "הערה"]
    header(w2, h2, c2, [12, 40, 8, 7, 8, 11, 10, 10, 12, 12, 11, 12, 10, 10, 12, 10, 11, 11, 11, 10, 20, 50], 58)
    for k, (i, r) in enumerate(R.iterrows(), h2 + 1):
        fl, mk = FILL[r.cls], rowof[i]
        p = lambda c, v, fmt=None, font=BASE: put(w2, k, c, v, fmt, font, fl)
        p(1, f"={M}!A{mk}"); p(2, f"={M}!B{mk}"); p(3, f"={M}!D{mk}"); p(4, f"={M}!H{mk}")
        p(5, f"={M}!{L(cpk)}{mk}", N2)
        for c, col in [(6, "oh_ok"), (7, "oh_q"), (8, "oh_exp"), (9, "wo_oth"), (10, "wo_chain"),
                       (15, "pl_short_oth"), (16, "po_short"), (20, "exp_short")]:
            p(c, round(float(r[col]), 3), N2, IN)
        p(11, f"={M}!{L(cq)}{mk}", N2)
        p(12, f"=F{k}-I{k}-J{k}-K{k}", N2)
        p(13, f'=IF(E{k}>0,L{k}/E{k},"")', N2)
        p(14, f'=IF(K{k}>0,MAX(0,(F{k}-I{k}-J{k})/K{k}),"")', '0.0;-0.0;"-"')
        p(17, f"=L{k}-O{k}+P{k}", N2)
        p(18, r.po_next, "dd/mm/yyyy"); p(19, r.exp_min, "dd/mm/yyyy")
        p(21, f'=IF(K{k}=0,"לא נדרש בדרישה הנוכחית",IF(L{k}<0,"חסר",IF(Q{k}<0,"חסר תוך הטווח הקצר",'
              f'IF(L{k}<K{k},"מספיק לדרישה אחת בלבד","תקין"))))')
        p(22, r.notes)
        w2.cell(k, 2).alignment = w2.cell(k, 22).alignment = Alignment(wrap_text=True)
    sl = h2 + len(R)
    rng = f"U{h2 + 1}:U{sl}"
    for txt, col in [("חסר", "FFC7CE"), ("חסר תוך הטווח הקצר", "FFEB9C"), ("מספיק לדרישה אחת בלבד", "FFF2CC")]:
        w2.conditional_formatting.add(rng, FormulaRule(formula=[f'$U{h2 + 1}="{txt}"'],
                                                       fill=PatternFill("solid", fgColor=col), font=Font(name=FN, bold=True)))
    for n, lbl in enumerate(["חסר", "חסר תוך הטווח הקצר", "מספיק לדרישה אחת בלבד", "תקין"], sl + 2):
        w2.cell(n, 2, f"מספר פריטים – {lbl}").font = BOLD
        put(w2, n, 3, f'=COUNTIF({rng},"{lbl}")', font=BOLD)
    w2.auto_filter.ref = f"A{h2}:{L(len(c2))}{sl}"

    wb.calculation.fullCalcOnLoad = True
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

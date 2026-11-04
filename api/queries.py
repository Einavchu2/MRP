"""
queries.py — the actual DB access for the data API, moved from src/db.py.

Same queries, same logic (including the bulk-fetch-then-explode BOM fix from
earlier this session) — the only real change is a pooled SQLAlchemy engine
instead of a fresh pyodbc.connect() per call, and results going out as
JSON-safe records instead of DataFrames.
"""
import re

import pandas as pd

from .cache import ttl_cache
from .config import get_dwh_engine


def _df_to_records(df: pd.DataFrame) -> list:
    """NaN -> None explicitly, so the JSON response uses standard `null`
    rather than relying on Python's non-standard NaN literal."""
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


# ── BOM ──────────────────────────────────────────────────────────

def _fetch_indented_bills(engine) -> pd.DataFrame:
    query = """
        SELECT DISTINCT
            ib.PRODUCT,
            ib.FORMULA_DESC1,
            CAST(ib.PRUDUCT_QTY AS FLOAT) AS PRUDUCT_QTY,
            ib.PRODUCT_UOM,
            ib.INGREDIENT,
            ib.INGREDIENT_DESCRIPTION,
            CAST(ib.ING_QTY AS FLOAT) AS ING_QTY,
            ib.ING_UOM,
            ISNULL(ib.PATH, '') AS PATH
        FROM Indented_Bills ib
        WHERE TRY_CAST(ib.ING_QTY AS FLOAT) IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df["PRODUCT"] = df["PRODUCT"].astype(str).str.strip()
    df["INGREDIENT"] = df["INGREDIENT"].astype(str).str.strip()
    return df


@ttl_cache(300)
def _indented_bills_cached() -> pd.DataFrame:
    return _fetch_indented_bills(get_dwh_engine())


def explode_bom(bom_all: pd.DataFrame, root_product: str) -> pd.DataFrame:
    """Recursive BOM explosion from one already-fetched full edge list — pure
    in-memory tree walk, no per-node SQL calls."""
    by_product = {p: g for p, g in bom_all.groupby("PRODUCT")}
    visited: set = set()
    levels = []

    def walk(product: str, level: int):
        if product in visited:
            return
        visited.add(product)
        g = by_product.get(product)
        if g is None or g.empty:
            return
        g = g.copy()
        g["level"] = level
        levels.append(g)
        for item in g["INGREDIENT"].unique():
            walk(item, level + 1)

    walk(str(root_product).strip(), 0)
    if not levels:
        return pd.DataFrame()
    return pd.concat(levels, ignore_index=True)


def get_bom(root_product: str) -> list:
    bom_all = _indented_bills_cached()
    exploded = explode_bom(bom_all, root_product)
    return _df_to_records(exploded)


# ── TRANSACTIONS + ASCP + UOM ──────────────────────────────────────

@ttl_cache(300)
def get_dwh_data() -> dict:
    engine = get_dwh_engine()

    df_trx = pd.read_sql(
        "SELECT material_Transactions.item, material_Transactions.description, "
        "material_Transactions.reason_name, "
        "material_Transactions.primary_quantity, material_Transactions.transaction_date, "
        "(MasterItemsBI.rocessing_lead_time/30) as LT, "
        "(MasterItemsBI.shelf_life_days/30) as SL, "
        "MasterItemsBI.planner_code, MasterItemsBI.list_price "
        "FROM material_Transactions "
        "LEFT JOIN MasterItemsBI ON material_Transactions.item = MasterItemsBI.item_number "
        "WHERE material_Transactions.organization_code = 'EM' "
        "AND material_Transactions.Transaction_Type_Name IN "
        "  ('WIP Issue', 'Account Alias Issue', 'Move Order Issue') "
        "AND MasterItemsBI.organization_code = 'EM' "
        "AND planner_code IN ('B_RM2', 'B_RM1') "
        # NULL never equals anything in SQL's three-valued logic — "= NULL" would
        # silently match zero rows; "IS NULL" is the correct comparison.
        "AND material_Transactions.reason_name IS NULL",
        engine,
    )

    df_ascp = pd.read_sql(
        "SELECT * FROM ASCPDemandSupplyBI WHERE planner_code IN ('B_RM2', 'B_RM1')",
        engine,
    )

    df_uom = pd.read_sql(
        "SELECT DISTINCT item, DESCRIPTION, PRIMARY_UOM_CODE, UOM_CODE, CONVERSION_RATE "
        "FROM BTG_Conversion_Rates",
        engine,
    )

    return {
        "transactions": _df_to_records(df_trx),
        "ascp": _df_to_records(df_ascp),
        "uom": _df_to_records(df_uom),
    }


# ── INVENTORY HISTORY (Holding Cost analysis) ───────────────────────

# Every org EM item, any planner code — the page filters RM1 / RM2 / other
# itself (src/holding_cost.py: filter_history).
_EM_ITEMS_SQL = "SELECT ITEM_NUMBER FROM MasterItemsBI WHERE ORGANIZATION_CODE = 'EM'"


# Material_Transactions contains exact duplicate rows (same TRANSACTION_ID,
# every column identical) — 514 of 44,663 rows since 2025-06 when checked on
# 2026-10-29. DISTINCT on the identifying columns removes them; without this,
# rolling on-hand back from today drifts (validated against
# BTG_Cover_Monts_Weekly_History 2025-06-25: 74% of items within 1% with
# duplicates, 94% after removing them).
_DEDUP_TRX_SQL = """
    SELECT DISTINCT mt.TRANSACTION_ID, mt.ITEM, mt.TRANSACTION_TYPE_NAME,
           mt.REASON_NAME, TRY_CAST(mt.PRIMARY_QUANTITY AS FLOAT) AS QTY,
           mt.TRANSACTION_DATE
    FROM Material_Transactions mt
    WHERE mt.ORGANIZATION_CODE = 'EM'
      AND mt.TRANSACTION_DATE >= ?
      AND mt.ITEM IN ({items})
"""


@ttl_cache(300)
def get_inventory_history(days: int) -> dict:
    """
    Raw inputs for the Inventory Holding Cost analysis — reconstruction and
    every calculation happen in src/holding_cost.py, this only fetches.

      onhand       current total on-hand per item, all subinventories (BTG_OnHand_V)
      daily_net    net signed qty per item per day, ALL transaction types,
                   deduplicated — used to roll today's on-hand backwards
      daily_issue  consumption per item per day: same definition the MRP pivot
                   uses (WIP Issue / Account alias issue / Move Order Issue,
                   REASON_NAME IS NULL), deduplicated
      cost         Oracle item cost, current period, org EM (ILS)
      items        description + ASCP Items Category from MasterItemsBI
      fwd_demand   total forward ASCP demand per item (Forecast, Planned/Work
                   order demand) — used only to support an "obsolete" flag
    """
    engine = get_dwh_engine()
    since = (pd.Timestamp.today().normalize() - pd.Timedelta(days=int(days) + 1)).to_pydatetime()
    trx_sql = _DEDUP_TRX_SQL.format(items=_EM_ITEMS_SQL)

    onhand = pd.read_sql(
        "SELECT ITEM, SUM(TOTAL_QOH) AS ONHAND_QTY, MAX(UpdateDate) AS ONHAND_DATE "
        f"FROM BTG_OnHand_V WHERE ORGANIZATION_CODE = 'EM' AND ITEM IN ({_EM_ITEMS_SQL}) "
        "GROUP BY ITEM",
        engine,
    )

    daily_net = pd.read_sql(
        f"SELECT ITEM, CAST(TRANSACTION_DATE AS date) AS TRX_DAY, SUM(QTY) AS NET_QTY "
        f"FROM ({trx_sql}) t GROUP BY ITEM, CAST(TRANSACTION_DATE AS date)",
        engine, params=(since,),
    )

    daily_issue = pd.read_sql(
        f"SELECT ITEM, CAST(TRANSACTION_DATE AS date) AS TRX_DAY, SUM(-QTY) AS ISSUE_QTY "
        f"FROM ({trx_sql}) t "
        "WHERE TRANSACTION_TYPE_NAME IN ('WIP Issue', 'Account alias issue', 'Move Order Issue') "
        "AND REASON_NAME IS NULL "
        "GROUP BY ITEM, CAST(TRANSACTION_DATE AS date)",
        engine, params=(since,),
    )

    cost = pd.read_sql(
        "SELECT ITEM, TRY_CAST(ITEM_COST AS FLOAT) AS ITEM_COST, PERIOD_CODE "
        "FROM Items_Cost_CurrentPeriod_V "
        f"WHERE ORGANIZATION_CODE = 'EM' AND ITEM IN ({_EM_ITEMS_SQL})",
        engine,
    )

    items = pd.read_sql(
        "SELECT ITEM_NUMBER AS ITEM, DESCRIPTION, PLANNER_CODE, PRIMARY_UOM_CODE, LIST_PRICE, "
        "[ASCP Items Category] AS CATEGORY "
        "FROM MasterItemsBI WHERE ORGANIZATION_CODE = 'EM'",
        engine,
    )

    fwd_demand = pd.read_sql(
        "SELECT ITEM, SUM(ABS(QUANTITY_RATE)) AS FWD_DEMAND_QTY FROM ASCPDemandSupplyBI "
        "WHERE ORDER_TYPE_TEXT IN ('Planned order demand', 'Work order demand', 'Forecast') "
        "GROUP BY ITEM",
        engine,
    )

    for df in (onhand, daily_net, daily_issue, cost, items, fwd_demand):
        df["ITEM"] = df["ITEM"].astype(str).str.strip()
    for df in (daily_net, daily_issue):
        df["TRX_DAY"] = df["TRX_DAY"].astype(str)
    onhand["ONHAND_DATE"] = onhand["ONHAND_DATE"].astype(str)

    return {
        "onhand": _df_to_records(onhand),
        "daily_net": _df_to_records(daily_net),
        "daily_issue": _df_to_records(daily_issue),
        "cost": _df_to_records(cost),
        "items": _df_to_records(items),
        "fwd_demand": _df_to_records(fwd_demand),
    }


# ── SUBSTITUTE ITEMS ────────────────────────────────────────────────

@ttl_cache(300)
def get_substitutes() -> list:
    """Main_Item -> Substitute_Item mapping. Empty list on failure, same
    defensive behavior as the original direct-DB version."""
    try:
        engine = get_dwh_engine()
        df = pd.read_sql(
            "SELECT Main_Item, Substitute_Item FROM BTG_Substitute_Items",
            engine,
        )
        df["Main_Item"] = df["Main_Item"].astype(str).str.strip()
        df["Substitute_Item"] = df["Substitute_Item"].astype(str).str.strip()
        return _df_to_records(df)
    except Exception:
        return []


# ── RM ANALYSIS (src/rm_analysis.py) ────────────────────────────────
# Raw inputs only; classification, requirement, prices and the stock
# position are all computed in src/rm_analysis.py.

_ITEM_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def _in_list(items) -> str:
    """Literal IN-list for item codes — validated, since they arrive from the
    query string and pyodbc caps bound parameters at 2,100."""
    items = sorted({str(i).strip() for i in items if str(i).strip()})
    bad = [i for i in items if not _ITEM_RE.match(i)]
    if bad:
        raise ValueError(f"Invalid item codes: {bad[:5]}")
    return ",".join(f"'{i}'" for i in items) or "''"


@ttl_cache(300)
def get_rm_bom() -> list:
    """Every Indented_Bills edge, with the columns the RM analysis needs
    (ITEM_TYPE, batch size PQ, vendor) that the simulation's BOM query omits."""
    df = pd.read_sql(
        """SELECT DISTINCT PRODUCT, FORMULA_DESC1, FORMULA_VERS, INGREDIENT, INGREDIENT_DESCRIPTION,
                  ITEM_TYPE, ING_UOM, TRY_CAST(ING_QTY AS FLOAT) AS ING_QTY,
                  TRY_CAST(PRUDUCT_QTY AS FLOAT) AS PQ, PRODUCT_UOM, ASL_VENDOR_NAME
           FROM Indented_Bills""",
        get_dwh_engine(),
    )
    for c in ("PRODUCT", "INGREDIENT"):
        df[c] = df[c].astype(str).str.strip()
    return _df_to_records(df)


@ttl_cache(300)
def get_rm_plan() -> dict:
    """Production plan and history for the whole site:
      supply   ASCP EM work orders + planned orders (all items) — the forward
               plan, and Order_Number -> parent item for RM demand rows
      wip3y    actual ingredient consumption per formula and item, last 3
               years (WIP_Last_3_Years, one row per batch/item/line)"""
    engine = get_dwh_engine()
    supply = pd.read_sql(
        "SELECT ITEM, ORDER_TYPE_TEXT, Order_Number, QUANTITY, NEW_DUE_DATE FROM ASCPDemandSupplyBI "
        "WHERE ORGANIZATION = 'EM' AND ORDER_TYPE_TEXT IN ('Planned order', 'Work order')",
        engine,
    )
    wip = pd.read_sql(
        """SELECT FORMULA_NO, ITEM, SUM(ACTUAL_QTY) AS ACTUAL_QTY
           FROM (SELECT FORMULA_NO, BATCH_ID, ITEM, LINE_NO, MAX(ACTUAL_QTY) AS ACTUAL_QTY
                 FROM WIP_Last_3_Years WHERE LINE_TYPE = 'Ingridient'
                 GROUP BY FORMULA_NO, BATCH_ID, ITEM, LINE_NO) t
           GROUP BY FORMULA_NO, ITEM""",
        engine,
    )
    supply["ITEM"] = supply["ITEM"].astype(str).str.strip()
    supply["NEW_DUE_DATE"] = supply["NEW_DUE_DATE"].astype(str)
    for c in ("FORMULA_NO", "ITEM"):
        wip[c] = wip[c].astype(str).str.strip()
    return {"supply": _df_to_records(supply), "wip3y": _df_to_records(wip)}


@ttl_cache(300)
def get_rm_items(items: tuple) -> dict:
    """Item-level data for the RMs of the analysed trees:
      po      every non-cancelled PO line (latest picked by PO_NO downstream —
              PO_CREATION_DATE is often empty)
      cost    standard cost, current period, all orgs
      master  primary UOM (org EM)
      onhand  lot-level on-hand with status and expiry
      ascp    every ASCP row for these items (on hand, POs, demand, forecast)"""
    engine = get_dwh_engine()
    ins = _in_list(items)
    po = pd.read_sql(
        f"SELECT [Item Number] AS ITEM, PO_NO, LN, CURR, RATE, LN_UOM, LN_U_PRICE, VENDOR_NAME, LN_CANCEL_F "
        f"FROM BTG_PO_V WHERE [Item Number] IN ({ins})",
        engine,
    )
    cost = pd.read_sql(
        f"SELECT ITEM, TRY_CAST(ITEM_COST AS FLOAT) AS ITEM_COST, ORGANIZATION_CODE "
        f"FROM Items_Cost_CurrentPeriod_V WHERE ITEM IN ({ins})",
        engine,
    )
    master = pd.read_sql(
        f"SELECT ITEM_NUMBER AS ITEM, PRIMARY_UOM_CODE FROM MasterItemsBI "
        f"WHERE ORGANIZATION_CODE = 'EM' AND ITEM_NUMBER IN ({ins})",
        engine,
    )
    onhand = pd.read_sql(
        f"SELECT ORGANIZATION_CODE, ITEM, LOT, TOTAL_QOH, PRIMARY_UOM_CODE, SUBINVENTORY_CODE, STATUS_CODE, "
        f"EXPIRATION_DATE FROM BTG_OnHand_V WHERE ITEM IN ({ins})",
        engine,
    )
    ascp = pd.read_sql(
        f"SELECT ITEM, ORGANIZATION, ORDER_TYPE_TEXT, QUANTITY, NEW_DUE_DATE, Order_Number "
        f"FROM ASCPDemandSupplyBI WHERE ITEM IN ({ins})",
        engine,
    )
    for df in (po, cost, master, onhand, ascp):
        df["ITEM"] = df["ITEM"].astype(str).str.strip()
    onhand["EXPIRATION_DATE"] = onhand["EXPIRATION_DATE"].astype(str)
    ascp["NEW_DUE_DATE"] = ascp["NEW_DUE_DATE"].astype(str)
    return {k: _df_to_records(v) for k, v in
            {"po": po, "cost": cost, "master": master, "onhand": onhand, "ascp": ascp}.items()}

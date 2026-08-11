"""
db.py  –  Database connections & data loaders
Credentials taken directly from model_sim.ipynb
"""

import pandas as pd
import pyodbc
import streamlit as st

# ─────────────────────────────────────────────
# CONNECTION STRINGS  (from notebook)
# ─────────────────────────────────────────────

CONN_STR_SCXL = (
    r"DRIVER={ODBC Driver 18 for SQL Server};"
    r"SERVER=FEIL1061\BI;"
    r"DATABASE=SupplyChainXL;"
    r"UID=SCXL_User;"
    r"PWD=wG-85rT&S6{t$Pg4;"
    r"TrustServerCertificate=yes;"
)

CONN_STR_DWH = (
    r"DRIVER={ODBC Driver 18 for SQL Server};"
    r"SERVER=FEIL1061\BI;"
    r"DATABASE=ORACLE_DWH;"
    r"UID=DWH_Viewer;"
    r"PWD=7Hc+=m2Dg9F_B-;"
    r"TrustServerCertificate=yes;"
)


# ─────────────────────────────────────────────
# BOM  (SupplyChainXL)
# ─────────────────────────────────────────────

def get_bom_recursive(product: str, conn, level: int = 0, visited: set = None) -> pd.DataFrame:
    """Recursive BOM explosion from Indented_Bills."""
    if visited is None:
        visited = set()
    if product in visited:
        return pd.DataFrame()
    visited.add(product)

    # Try to fetch PATH + CONVERSION_RATE; fall back gracefully if columns missing
    query = f"""
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
        WHERE ib.product = '{product}'
          AND TRY_CAST(ib.ING_QTY AS FLOAT) IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df

    df["level"] = level
    all_data = [df]
    for item in df["INGREDIENT"].unique():
        child = get_bom_recursive(item, conn, level + 1, visited)
        if not child.empty:
            all_data.append(child)
    return pd.concat(all_data, ignore_index=True)


@st.cache_data(ttl=300, show_spinner="Loading BOM from ORACLE_DWH…")
def load_full_bom(root_product: str) -> pd.DataFrame:
    # Notebook calls get_bom() with conn_trx (ORACLE_DWH), not SupplyChainXL
    conn = pyodbc.connect(CONN_STR_DWH)
    bom = get_bom_recursive(root_product, conn)
    conn.close()
    return bom


# ─────────────────────────────────────────────
# TRANSACTIONS + ASCP + UOM  (ORACLE_DWH)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading transactions & ASCP from ORACLE_DWH…")
def load_dwh_data() -> dict:
    conn = pyodbc.connect(CONN_STR_DWH)

    df_trx = pd.read_sql(
        "SELECT material_Transactions.item, material_Transactions.description, "
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
        "AND planner_code IN ('B_RM2', 'B_RM1')",
        conn,
    )

    df_ascp = pd.read_sql(
        "SELECT * FROM ASCPDemandSupplyBI WHERE planner_code IN ('B_RM2', 'B_RM1')",
        conn,
    )

    df_uom = pd.read_sql(
        "SELECT DISTINCT item, DESCRIPTION, PRIMARY_UOM_CODE, UOM_CODE, CONVERSION_RATE "
        "FROM BTG_Conversion_Rates",
        conn,
    )

    conn.close()
    return {"transactions": df_trx, "ascp": df_ascp, "uom": df_uom}


# ─────────────────────────────────────────────
# SUBSTITUTE ITEMS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading substitute items…")
def load_substitutes() -> pd.DataFrame:
    """
    Load Main_Item → Substitute_Item mapping from BTG_Substitute_Items.
    Returns DataFrame with columns: Main_Item, Substitute_Item
    """
    try:
        conn = pyodbc.connect(CONN_STR_DWH)
        df = pd.read_sql(
            "SELECT Main_Item, Substitute_Item FROM BTG_Substitute_Items",
            conn,
        )
        conn.close()
        df["Main_Item"]       = df["Main_Item"].astype(str).str.strip()
        df["Substitute_Item"] = df["Substitute_Item"].astype(str).str.strip()
        return df
    except Exception as e:
        return pd.DataFrame(columns=["Main_Item", "Substitute_Item"])

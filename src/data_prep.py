"""
data_prep.py  –  BOM enrichment + MRP pivot builder
Logic follows model_sim.ipynb exactly.
Outputs: full_bom (bom_with_conversion.xlsx) + pivot_df (mrp_output.xlsx)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# STEP 1 – enrich BOM with UOM conversions
# ─────────────────────────────────────────────

def enrich_bom(bom_raw: pd.DataFrame, df_uom: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BOM with UOM conversion rates → adds BOM_CONV column.
    Saves bom_with_conversion.xlsx in the outputs folder.
    """
    df_bom = bom_raw.copy()
    df_uom = df_uom.copy()

    df_bom.columns = df_bom.columns.str.strip()
    df_uom.columns = df_uom.columns.str.strip()

    df_bom["INGREDIENT"] = df_bom["INGREDIENT"].astype(str)
    df_bom["ING_UOM"] = df_bom["ING_UOM"].astype(str)
    df_uom["ITEM"] = df_uom["item"].astype(str)
    df_uom["UOM_CODE"] = df_uom["UOM_CODE"].astype(str)

    df_merged = df_bom.merge(
        df_uom,
        left_on=["INGREDIENT", "ING_UOM"],
        right_on=["ITEM", "UOM_CODE"],
        how="left",
    )

    df_merged["ING_QTY"]         = pd.to_numeric(df_merged["ING_QTY"],         errors="coerce")
    df_merged["CONVERSION_RATE"] = pd.to_numeric(df_merged["CONVERSION_RATE"], errors="coerce")

    # ── BOM Conversion formula ────────────────────────────────────────────
    # In Oracle Indented_Bills, ING_QTY is ALREADY per 1 unit of PRODUCT.
    # PRUDUCT_QTY is a reference batch size — NOT a divisor for ING_QTY.
    #
    # Correct formula:
    #   BOM_CONV = ING_QTY × CONVERSION_RATE
    #
    # Example:
    #   ING_QTY = 0.5, ING_UOM = KG, CONVERSION_RATE = 1.0
    #   → BOM_CONV = 0.5 KG per 1 unit of parent
    #   → For production_change=4: qty_change = 0.5 × 4 = 2.0  ✓

    df_merged["BOM_CONV"] = df_merged["ING_QTY"] * df_merged["CONVERSION_RATE"]

    # Fallback: no CONVERSION_RATE → use ING_QTY as-is
    no_conv = df_merged["CONVERSION_RATE"].isna()
    df_merged.loc[no_conv, "BOM_CONV"] = df_merged.loc[no_conv, "ING_QTY"]

    df_merged["CONV_STATUS"] = df_merged["CONVERSION_RATE"].apply(
        lambda x: "OK" if pd.notna(x) else "Missing Conversion (using ING_QTY as-is)"
    )

    # Audit: keep PRUDUCT_QTY for reference only (not used in calculation)
    if "PRUDUCT_QTY" in df_merged.columns:
        df_merged["PRUDUCT_QTY"] = pd.to_numeric(df_merged["PRUDUCT_QTY"], errors="coerce")

    df_merged.to_excel("outputs/bom_with_conversion.xlsx", index=False, engine="openpyxl")
    return df_merged


# ─────────────────────────────────────────────
# STEP 2 – build MRP pivot
# ─────────────────────────────────────────────

def build_mrp_pivot(df_trx: pd.DataFrame, df_ascp: pd.DataFrame) -> pd.DataFrame:
    """
    Build the MRP pivot from transactions + ASCP.
    Saves mrp_output.xlsx in the outputs folder.
    """
    # ── Actual transactions ──────────────────
    df = df_trx.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month_year"] = df["transaction_date"].dt.strftime("%Y-%m")
    df = df.sort_values("month_year")
    df["primary_quantity"] = (
        pd.to_numeric(df["primary_quantity"], errors="coerce").fillna(0).abs()
    )
    result = df.groupby(
        ["item", "description", "month_year", "LT", "SL", "planner_code", "list_price"],
        as_index=False,
    ).agg({"primary_quantity": "sum"})

    # ── ASCP ────────────────────────────────
    df_a = df_ascp.copy()
    df_a["primary_quantity"] = (
        pd.to_numeric(df_a["QUANTITY_RATE"], errors="coerce").fillna(0).abs()
    )
    df_a["NEW_DUE_DATE"] = pd.to_datetime(df_a["NEW_DUE_DATE"]).dt.strftime("%Y-%m")
    df_a = df_a.sort_values("NEW_DUE_DATE")

    # Demand / Forecast
    df_ascp_f = df_a[
        df_a["ORDER_TYPE_TEXT"].isin(["Planned order demand", "Work order demand", "Forecast"])
    ]
    result_ascp = (
        df_ascp_f.groupby(["ITEM", "NEW_DUE_DATE"], as_index=False)
        .agg({"primary_quantity": "sum"})
        .rename(columns={"NEW_DUE_DATE": "month_year", "ITEM": "item"})
    )

    # On Hand
    result_onhand = (
        df_a[df_a["ORDER_TYPE_TEXT"] == "On Hand"]
        .groupby(["ITEM", "NEW_DUE_DATE", "ORDER_TYPE_TEXT"], as_index=False)
        .agg({"primary_quantity": "sum"})
        .rename(columns={"NEW_DUE_DATE": "month_year", "ITEM": "item"})
    )

    # Purchase Orders
    result_po = (
        df_a[df_a["ORDER_TYPE_TEXT"] == "Purchase order"]
        .groupby(["ITEM", "NEW_DUE_DATE", "ORDER_TYPE_TEXT"], as_index=False)
        .agg({"primary_quantity": "sum"})
        .rename(columns={"NEW_DUE_DATE": "month_year", "ITEM": "item"})
    )

    # ── Combine ──────────────────────────────
    df_all = pd.concat([result, result_ascp, result_onhand, result_po], ignore_index=True)
    df_all = df_all.sort_values("month_year")

    # ── Classify ORDER_TYPE_FINAL ────────────
    df_all["ORDER_TYPE_TEXT"] = df_all["ORDER_TYPE_TEXT"].astype(object)
    df_all["description"] = df_all["description"].astype(str)

    df_all["ORDER_TYPE_TEXT"] = np.where(
        df_all["ORDER_TYPE_TEXT"].notna(),
        df_all["ORDER_TYPE_TEXT"],
        np.where(df_all["LT"].isna(), "demand", "ACTUAL"),
    )

    def classify_order_type(row):
        if pd.notna(row["ORDER_TYPE_TEXT"]):
            return row["ORDER_TYPE_TEXT"]
        if "on hand" in str(row["description"]).lower():
            return "3.On Hand"
        return "Other"

    df_all["ORDER_TYPE_FINAL"] = df_all.apply(classify_order_type, axis=1)

    mapping = {
        "Forecast": "1.Forecast",
        "On Hand": "3.On Hand",
        "demand": "1.Forecast",
        "ACTUAL": "2.ACTUAL",
    }
    df_all["ORDER_TYPE_FINAL"] = df_all["ORDER_TYPE_FINAL"].replace(mapping)

    # ── Pivot ────────────────────────────────
    df_all["month_year"] = pd.to_datetime(df_all["month_year"], errors="coerce")
    df_all["month_year_str"] = df_all["month_year"].dt.strftime("%Y-%m")
    df_all["primary_quantity"] = pd.to_numeric(df_all["primary_quantity"], errors="coerce")

    pivot_df = df_all.pivot_table(
        index=["item", "ORDER_TYPE_FINAL"],
        columns="month_year_str",
        values="primary_quantity",
        aggfunc="sum",
    ).reset_index()

    order_sort = {
        "1.Forecast": 1, "2.ACTUAL": 2, "Planned order demand": 3,
        "Work order demand": 4, "Expired lot": 5, "Planned order": 6,
        "Issue PR": 7, "PO Number": 8, "Purchase order": 9,
        "Purchase requisition": 10, "3.On Hand": 11, "Other": 99,
    }
    pivot_df["sort_key"] = pivot_df["ORDER_TYPE_FINAL"].map(order_sort)
    pivot_df = pivot_df.sort_values(["item", "sort_key"]).drop(columns="sort_key")

    desc_map = df_all.dropna(subset=["description"]).groupby("item")["description"].first()
    pivot_df["description"] = pivot_df["item"].map(desc_map)

    cols = ["item", "description", "ORDER_TYPE_FINAL"] + [
        c for c in pivot_df.columns if c not in ["item", "description", "ORDER_TYPE_FINAL"]
    ]
    pivot_df = pivot_df[cols]

    pivot_df.to_excel("outputs/mrp_output.xlsx", index=False)
    return pivot_df

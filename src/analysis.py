"""
analysis.py  –  ML-powered supply chain analysis
Models: Isolation Forest (anomaly), Linear regression (trend), Risk scoring
"""
import numpy as np
import pandas as pd


def extract_item_series(pivot, row_type):
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    rows = pivot[pivot["ORDER_TYPE_FINAL"].str.contains(row_type, case=False, na=False)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["item", "description", "month", "value"])
    melted = rows.melt(
        id_vars=["item", "description"],
        value_vars=[c for c in month_cols if c in rows.columns],
        var_name="month", value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted["month"] = pd.to_datetime(melted["month"])
    return melted.dropna(subset=["value"]).sort_values(["item", "month"])


def detect_anomalies(pivot, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    actual_df = extract_item_series(pivot, "actual")
    fcst_df   = extract_item_series(pivot, "forecast")
    combined  = pd.concat([actual_df, fcst_df], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    results = []
    for item, grp in combined.groupby("item"):
        if len(grp) < 4:
            continue
        vals   = grp["value"].values.reshape(-1, 1)
        clf    = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(vals)
        scores = clf.score_samples(vals)
        labels = clf.predict(vals)
        grp = grp.copy()
        grp["anomaly"]       = labels == -1
        grp["anomaly_score"] = -scores
        results.append(grp)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_trends(pivot):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    actual_df = extract_item_series(pivot, "actual")
    if actual_df.empty:
        return pd.DataFrame()
    rows = []
    for item, grp in actual_df.groupby("item"):
        grp = grp.sort_values("month").dropna(subset=["value"])
        if len(grp) < 3:
            continue
        x  = np.arange(len(grp)).reshape(-1, 1)
        y  = grp["value"].values
        lr = LinearRegression().fit(x, y)
        r2 = r2_score(y, lr.predict(x)) if len(y) > 1 else 0
        slope      = lr.coef_[0]
        avg        = y.mean()
        norm_slope = slope / avg if avg > 0 else 0
        if norm_slope > 0.05:    label = "📈 Growing"
        elif norm_slope < -0.05: label = "📉 Declining"
        else:                    label = "➡️ Stable"
        rows.append({
            "item": item, "description": grp["description"].iloc[0],
            "slope": round(slope, 3), "norm_slope": round(norm_slope, 3),
            "r2": round(r2, 3), "trend_label": label,
            "last_value": round(float(y[-1]), 2),
            "avg_value":  round(float(avg), 2),
            "n_months":   len(grp),
        })
    return pd.DataFrame(rows).sort_values("norm_slope")


def compute_risk(pivot, trends_df=None):
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    trend_map  = {}
    if trends_df is not None and not trends_df.empty:
        trend_map = dict(zip(trends_df["item"], trends_df["trend_label"]))
    results = []
    for item, grp in pivot.groupby("item"):
        # Use ONLY the original COVER_MONTHS row (exact match, not UPDATED)
        cover_row = grp[grp["ORDER_TYPE_FINAL"] == "COVER_MONTHS"]
        inv_row   = grp[grp["ORDER_TYPE_FINAL"] == "INV"]
        desc      = grp["description"].dropna()
        desc      = desc.iloc[0] if not desc.empty else ""

        # ── Find first On Hand month to exclude historical zeros ──
        onhand_row = grp[grp["ORDER_TYPE_FINAL"].str.contains("on hand|3.on hand", case=False, na=False)]
        first_oh_month = None
        if not onhand_row.empty:
            for m in month_cols:
                if m in onhand_row.columns:
                    v = pd.to_numeric(onhand_row[m].values, errors="coerce")
                    if len(v) > 0 and not np.isnan(v[0]) and v[0] > 0:
                        first_oh_month = m
                        break

        # Only use months from first On Hand onwards (avoids historical NaN/zeros)
        if first_oh_month and first_oh_month in month_cols:
            active_months = month_cols[month_cols.index(first_oh_month):]
        else:
            active_months = month_cols

        cover_vals = pd.to_numeric(
            cover_row[active_months].values.flatten() if not cover_row.empty and active_months else [],
            errors="coerce")
        # Remove NaN AND zeros (zeros = months with no forecast, not real coverage)
        cover_vals = cover_vals[~np.isnan(cover_vals) & (cover_vals != 0)]

        inv_vals = pd.to_numeric(
            inv_row[active_months].values.flatten() if not inv_row.empty and active_months else [],
            errors="coerce")
        inv_vals = inv_vals[~np.isnan(inv_vals)]

        if len(cover_vals) == 0:
            continue

        min_cover       = float(cover_vals.min())
        avg_cover       = float(cover_vals.mean())
        months_below_7  = int((cover_vals < 7).sum())
        months_negative = int((inv_vals < 0).sum()) if len(inv_vals) > 0 else 0
        # Volatility: only meaningful if we have ≥3 data points
        cover_std       = float(cover_vals.std()) if len(cover_vals) >= 3 else 0
        trend_label     = trend_map.get(item, "➡️ Stable")

        # ── Score components ──
        # Coverage: only penalize if min_cover is genuinely below 7
        cover_score = min(40, max(0, (7 - min_cover) / 7 * 40)) if min_cover < 7 else 0
        # Negative INV: hard signal
        neg_score   = min(30, months_negative * 8)
        # Declining trend: only penalize if combined with coverage risk
        trend_score = 20 if ("Declining" in trend_label and min_cover < 15) else 0
        # Volatility: only meaningful relative to avg
        vol_score   = min(10, cover_std / max(avg_cover, 1) * 10) if avg_cover > 0 else 0

        total = cover_score + neg_score + trend_score + vol_score

        if total >= 70:   level = "🔴 Critical"
        elif total >= 45: level = "🟠 High"
        elif total >= 20: level = "🟡 Medium"
        else:             level = "🟢 Low"

        flags = []
        if min_cover < 0:               flags.append("Negative inventory")
        elif min_cover < 3:             flags.append("Cover < 3 months")
        elif min_cover < 7:             flags.append("Cover < 7 months")
        if months_below_7 >= 3:         flags.append(f"{months_below_7} months below 7")
        if months_negative > 0:         flags.append(f"{months_negative} months negative INV")
        if "Declining" in trend_label:  flags.append("Declining demand trend")
        if cover_std > avg_cover * 0.5 and avg_cover > 0: flags.append("High coverage volatility")
        if not flags:                   flags.append("None")

        results.append({
            "item":               item,
            "description":        desc,
            "risk_score":         round(total, 1),
            "risk_level":         level,
            "min_cover":          round(min_cover, 1),
            "avg_cover":          round(avg_cover, 1),
            "months_below_7":     months_below_7,
            "months_negative_inv":months_negative,
            "trend":              trend_label,
            "flags":              ", ".join(flags),
            "active_months_used": len(active_months),
        })
    return pd.DataFrame(results).sort_values("risk_score", ascending=False)


def compute_demand_spikes(pivot, z_thresh=2.0, pct_thresh=0.6):
    """Flag items where a FUTURE forecast month spikes above the item's own baseline.

    Baseline = mean/std of the item's forecast history up to (not including) the first
    On-Hand month (falls back to the future window itself if there's no history).
    A future month is flagged as a spike when it is both:
      - at least `pct_thresh` (e.g. 0.6 = 60%) above the baseline mean, AND
      - at least `z_thresh` standard deviations above the baseline mean (when std is known).
    Mirrors the Hebrew request: "מצא פריטים עם חריגת Spike בביקוש קדימה" —
    find items with a spike deviation in forward (future) demand.
    """
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    if not month_cols:
        return pd.DataFrame()

    fcst_rows = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)]
    oh_rows   = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("on hand", case=False, na=False)]
    if fcst_rows.empty:
        return pd.DataFrame()

    first_oh = None
    for m in month_cols:
        if m in oh_rows.columns and pd.to_numeric(oh_rows[m], errors="coerce").sum() > 0:
            first_oh = m
            break
    if first_oh and first_oh in month_cols:
        idx = month_cols.index(first_oh)
        history_months = month_cols[:idx]
        future_months  = month_cols[idx:]
    else:
        history_months = []
        future_months  = month_cols

    rows = []
    for item, grp in fcst_rows.groupby("item"):
        desc = grp["description"].dropna()
        desc = desc.iloc[0] if not desc.empty else ""

        hist_vals = pd.to_numeric(grp[history_months].values.flatten(), errors="coerce") if history_months else np.array([])
        hist_vals = hist_vals[~np.isnan(hist_vals) & (hist_vals > 0)]

        if len(hist_vals) >= 3:
            baseline_mean, baseline_std = float(hist_vals.mean()), float(hist_vals.std())
        else:
            fut_vals = pd.to_numeric(grp[future_months].values.flatten(), errors="coerce") if future_months else np.array([])
            fut_vals = fut_vals[~np.isnan(fut_vals) & (fut_vals > 0)]
            if len(fut_vals) >= 3:
                baseline_mean, baseline_std = float(fut_vals.mean()), float(fut_vals.std())
            else:
                continue

        if baseline_mean <= 0:
            continue

        for m in future_months:
            if m not in grp.columns:
                continue
            v = pd.to_numeric(grp[m].values[0], errors="coerce") if len(grp) > 0 else np.nan
            if pd.isna(v) or v <= 0:
                continue
            pct_above = (v - baseline_mean) / baseline_mean
            z = (v - baseline_mean) / baseline_std if baseline_std > 0 else np.nan
            is_spike = pct_above >= pct_thresh and (np.isnan(z) or z >= z_thresh)
            if not is_spike:
                continue
            rows.append({
                "item": item, "description": desc, "month": m,
                "forecast": round(float(v), 2),
                "baseline": round(baseline_mean, 2),
                "pct_above_baseline": round(pct_above * 100, 1),
                "z_score": round(float(z), 2) if not np.isnan(z) else None,
            })

    if not rows:
        return pd.DataFrame(columns=["item","description","month","forecast","baseline","pct_above_baseline","z_score"])
    return pd.DataFrame(rows).sort_values("pct_above_baseline", ascending=False)


def compute_forecast_accuracy(pivot):
    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    fcst_rows  = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)]
    act_rows   = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("actual",   case=False, na=False)]
    rows = []
    for item in pivot["item"].dropna().unique():
        fi   = fcst_rows[fcst_rows["item"] == item]
        ai   = act_rows[act_rows["item"]   == item]
        desc = pivot[pivot["item"] == item]["description"].dropna()
        desc = desc.iloc[0] if not desc.empty else ""
        for m in month_cols:
            f = pd.to_numeric(fi[m].values[0], errors="coerce") if not fi.empty and m in fi.columns else np.nan
            a = pd.to_numeric(ai[m].values[0], errors="coerce") if not ai.empty and m in ai.columns else np.nan
            if pd.isna(f) or pd.isna(a):
                continue
            err  = a - f
            mape = abs(err) / f * 100 if f != 0 else np.nan
            rows.append({
                "item": item, "description": desc, "month": m,
                "forecast": round(f, 2), "actual": round(a, 2),
                "error": round(err, 2), "abs_error": round(abs(err), 2),
                "mape": round(mape, 1) if not np.isnan(mape) else np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=["item","description","month","forecast","actual","error","abs_error","mape"])
    return pd.DataFrame(rows).sort_values(["item", "month"])

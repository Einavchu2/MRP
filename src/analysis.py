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
            # No forecast history to compare against — fall back to the future window
            # itself, but keep blank/zero months IN the baseline (as 0). Filtering them
            # out (as the history-based baseline does) would average the spikes against
            # each other and wash out exactly the "isolated peak among quiet months"
            # pattern this is meant to catch.
            fut_vals = pd.to_numeric(grp[future_months].values.flatten(), errors="coerce") if future_months else np.array([])
            fut_vals = np.nan_to_num(fut_vals, nan=0.0)
            fut_vals = np.clip(fut_vals, a_min=0, a_max=None)
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


def compute_forecast_vs_recent_trend(pivot, forecast_months=18, actual_months=24,
                                      mismatch_ratio=1.5, today=None):
    """
    Flags items where the forward forecast's average monthly rate diverges
    significantly from the recent actual consumption rate.

    This catches what compute_forecast_accuracy() structurally cannot: that
    function only compares forecast and actual within the SAME calendar month,
    but a rolling forecast is entered for future months and actuals only exist
    for past ones — they almost never land in the same month, so accuracy
    comes back empty for most items regardless of whether the forecast is any
    good. This check instead compares two separate windows: real consumption
    over the trailing `actual_months`, against the forecast over the next
    `forecast_months` — the actual question "does the plan still match how
    this item is really being used lately?"

    Deliberately NOT restricted to RM1/RM2 planner codes the way every other
    ML Analysis tab is — an item with planner_code "KPI - Ignore" (or any
    other code) is exactly the kind of item that's invisible everywhere else,
    which is the reason this check exists.

    today: anchor splitting the actual window (the `actual_months` months
    ending the month before `today`) from the forecast window (`forecast_months`
    months starting this month) — defaults to the real current date. Not
    derived from On Hand data, so it works for items with no On Hand row at
    all (unlike compute_risk).

    A month with no transaction/forecast row is treated as a genuine zero,
    not missing data — both rates are true monthly averages over the full
    window, not just the populated months.

    Returns one row per flagged item: item, description, planner_code,
    actual_rate, forecast_rate, ratio, direction, and the window boundaries
    used (so the UI can show exactly what was compared).
    """
    if today is None:
        today = pd.Timestamp.now()
    today = pd.Timestamp(today).replace(day=1)

    actual_start = today - pd.DateOffset(months=actual_months)
    actual_window = [(actual_start + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(actual_months)]
    forecast_window = [(today + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(forecast_months)]

    fcst_rows = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)]
    act_rows  = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("actual",   case=False, na=False)]
    # dropna first — an item can have a real description on one row and NaN on
    # another (e.g. its On Hand row), and without this, drop_duplicates("item")
    # can keep the NaN row and leave every consumer with a float, not a string.
    mm_local  = (pivot[["item", "description"]].dropna(subset=["description"])
                 .drop_duplicates("item").set_index("item")["description"].to_dict())

    all_items = pivot["item"].dropna().unique()
    rows = []
    for item in all_items:
        ai = act_rows[act_rows["item"] == item]
        fi = fcst_rows[fcst_rows["item"] == item]

        act_cols = [m for m in actual_window if m in ai.columns]
        fcst_cols = [m for m in forecast_window if m in fi.columns]

        # np.nansum, not .sum() — pd.to_numeric() on a raw ndarray (not a
        # Series) returns a plain ndarray, and ndarray.sum() propagates NaN
        # instead of skipping it the way Series.sum() does. That silently
        # zeroed out every item with any gap month in its window (i.e. most
        # of them) until caught by hand-tracing a real item against this code.
        act_sum = np.nansum(pd.to_numeric(ai[act_cols].values.flatten(), errors="coerce")) if not ai.empty and act_cols else 0.0
        fcst_sum = np.nansum(pd.to_numeric(fi[fcst_cols].values.flatten(), errors="coerce")) if not fi.empty and fcst_cols else 0.0
        act_sum = 0.0 if np.isnan(act_sum) else float(act_sum)
        fcst_sum = 0.0 if np.isnan(fcst_sum) else float(fcst_sum)

        actual_rate = act_sum / actual_months
        forecast_rate = fcst_sum / forecast_months

        if actual_rate <= 0 and forecast_rate <= 0:
            continue

        if actual_rate <= 0:
            direction, ratio = "no recent demand", None
            flagged = forecast_rate > 0
        else:
            ratio = forecast_rate / actual_rate
            if ratio >= mismatch_ratio:
                direction, flagged = "over-forecast", True
            elif ratio <= 1 / mismatch_ratio:
                direction, flagged = "under-forecast", True
            else:
                direction, flagged = "in line", False

        if not flagged:
            continue

        rows.append({
            "item": item,
            "description": mm_local.get(item, ""),
            "actual_rate": round(actual_rate, 2),
            "forecast_rate": round(forecast_rate, 2),
            "ratio": round(ratio, 2) if ratio is not None else None,
            "direction": direction,
            "actual_window": f"{actual_window[0]} → {actual_window[-1]}",
            "forecast_window": f"{forecast_window[0]} → {forecast_window[-1]}",
        })

    cols = ["item","description","actual_rate","forecast_rate","ratio","direction",
            "actual_window","forecast_window"]
    if not rows:
        return pd.DataFrame(columns=cols)

    def _extremity(r):
        # How far the ratio sits from 1 (in either direction) — used only to
        # rank rows worst-first. None ("no recent demand") sorts as most extreme.
        if r is None or pd.isna(r):
            return np.inf
        if r <= 0:
            return np.inf
        return max(r, 1 / r)

    out = pd.DataFrame(rows, columns=cols)
    out["_extremity"] = out["ratio"].apply(_extremity)
    return out.sort_values("_extremity", ascending=False).drop(columns="_extremity")


# ─── Spike anomaly classification ───────────────────────────────────────────

ANOMALY_CATEGORIES = {
    "demand_consumption_mismatch": ("Demand-Consumption Mismatch",     "חוסר התאמה בין תחזית/ביקוש לבין צריכה בפועל"),
    "forecast_bias":               ("Forecast Bias",                  "הטיה שיטתית בתחזית"),
    "structural_break":            ("Structural Break / Regime Change","שינוי מבני / שינוי דפוס ביקוש"),
    "over_forecasting_risk":       ("Over-Forecasting Risk",          "סיכון לתחזית יתר"),
    "demand_signal_distortion":    ("Demand Signal Distortion",       "עיוות באות הביקוש"),
    "inventory_exposure_risk":     ("Inventory Exposure Risk",        "חשיפה לסיכון מלאי"),
}


def classify_spike_anomalies(pivot, spike_df, trends_df=None, accuracy_df=None, risk_df=None):
    """
    For each item flagged in spike_df, decide which of the six anomaly categories
    genuinely apply — by cross-referencing the spike against the item's actual
    consumption history, historical forecast accuracy (bias/MAPE), demand trend
    direction, and existing inventory risk. Returns one row per (item, category)
    match found, with a short rationale; items/categories with no supporting
    evidence are simply absent (not every category applies to every item).
    """
    cols = ["item","description","category_key","category_en","category_he","rationale"]
    if spike_df is None or spike_df.empty:
        return pd.DataFrame(columns=cols)

    month_cols = sorted([c for c in pivot.columns if str(c).startswith("202")])
    fcst_rows = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("forecast", case=False, na=False)]
    act_rows  = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("actual",   case=False, na=False)]
    oh_rows   = pivot[pivot["ORDER_TYPE_FINAL"].str.contains("on hand",  case=False, na=False)]

    first_oh = None
    for m in month_cols:
        if m in oh_rows.columns and pd.to_numeric(oh_rows[m], errors="coerce").sum() > 0:
            first_oh = m
            break
    hist_idx = month_cols.index(first_oh) if first_oh in month_cols else 0
    history_months = month_cols[:hist_idx]

    trend_map = dict(zip(trends_df["item"], trends_df["trend_label"])) if trends_df is not None and not trends_df.empty else {}
    risk_map  = {r["item"]: r for _, r in risk_df.iterrows()} if risk_df is not None and not risk_df.empty else {}

    acc_by_item = {}
    if accuracy_df is not None and not accuracy_df.empty:
        for item, g in accuracy_df.groupby("item"):
            acc_by_item[item] = {"bias": g["error"].mean(), "mape": g["mape"].dropna().mean(), "n": len(g)}

    rows = []
    for item, spikes in spike_df.groupby("item"):
        desc = spikes["description"].iloc[0]
        top  = spikes.sort_values("pct_above_baseline", ascending=False).iloc[0]
        spike_val = float(top["forecast"])
        baseline  = float(top["baseline"])
        month     = top["month"]

        act_vals = pd.to_numeric(act_rows[act_rows["item"] == item][month_cols].values.flatten(), errors="coerce") \
            if not act_rows.empty else np.array([])
        act_vals = act_vals[~np.isnan(act_vals) & (act_vals > 0)]
        avg_actual = float(act_vals.mean()) if len(act_vals) else None
        max_actual = float(act_vals.max())  if len(act_vals) else None

        fcst_hist_vals = pd.to_numeric(
            fcst_rows[fcst_rows["item"] == item][history_months].values.flatten(), errors="coerce"
        ) if history_months else np.array([])
        fcst_hist_vals = fcst_hist_vals[~np.isnan(fcst_hist_vals) & (fcst_hist_vals > 0)]

        trend = trend_map.get(item, "")
        acc   = acc_by_item.get(item, {})
        risk  = risk_map.get(item)  # Series or None

        matches = []

        # 1. Demand-Consumption Mismatch — the spike doesn't line up with real usage
        if avg_actual is None:
            matches.append(("demand_consumption_mismatch",
                f"No actual consumption history exists to validate this demand — the {spike_val:,.0f}-unit spike "
                f"in {month} can't be confirmed against real usage."))
        elif spike_val > avg_actual * 1.5 or spike_val < avg_actual * 0.5:
            direction = "far above" if spike_val > avg_actual else "far below"
            matches.append(("demand_consumption_mismatch",
                f"Forecast spike of {spike_val:,.0f} in {month} is {direction} the item's average actual "
                f"consumption of {avg_actual:,.1f}."))

        # 2. Forecast Bias — forecast systematically runs higher than actual
        if acc.get("n", 0) >= 3 and acc.get("bias") is not None and avg_actual:
            bias = acc["bias"]
            if bias < 0 and abs(bias) / max(avg_actual, 1) >= 0.2:
                matches.append(("forecast_bias",
                    f"Forecast has systematically over-stated demand: avg bias {bias:+.1f} units/month over "
                    f"{acc['n']} overlapping months ({abs(bias)/max(avg_actual,1)*100:.0f}% of typical consumption), "
                    f"MAPE {acc.get('mape', 0):.0f}%."))

        # 3. Structural Break / Regime Change — plan hasn't adapted to changed behavior
        if "Declining" in trend:
            matches.append(("structural_break",
                f"Actual consumption trend is declining, yet the plan still forecasts a {spike_val:,.0f}-unit "
                f"spike in {month} — planning hasn't adapted to the changed demand pattern."))

        # 4. Over-Forecasting Risk — spike exceeds anything ever actually consumed
        if max_actual is not None and spike_val > max_actual * 1.3:
            matches.append(("over_forecasting_risk",
                f"The {spike_val:,.0f}-unit spike in {month} exceeds the item's highest-ever actual monthly "
                f"consumption ({max_actual:,.1f}) by {(spike_val/max_actual-1)*100:.0f}%."))

        # 5. Demand Signal Distortion — planning data isn't grounded in real consumption
        if len(fcst_hist_vals) < 3:
            matches.append(("demand_signal_distortion",
                "No reliable forecast history exists for this item before the current period — the spike is "
                "being judged against itself rather than a validated demand signal."))
        elif acc.get("mape") and acc["mape"] > 75:
            matches.append(("demand_signal_distortion",
                f"Forecast accuracy has historically been very poor (MAPE {acc['mape']:.0f}%), meaning planning "
                f"data has not reliably tracked real consumption."))

        # 6. Inventory Exposure Risk — this spike has real stock/PO consequences
        if risk is not None and ("Critical" in str(risk.get("risk_level", "")) or "High" in str(risk.get("risk_level", ""))):
            matches.append(("inventory_exposure_risk",
                f"Item is independently flagged {risk['risk_level']} in the Risk Dashboard "
                f"(min cover {risk.get('min_cover','?')} mo) — this spike compounds an existing inventory risk."))
        elif baseline > 0 and spike_val > baseline * 2:
            matches.append(("inventory_exposure_risk",
                f"A spike this size ({spike_val:,.0f} vs baseline {baseline:,.1f}) would likely trigger an "
                f"oversized one-time purchase order and leave excess stock once demand reverts."))

        for key, rationale in matches:
            en, he = ANOMALY_CATEGORIES[key]
            rows.append({"item": item, "description": desc, "category_key": key,
                         "category_en": en, "category_he": he, "rationale": rationale})

    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

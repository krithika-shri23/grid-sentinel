"""
validation.py
==============
Validation harness for the electricity anomaly-detection pipeline (Innovator Track).

WHAT THIS DOES
--------------
1. Loads real consumption data if a CSV matching the schema is found, otherwise
   simulates realistic daily-load-shaped data (so this script runs standalone
   even before Person A's real pipeline is wired in).
2. Injects synthetic anomalies (spike, drop, flatline, pattern_break) with
   KNOWN ground-truth labels -- because the UCI dataset has no real theft/
   anomaly labels, this is how we can honestly measure precision/recall.
3. Runs three independent detectors:
     - z-score vs a weekday+hour-of-day baseline  (sudden extreme spikes/drops)
     - Isolation Forest on rolling-window features (unusual pattern combos)
     - flatline detector (near-zero rolling variance -> meter/data issue)
4. Combines them into a single risk_score/confidence per window (matches
   Person A's output schema).
5. Applies a PERSISTENCE FILTER (must flag 3+ consecutive windows to alert)
   as the false-positive-reduction mechanism.
6. Reports precision, recall, F1, and false-positive rate WITH and WITHOUT
   the persistence filter, so you have a real "before/after" slide.

RAW DATA SCHEMA EXPECTED (if you supply your own CSV via --data_csv):
  account_id: string
  timestamp: datetime, 15-min intervals
  consumption_kwh: float

Run:
    python validation.py
    python validation.py --data_csv path/to/real_data.csv
    python validation.py --n_accounts 30 --days 14 --seed 7
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RNG_DEFAULT_SEED = 42


# ----------------------------------------------------------------------------
# 1. DATA LOADING / SIMULATION
# ----------------------------------------------------------------------------

def load_or_simulate_data(data_csv: str | None, n_accounts: int, days: int, seed: int) -> pd.DataFrame:
    """Load a real CSV matching the schema, or simulate realistic 15-min load data.

    Simulated load shape: a daily double-hump curve (morning + evening peak),
    weekday/weekend variation, small per-account baseline differences, and
    Gaussian noise -- close enough in *shape* to real household smart-meter
    data (UCI ElectricityLoadDiagrams) to sanity-check the detectors before
    real data is plugged in.
    """
    if data_csv:
        df = pd.read_csv(data_csv, parse_dates=["timestamp"])
        required = {"account_id", "timestamp", "consumption_kwh"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"data_csv is missing required columns: {missing}")
        return df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    periods_per_day = 96  # 15-min intervals
    n_periods = days * periods_per_day
    start = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start, periods=n_periods, freq="15min")

    rows = []
    for acc_idx in range(n_accounts):
        account_id = f"ACC_{acc_idx:04d}"
        base_level = rng.uniform(0.15, 0.6)  # kWh baseline per 15-min slot
        for t in timestamps:
            hour = t.hour + t.minute / 60.0
            is_weekend = t.dayofweek >= 5
            # double-hump daily curve: morning ~7-9am, evening ~18-22
            morning = 1.4 * np.exp(-0.5 * ((hour - 8.0) / 1.3) ** 2)
            evening = 2.0 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)
            weekend_bump = 0.3 if is_weekend else 0.0
            noise = rng.normal(0, 0.06)
            value = max(0.02, base_level * (0.6 + morning + evening + weekend_bump) + noise)
            rows.append((account_id, t, value))

    df = pd.DataFrame(rows, columns=["account_id", "timestamp", "consumption_kwh"])
    return df


# ----------------------------------------------------------------------------
# 2. SYNTHETIC ANOMALY INJECTION (WITH GROUND TRUTH)
# ----------------------------------------------------------------------------

def inject_anomalies(df: pd.DataFrame, seed: int, n_events_per_type: int = 8) -> pd.DataFrame:
    """Injects spike / drop / flatline / pattern_break events into the data and
    returns df with an added `is_anomaly` ground-truth column (1 = anomalous
    window, 0 = normal). Injection points are chosen at random per account,
    with parameters designed to mimic plausible real-world causes without
    ever claiming to represent confirmed theft cases.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["is_anomaly"] = 0
    df["injected_type"] = None

    accounts = df["account_id"].unique()
    idx_by_account = {a: df.index[df["account_id"] == a].to_numpy() for a in accounts}

    anomaly_specs = {
        "spike": dict(duration_range=(1, 4), factor_range=(2.2, 4.0)),
        "drop": dict(duration_range=(2, 6), factor_range=(0.05, 0.35)),
        "flatline": dict(duration_range=(4, 12), factor_range=(1.0, 1.0)),  # value pinned constant
        "pattern_break": dict(duration_range=(8, 20), factor_range=(1.4, 2.2)),  # shifted/elevated plateau
    }

    for anomaly_type, spec in anomaly_specs.items():
        for _ in range(n_events_per_type):
            account = rng.choice(accounts)
            positions = idx_by_account[account]
            dur = rng.integers(spec["duration_range"][0], spec["duration_range"][1] + 1)
            if len(positions) <= dur + 5:
                continue
            start_pos = rng.integers(2, len(positions) - dur - 2)
            window_idx = positions[start_pos:start_pos + dur]
            factor = rng.uniform(*spec["factor_range"])

            if anomaly_type == "spike":
                df.loc[window_idx, "consumption_kwh"] *= factor
            elif anomaly_type == "drop":
                df.loc[window_idx, "consumption_kwh"] *= factor
            elif anomaly_type == "flatline":
                pinned_value = df.loc[window_idx[0], "consumption_kwh"]
                df.loc[window_idx, "consumption_kwh"] = pinned_value
            elif anomaly_type == "pattern_break":
                df.loc[window_idx, "consumption_kwh"] *= factor

            df.loc[window_idx, "is_anomaly"] = 1
            df.loc[window_idx, "injected_type"] = anomaly_type

    return df


# ----------------------------------------------------------------------------
# 3. DETECTOR 1: Z-SCORE vs WEEKDAY+HOUR BASELINE
# ----------------------------------------------------------------------------

def _mad(x: pd.Series) -> float:
    """Median Absolute Deviation, scaled to be comparable to std for
    normally-distributed data (the 1.4826 constant is the standard scale
    factor). Far more robust than mean/std when a handful of the historical
    samples in a bucket are themselves anomalies -- mean/std blend an
    anomaly's own value into the baseline it's being compared against
    (self-contamination), which quietly raises the baseline and hides the
    very anomaly we're trying to catch, especially with a short data window
    where each (dow, hour) slot only has a few historical samples.
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account baseline (median + MAD) for each (dayofweek, hour, minute)
    slot. Using robust statistics instead of mean/std matters most when the
    baseline window is short (few historical samples per slot) -- run with
    enough --days (45-60+) that each weekday/time-of-day bucket has a
    reasonable number of historical points to estimate from.
    """
    tmp = df.copy()
    tmp["dow"] = tmp["timestamp"].dt.dayofweek
    tmp["hm"] = tmp["timestamp"].dt.hour * 100 + (tmp["timestamp"].dt.minute // 15) * 15
    baseline = (
        tmp.groupby(["account_id", "dow", "hm"])["consumption_kwh"]
        .agg(baseline_mean="median", baseline_std=_mad)
        .reset_index()
    )
    baseline["baseline_std"] = baseline["baseline_std"].fillna(0.05).clip(lower=0.05)
    return baseline


def detect_zscore(df: pd.DataFrame, baseline: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    tmp = df.copy()
    tmp["dow"] = tmp["timestamp"].dt.dayofweek
    tmp["hm"] = tmp["timestamp"].dt.hour * 100 + (tmp["timestamp"].dt.minute // 15) * 15
    merged = tmp.merge(baseline, on=["account_id", "dow", "hm"], how="left")
    merged["zscore"] = (merged["consumption_kwh"] - merged["baseline_mean"]) / merged["baseline_std"]
    merged["flag_zscore"] = (merged["zscore"].abs() >= z_threshold).astype(int)
    merged["anomaly_subtype_zscore"] = np.where(
        merged["zscore"] >= z_threshold, "spike",
        np.where(merged["zscore"] <= -z_threshold, "drop", None)
    )
    return merged[["account_id", "timestamp", "consumption_kwh", "baseline_mean",
                   "zscore", "flag_zscore", "anomaly_subtype_zscore"]]


# ----------------------------------------------------------------------------
# 4. DETECTOR 2: ISOLATION FOREST ON ROLLING-WINDOW FEATURES
# ----------------------------------------------------------------------------

def build_rolling_features(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Rolling mean, std, and rate-of-change per account -- captures 'unusual
    combinations' of behavior that a single-point z-score can miss (e.g. a
    moderately elevated but erratic plateau)."""
    out = []
    for account, g in df.groupby("account_id"):
        g = g.sort_values("timestamp").copy()
        g["roll_mean"] = g["consumption_kwh"].rolling(window, min_periods=3).mean()
        g["roll_std"] = g["consumption_kwh"].rolling(window, min_periods=3).std()  # leave NaN for warm-up rows (first `window`-1 points) -- there isn't enough history yet to claim flatline or anything else about them, so they should not be flagged rather than defaulting to 0.0
        g["roll_delta"] = g["consumption_kwh"].diff().fillna(0.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def detect_isolation_forest(df_feat: pd.DataFrame, contamination: float = 0.05, seed: int = 0) -> pd.DataFrame:
    results = []
    for account, g in df_feat.groupby("account_id"):
        g = g.copy()
        features = g[["consumption_kwh", "roll_mean", "roll_std", "roll_delta"]].fillna(0.0)
        if len(features) < 20:
            g["flag_isoforest"] = 0
            g["if_score"] = 0.0
            results.append(g)
            continue
        model = IsolationForest(n_estimators=150, contamination=contamination, random_state=seed)
        preds = model.fit_predict(features)          # -1 = anomaly, 1 = normal
        scores = -model.score_samples(features)       # higher = more anomalous
        g["flag_isoforest"] = (preds == -1).astype(int)
        g["if_score"] = scores
        results.append(g)
    return pd.concat(results, ignore_index=True)


# ----------------------------------------------------------------------------
# 5. DETECTOR 3: FLATLINE
# ----------------------------------------------------------------------------

def detect_flatline(df_feat: pd.DataFrame, window: int = 4, std_threshold: float = 0.01) -> pd.DataFrame:
    """Flags windows where rolling std is near-zero AND the value itself is
    non-trivial (avoids flagging legitimately near-zero idle consumption as
    a 'flatline fault')."""
    df_feat = df_feat.copy()
    df_feat["flag_flatline"] = (
        (df_feat["roll_std"] <= std_threshold) & (df_feat["consumption_kwh"] > 0.05)
    ).astype(int)
    return df_feat


# ----------------------------------------------------------------------------
# 6. COMBINE INTO RISK / CONFIDENCE SCORE (matches Person A's output schema)
# ----------------------------------------------------------------------------

def combine_signals(merged: pd.DataFrame) -> pd.DataFrame:
    """Weighted combination -> risk_score (0-100) and confidence (0-100).
    Weighting rationale:
      - z-score flag is the strongest single signal for sudden extremes
      - isolation forest adds evidence for 'unusual pattern' without a hard spike
      - flatline is treated as high-confidence on its own (distinct failure mode)
    """
    df = merged.copy()
    z_component = df["flag_zscore"] * 45
    if_component = df["flag_isoforest"] * 30
    flat_component = df["flag_flatline"] * 40

    raw_score = z_component + if_component + flat_component
    df["risk_score"] = raw_score.clip(upper=100).astype(int)

    n_signals_firing = df[["flag_zscore", "flag_isoforest", "flag_flatline"]].sum(axis=1)
    df["confidence"] = (40 + n_signals_firing * 20).clip(upper=100).astype(int)

    df["flag_any"] = (df["risk_score"] >= 40).astype(int)

    def label(row):
        if row["flag_flatline"]:
            return "flatline"
        if row["anomaly_subtype_zscore"] == "spike" or (row["flag_isoforest"] and row["roll_delta"] > 0):
            return "spike"
        if row["anomaly_subtype_zscore"] == "drop" or (row["flag_isoforest"] and row["roll_delta"] < 0):
            return "drop"
        return "pattern_break"

    df["anomaly_type"] = df.apply(label, axis=1)
    return df


# ----------------------------------------------------------------------------
# 7. PERSISTENCE FILTER (false-positive reduction mechanism)
# ----------------------------------------------------------------------------

def apply_persistence_filter(df: pd.DataFrame, min_consecutive: int = 3) -> pd.DataFrame:
    """A flagged window only becomes an ALERT if `min_consecutive` consecutive
    15-min windows are flagged for the same account. This suppresses one-off
    noisy blips (e.g. a single high-load appliance cycle) while still catching
    sustained anomalies fast (3 windows = 45 minutes)."""
    df = df.sort_values(["account_id", "timestamp"]).copy()
    df["alert_persistent"] = 0

    for account, g in df.groupby("account_id"):
        flags = g["flag_any"].to_numpy()
        run_length = 0
        alert = np.zeros(len(flags), dtype=int)
        for i, f in enumerate(flags):
            run_length = run_length + 1 if f else 0
            if run_length >= min_consecutive:
                alert[max(0, i - min_consecutive + 1): i + 1] = 1
        df.loc[g.index, "alert_persistent"] = alert

    return df


# ----------------------------------------------------------------------------
# 8. EVALUATION
# ----------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, prediction_col: str) -> dict:
    y_true = df["is_anomaly"].to_numpy()
    y_pred = df[prediction_col].to_numpy()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, fpr=fpr)


def build_reason_string(row) -> str:
    """Human-readable, framing-safe explanation string (per locked pitch rules:
    never 'theft', always a possible-cause label)."""
    ratio = row["consumption_kwh"] / row["baseline_mean"] if row["baseline_mean"] > 0 else 0.0
    if row["anomaly_type"] == "flatline":
        return "Consumption reading is suspiciously constant vs. normal variation (possible meter/data issue)."
    if row["anomaly_type"] == "spike":
        return f"Consumption is {ratio:.1f}x higher than the normal baseline for this time slot (possible unexpected load / equipment issue)."
    if row["anomaly_type"] == "drop":
        return f"Consumption is {ratio:.1f}x the normal baseline for this time slot, a significant drop (possible outage, vacancy, or unauthorized bypass — cause not confirmed)."
    return "Consumption pattern deviates from the account's usual behavior across several combined signals (possible unexpected load pattern)."


def export_locked_schema(df: pd.DataFrame, out_json: str = "flagged_accounts.json", alert_col: str = "alert_persistent"):
    """Exports ONLY alerted windows in the exact locked Detection Output Schema
    from SCHEMA.md, so Person B can consume this directly without renaming
    columns. Uses `alert_persistent` (post-persistence-filter) as the alert
    condition since that's the production-facing signal."""
    alerted = df[df[alert_col] == 1].copy()
    if alerted.empty:
        print(f"  No alerts to export under column '{alert_col}'.")
        pd.DataFrame([]).to_json(out_json, orient="records")
        return

    alerted["reason"] = alerted.apply(build_reason_string, axis=1)
    alerted["window_start"] = alerted["timestamp"]
    alerted["window_end"] = alerted["timestamp"] + pd.Timedelta(minutes=15)
    alerted["actual_value"] = alerted["consumption_kwh"]
    alerted["baseline_value"] = alerted["baseline_mean"]

    schema_cols = ["account_id", "window_start", "window_end", "risk_score", "confidence",
                   "anomaly_type", "reason", "actual_value", "baseline_value"]
    out = alerted[schema_cols].copy()
    out["window_start"] = out["window_start"].astype(str)
    out["window_end"] = out["window_end"].astype(str)
    out.to_json(out_json, orient="records", indent=2)
    print(f"  Exported {len(out)} alerts to {out_json} (locked schema, matches SCHEMA.md exactly)")


def print_report(title: str, metrics: dict):
    print(f"\n--- {title} ---")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  Precision:            {metrics['precision']:.3f}")
    print(f"  Recall:               {metrics['recall']:.3f}")
    print(f"  F1:                   {metrics['f1']:.3f}")
    print(f"  False Positive Rate:  {metrics['fpr']:.4f}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate the anomaly-detection pipeline against synthetic ground truth.")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to real data CSV (account_id, timestamp, consumption_kwh). If omitted, data is simulated.")
    parser.add_argument("--n_accounts", type=int, default=25)
    parser.add_argument("--days", type=int, default=60, help="More days = more historical samples per weekday/time-of-day baseline bucket = more stable detection. 14 is too few; 45-60+ recommended.")
    parser.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED)
    parser.add_argument("--min_consecutive", type=int, default=3, help="Persistence filter window (# consecutive 15-min slots).")
    parser.add_argument("--out_csv", type=str, default="validation_output.csv", help="Where to save the per-window detection output.")
    parser.add_argument("--out_json", type=str, default="flagged_accounts.json", help="Where to save alerts in the locked Detection Output Schema (SCHEMA.md).")
    args = parser.parse_args()

    print("Loading / simulating data...")
    df = load_or_simulate_data(args.data_csv, args.n_accounts, args.days, args.seed)
    print(f"  {df['account_id'].nunique()} accounts, {len(df)} rows")

    print("Injecting synthetic anomalies with known ground truth...")
    df = inject_anomalies(df, seed=args.seed)
    print(f"  {int(df['is_anomaly'].sum())} anomalous windows injected "
          f"({df['is_anomaly'].mean() * 100:.2f}% of all windows)")

    print("Running z-score detector...")
    baseline = build_baseline(df)
    z_result = detect_zscore(df, baseline)

    print("Building rolling features + running Isolation Forest...")
    feat = build_rolling_features(df)
    if_result = detect_isolation_forest(feat, seed=args.seed)

    print("Running flatline detector...")
    flat_result = detect_flatline(if_result)

    print("Combining signals into risk/confidence score...")
    merged = df.merge(
        z_result[["account_id", "timestamp", "baseline_mean", "zscore", "flag_zscore", "anomaly_subtype_zscore"]],
        on=["account_id", "timestamp"], how="left"
    ).merge(
        flat_result[["account_id", "timestamp", "roll_mean", "roll_std", "roll_delta",
                      "flag_isoforest", "if_score", "flag_flatline"]],
        on=["account_id", "timestamp"], how="left"
    )
    combined = combine_signals(merged)

    print(f"Applying persistence filter (min_consecutive={args.min_consecutive})...")
    final = apply_persistence_filter(combined, min_consecutive=args.min_consecutive)

    # ---- Evaluation: raw flag vs. persistence-filtered alert ----
    metrics_raw = evaluate(final, "flag_any")
    metrics_filtered = evaluate(final, "alert_persistent")

    print_report("WITHOUT persistence filter (raw per-window flags)", metrics_raw)
    print_report(f"WITH persistence filter ({args.min_consecutive}+ consecutive windows)", metrics_filtered)

    fp_reduction = (
        (metrics_raw["fp"] - metrics_filtered["fp"]) / metrics_raw["fp"] * 100
        if metrics_raw["fp"] > 0 else 0.0
    )
    recall_change = (metrics_filtered["recall"] - metrics_raw["recall"]) * 100

    print("\n=== HEADLINE NUMBERS FOR THE DECK ===")
    print(f"False positives reduced by {fp_reduction:.1f}% "
          f"({metrics_raw['fp']} -> {metrics_filtered['fp']})")
    print(f"Recall change: {recall_change:+.1f} percentage points "
          f"({metrics_raw['recall']*100:.1f}% -> {metrics_filtered['recall']*100:.1f}%)")
    print(f"Precision: {metrics_raw['precision']*100:.1f}% (raw) -> {metrics_filtered['precision']*100:.1f}% (filtered)")

    # Save per-window output matching Person A's detection schema for handoff
    schema_cols = ["account_id", "timestamp", "consumption_kwh", "risk_score", "confidence",
                   "anomaly_type", "flag_any", "alert_persistent", "is_anomaly", "injected_type"]
    final[schema_cols].to_csv(args.out_csv, index=False)
    print(f"\nSaved per-window results to: {args.out_csv}")

    print("\nExporting alerted windows in the locked Detection Output Schema...")
    export_locked_schema(final, out_json=args.out_json, alert_col="alert_persistent")


if __name__ == "__main__":
    main()

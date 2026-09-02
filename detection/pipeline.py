"""
Grid Sentinel — Detection Pipeline (Person A)

Loads electricity consumption data, injects synthetic anomalies with
known ground truth (since UCI has no real theft/anomaly labels), runs
three detectors (z-score, Isolation Forest, flatline), combines them
into a risk/confidence score, applies a persistence filter to reduce
false positives, and exports output matching SCHEMA.md.

Run: python pipeline.py
Output: data/flagged_accounts.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_PATH = "data/flagged_accounts.json"
N_ACCOUNTS = 20
DAYS = 14
INTERVALS_PER_DAY = 96  # 15-min intervals


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
def load_or_generate_data():
    """
    Tries to load a real UCI-style CSV from data/raw/. If not found,
    generates realistic synthetic multi-account consumption data so
    the pipeline is runnable immediately without waiting on a download.

    To use the real UCI ElectricityLoadDiagrams2011-2014 dataset:
    https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    Raw values are in kW — divide by 4 to get kWh per 15-min interval.
    Save as data/raw/consumption.csv with columns: account_id,
    timestamp, consumption_kw (script converts to kWh automatically).
    """
    raw_path = "data/raw/consumption.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
        df["consumption_kwh"] = df["consumption_kw"] / 4.0
        print(f"Loaded real data from {raw_path}: {len(df)} rows")
        return df

    print("No real data found at data/raw/consumption.csv — generating synthetic data instead.")
    return _generate_synthetic_data()


def _generate_synthetic_data():
    rows = []
    start = pd.Timestamp("2024-01-01")
    timestamps = [start + pd.Timedelta(minutes=15 * i) for i in range(DAYS * INTERVALS_PER_DAY)]

    for acc_id in range(1, N_ACCOUNTS + 1):
        account_id = f"ACC{acc_id:03d}"
        base_level = np.random.uniform(0.3, 1.2)  # kWh baseline
        for ts in timestamps:
            hour = ts.hour
            # simple daily pattern: low at night, peaks morning/evening
            daily_factor = 0.5 + 0.5 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
            noise = np.random.normal(0, 0.05)
            consumption = max(0.05, base_level * daily_factor + noise)
            rows.append((account_id, ts, consumption))

    df = pd.DataFrame(rows, columns=["account_id", "timestamp", "consumption_kwh"])
    return df


# ---------------------------------------------------------------------
# 2. SYNTHETIC ANOMALY INJECTION (with ground truth labels)
# ---------------------------------------------------------------------
def inject_anomalies(df):
    """
    Injects three anomaly types into a subset of accounts, at known
    windows, and records ground truth so we can compute precision/
    recall later. This is what makes validation possible without real
    labeled theft data — and it's a limitation we state openly in the
    pitch, not something we hide.
    """
    df = df.copy()
    df["is_injected_anomaly"] = False
    df["injected_type"] = None

    accounts = df["account_id"].unique()
    anomaly_accounts = np.random.choice(accounts, size=max(3, len(accounts) // 4), replace=False)
    anomaly_types = ["spike", "drop", "flatline"]

    for i, acc in enumerate(anomaly_accounts):
        mask_acc = df["account_id"] == acc
        acc_df = df[mask_acc]
        anomaly_type = anomaly_types[i % len(anomaly_types)]

        # pick a random 4-hour window (16 intervals) to inject into
        start_idx = np.random.randint(0, len(acc_df) - 20)
        window_idx = acc_df.index[start_idx:start_idx + 16]

        if anomaly_type == "spike":
            df.loc[window_idx, "consumption_kwh"] *= np.random.uniform(2.5, 4.0)
        elif anomaly_type == "drop":
            df.loc[window_idx, "consumption_kwh"] *= np.random.uniform(0.05, 0.2)
        elif anomaly_type == "flatline":
            flat_value = df.loc[window_idx, "consumption_kwh"].mean()
            df.loc[window_idx, "consumption_kwh"] = flat_value

        df.loc[window_idx, "is_injected_anomaly"] = True
        df.loc[window_idx, "injected_type"] = anomaly_type

    print(f"Injected anomalies into {len(anomaly_accounts)} accounts: {list(anomaly_accounts)}")
    return df


# ---------------------------------------------------------------------
# 3. BASELINE COMPUTATION
# ---------------------------------------------------------------------
def compute_baselines(df):
    """Per-account, per-hour-of-day baseline (mean, std)."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    baseline = (
        df.groupby(["account_id", "hour"])["consumption_kwh"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
        .reset_index()
    )
    baseline["baseline_std"] = baseline["baseline_std"].fillna(0.01).replace(0, 0.01)
    df = df.merge(baseline, on=["account_id", "hour"], how="left")
    return df


# ---------------------------------------------------------------------
# 4. THREE DETECTORS
# ---------------------------------------------------------------------
def detect_zscore(df, threshold=2.5):
    """Catches sudden extreme spikes/drops relative to hour-of-day baseline."""
    df["zscore"] = (df["consumption_kwh"] - df["baseline_mean"]) / df["baseline_std"]
    df["zscore_signal"] = (df["zscore"].abs() > threshold).astype(float) * \
        np.clip(df["zscore"].abs() / (threshold * 2), 0, 1)
    return df


def detect_isolation_forest(df):
    """Catches unusual pattern combinations, not necessarily extreme values."""
    df = df.copy()
    results = np.zeros(len(df))
    for acc in df["account_id"].unique():
        mask = df["account_id"] == acc
        acc_df = df.loc[mask]
        features = acc_df[["consumption_kwh", "hour"]].values
        if len(features) < 20:
            continue
        model = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
        raw_scores = model.fit_predict(features)  # -1 = anomaly, 1 = normal
        anomaly_scores = model.decision_function(features)  # lower = more anomalous
        normalized = 1 - (anomaly_scores - anomaly_scores.min()) / (
            anomaly_scores.max() - anomaly_scores.min() + 1e-9
        )
        results[mask.values] = normalized
    df["iforest_signal"] = results
    return df


def detect_flatline(df, window=8, std_threshold=0.02):
    """Catches suspiciously constant readings — meter/data issue, not necessarily theft."""
    df = df.sort_values(["account_id", "timestamp"]).copy()
    df["rolling_std"] = (
        df.groupby("account_id")["consumption_kwh"]
        .transform(lambda x: x.rolling(window, min_periods=window).std())
    )
    df["flatline_signal"] = (df["rolling_std"] < std_threshold).astype(float)
    df["flatline_signal"] = df["flatline_signal"].fillna(0)
    return df


# ---------------------------------------------------------------------
# 5. COMBINE SIGNALS + PERSISTENCE FILTER
# ---------------------------------------------------------------------
def combine_and_filter(df, persistence_windows=3):
    """
    Weighted combination of the three signals into a 0-100 risk score,
    then requires the combined signal to stay elevated for N consecutive
    windows before it counts as a real alert (false-positive reduction).
    """
    df = df.copy()
    df["combined_signal"] = (
        0.4 * df["zscore_signal"] + 0.4 * df["iforest_signal"] + 0.2 * df["flatline_signal"]
    )
    df["risk_score"] = (df["combined_signal"] * 100).clip(0, 100).round().astype(int)

    alert_threshold = 0.4
    df["is_alert_raw"] = df["combined_signal"] > alert_threshold

    # persistence filter: raw alert must hold for N consecutive windows per account
    df = df.sort_values(["account_id", "timestamp"])
    df["alert_run"] = (
        df.groupby("account_id")["is_alert_raw"]
        .transform(lambda x: x.rolling(persistence_windows, min_periods=persistence_windows).sum())
    )
    df["is_alert_filtered"] = df["alert_run"] >= persistence_windows

    return df


# ---------------------------------------------------------------------
# 6. VALIDATION (against injected ground truth)
# ---------------------------------------------------------------------
def validate(df):
    def precision_recall(alert_col):
        tp = ((df[alert_col]) & (df["is_injected_anomaly"])).sum()
        fp = ((df[alert_col]) & (~df["is_injected_anomaly"])).sum()
        fn = ((~df[alert_col]) & (df["is_injected_anomaly"])).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall, fp

    p_raw, r_raw, fp_raw = precision_recall("is_alert_raw")
    p_filt, r_filt, fp_filt = precision_recall("is_alert_filtered")

    print("\n--- Validation against synthetically injected anomalies ---")
    print(f"Before persistence filter: precision={p_raw:.2f}, recall={r_raw:.2f}, false_positives={fp_raw}")
    print(f"After persistence filter:  precision={p_filt:.2f}, recall={r_filt:.2f}, false_positives={fp_filt}")
    print("(Report this before/after comparison in the pitch — it's the false-positive")
    print(" reduction deliverable, and it's honest since it's measured against known")
    print(" injected anomalies, not confirmed real-world theft cases.)\n")


# ---------------------------------------------------------------------
# 7. EXPORT (matches SCHEMA.md exactly)
# ---------------------------------------------------------------------
def export_flagged_accounts(df, path=OUTPUT_PATH):
    alerts = df[df["is_alert_filtered"]].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    output = []
    for acc, group in alerts.groupby("account_id"):
        row = group.iloc[-1]  # most recent alert window for this account
        anomaly_type = row["injected_type"] if pd.notna(row["injected_type"]) else _infer_type(row)
        reason = _build_reason(row, anomaly_type)

        output.append({
            "account_id": acc,
            "window_start": group["timestamp"].min().isoformat(),
            "window_end": group["timestamp"].max().isoformat(),
            "risk_score": int(group["risk_score"].max()),
            "confidence": int(group["risk_score"].max()),  # using risk_score as confidence proxy
            "anomaly_type": anomaly_type,
            "reason": reason,
            "actual_value": round(float(row["consumption_kwh"]), 3),
            "baseline_value": round(float(row["baseline_mean"]), 3),
        })

    output.sort(key=lambda x: x["risk_score"], reverse=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {len(output)} flagged accounts to {path}")
    return output


def _infer_type(row):
    if row["flatline_signal"] > 0.5:
        return "flatline"
    if row["consumption_kwh"] > row["baseline_mean"]:
        return "spike"
    return "drop"


def _build_reason(row, anomaly_type):
    ratio = row["consumption_kwh"] / max(row["baseline_mean"], 0.01)
    if anomaly_type == "spike":
        return f"Consumption is {ratio:.1f}x higher than normal baseline for this hour."
    elif anomaly_type == "drop":
        return f"Consumption dropped to {ratio:.1f}x of normal baseline for this hour."
    else:
        return "Consumption is suspiciously constant — possible meter or data issue."


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_or_generate_data()
    df = inject_anomalies(df)
    df = compute_baselines(df)
    df = detect_zscore(df)
    df = detect_isolation_forest(df)
    df = detect_flatline(df)
    df = combine_and_filter(df)

    validate(df)
    export_flagged_accounts(df)


if __name__ == "__main__":
    main()
"""
Reshape UCI ElectricityLoadDiagrams2011-2014 raw file into the format
pipeline.py expects: account_id, timestamp, consumption_kw

Raw file format: semicolon-delimited, first column is timestamp,
remaining columns are client IDs (e.g. MT_001, MT_002...), values use
comma as decimal separator, values in kW.

Usage:
    python reshape_uci.py

Reads:  data/raw/LD2011_2014.txt
Writes: data/raw/consumption.csv  (auto-detected by pipeline.py)
"""

import pandas as pd

INPUT_PATH = "data/raw/LD2011_2014.txt"
OUTPUT_PATH = "data/raw/consumption.csv"

# The full file is huge (370 clients x 4 years x 15-min intervals =
# millions of rows). For a 24hr hackathon, use a subset: fewer clients
# and a shorter date range, so the pipeline runs fast and the dashboard
# stays responsive during a live demo.
N_CLIENTS = 20
N_DAYS = 30  # most recent N_DAYS of data


def main():
    print(f"Reading {INPUT_PATH} (this file is large, may take a minute)...")
    df = pd.read_csv(INPUT_PATH, sep=";", decimal=",", index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    print(f"Full shape: {df.shape[0]} timestamps x {df.shape[1]} clients")

    # Take a subset of clients and the most recent N_DAYS
    selected_clients = df.columns[:N_CLIENTS]
    df = df[selected_clients]

    cutoff = df.index.max() - pd.Timedelta(days=N_DAYS)
    df = df[df.index >= cutoff]

    print(f"Subset shape: {df.shape[0]} timestamps x {df.shape[1]} clients")

    # Melt wide -> long format
    df_long = df.reset_index().melt(
        id_vars="timestamp", var_name="account_id", value_name="consumption_kw"
    )

    # UCI values are already in kW per the dataset documentation
    df_long = df_long.dropna(subset=["consumption_kw"])
    df_long = df_long.sort_values(["account_id", "timestamp"])

    df_long.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df_long)} rows to {OUTPUT_PATH}")
    print("Now run: python detection/pipeline.py")
    print("(it will auto-detect this file and use real data instead of synthetic)")


if __name__ == "__main__":
    main()
"""
Grid Sentinel — Investigation Console (Person B)

Frontend redesign only. Backend contract is unchanged from the original
dashboard: same Detection Output Schema (SCHEMA.md), same data sources,
same detection logic (all of that lives in Person A's pipeline.py).

===========================================================================
SWAP-IN POINT: once Person A's real pipeline output changes location, this
is the only thing to touch:
===========================================================================
"""

FLAGGED_ACCOUNTS_SOURCE = "../data/flagged_accounts.json"
CONSUMPTION_DATA_SOURCE = "../data/consumption_timeseries.csv"

# ===========================================================================

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Grid Sentinel", layout="wide")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Locked pitch framing: never say "theft" -- a neutral possible-cause label
# for human review instead.
POSSIBLE_CAUSE = {
    "spike": "Possible unexpected load or equipment issue",
    "drop": "Possible meter fault, wiring issue, or unrecorded load change",
    "pattern_break": "Possible change in usage behavior or unauthorized load",
    "flatline": "Possible meter malfunction or communication outage",
}

ANOMALY_LABEL = {
    "spike": "SPIKE",
    "drop": "DROP",
    "pattern_break": "PATTERN BREAK",
    "flatline": "FLATLINE",
}


def risk_tier(score: int):
    """Three tiers, thresholds chosen for display grouping only -- not a
    detection signal, just how we group real risk_score values on screen."""
    if score >= 85:
        return "CRITICAL", "var(--c-red)"
    if score >= 60:
        return "WARNING", "var(--c-amber)"
    return "ELEVATED", "var(--c-teal)"


def format_duration(td: pd.Timedelta) -> str:
    total_min = max(int(td.total_seconds() // 60), 0)
    h, m = divmod(total_min, 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_flagged_accounts(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported flagged accounts file type: {ext}")

    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])
    return df


@st.cache_data
def load_consumption(path: str):
    """Returns None if the raw series file isn't available -- the app
    falls back to a simpler view rather than crashing."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


try:
    flagged_df = load_flagged_accounts(FLAGGED_ACCOUNTS_SOURCE)
except FileNotFoundError as e:
    st.error(f"Couldn't find the flagged accounts file: {e}")
    st.stop()

consumption_df = load_consumption(CONSUMPTION_DATA_SOURCE)
HAS_TIMESERIES = consumption_df is not None

DATA_START = consumption_df["timestamp"].min() if HAS_TIMESERIES else None
DATA_END = consumption_df["timestamp"].max() if HAS_TIMESERIES else None
N_ACCOUNTS_MONITORED = consumption_df["account_id"].nunique() if HAS_TIMESERIES else flagged_df["account_id"].nunique()


# ---------------------------------------------------------------------------
# Derived, real-data-only analysis helpers (frontend computation for
# visualization -- does not touch or duplicate detection logic)
# ---------------------------------------------------------------------------

def time_of_day_minutes(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def compute_baseline_profile(series: pd.DataFrame, window_start, window_end) -> pd.Series:
    """This account's own typical consumption by time-of-day, computed with
    the flagged window excluded so the anomaly doesn't bias its own
    baseline. Real historical data only -- index is minutes-since-midnight,
    value is mean kWh at that time of day."""
    if series.empty:
        return pd.Series(dtype=float)
    outside = series[(series["timestamp"] < window_start) | (series["timestamp"] > window_end)]
    if outside.empty:
        outside = series
    tod = time_of_day_minutes(outside["timestamp"])
    return outside.groupby(tod)["consumption_kwh"].mean()


def minutes_to_label(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h:02d}:{mm:02d}"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0A0E13;
  --panel: #10151C;
  --panel-alt: #141B22;
  --border: #1E262E;
  --text: #E7EDF2;
  --muted: #7E8B96;
  --c-teal: #2DD4BF;
  --c-amber: #F2B84B;
  --c-red: #EF4444;
  --c-blue: #38BDF8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--bg); }
.stApp, .stApp p, .stApp span, .stApp label { color: var(--text); }

.mono { font-family: 'JetBrains Mono', monospace; }

section[data-testid="stSidebar"] {
  background-color: var(--panel);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; color: var(--muted) !important; }

hr { border-color: var(--border) !important; }

.stButton>button {
  border-radius: 3px;
  border: 1px solid var(--border);
  background-color: var(--panel-alt);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  letter-spacing: 0.03em;
  padding: 0.35rem 0.7rem;
}
.stButton>button:hover { border-color: var(--c-teal); color: var(--c-teal); }
.stButton>button[kind="primary"] {
  background-color: rgba(45, 212, 191, 0.12);
  border-color: var(--c-teal);
  color: var(--c-teal);
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 3px !important;
  border-color: var(--border) !important;
  background-color: var(--panel-alt);
}

.brand {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--text);
  padding-bottom: 0.1rem;
}
.brand-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: var(--muted);
  letter-spacing: 0.08em;
}

.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: var(--muted);
}

.status-pill {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.12rem 0.5rem;
  border-radius: 2px;
  border: 1px solid currentColor;
}

.row-account {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.05rem;
  font-weight: 600;
}

.row-reason {
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.4;
}

.row-cause {
  font-size: 0.78rem;
  color: var(--muted);
}

.metric-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.metric-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text);
}

.evidence-quote {
  border-left: 2px solid var(--c-teal);
  padding: 0.6rem 0.9rem;
  background-color: rgba(45, 212, 191, 0.06);
  font-size: 0.92rem;
  margin: 0.6rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "view" not in st.session_state:
    st.session_state.view = "queue"
if "selected_window" not in st.session_state:
    st.session_state.selected_window = None


def goto_analysis(window_idx):
    st.session_state.selected_window = window_idx
    st.session_state.view = "analysis"


# ---------------------------------------------------------------------------
# Sidebar: brand, nav, filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">⚡ GRID SENTINEL</div>'
        '<div class="brand-sub">INVESTIGATION CONSOLE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "01 · INVESTIGATION QUEUE",
        width="stretch",
        type="primary" if st.session_state.view == "queue" else "secondary",
    ):
        st.session_state.view = "queue"

    if st.button(
        "02 · ACCOUNT ANALYSIS",
        width="stretch",
        disabled=st.session_state.selected_window is None,
        type="primary" if st.session_state.view == "analysis" else "secondary",
    ):
        st.session_state.view = "analysis"

    st.divider()

    if st.session_state.view == "queue":
        st.markdown('<div class="eyebrow">FILTERS</div>', unsafe_allow_html=True)
        anomaly_options = sorted(flagged_df["anomaly_type"].unique())
        selected_types = st.multiselect(
            "Anomaly type", options=anomaly_options, default=anomaly_options, label_visibility="collapsed"
        )
        min_risk = st.slider("Minimum risk score", min_value=0, max_value=100, value=0, step=1)
        search_id = st.text_input("Search account ID", placeholder="e.g. MT_003")
        sort_by = st.selectbox("Sort by", ["Risk score", "Confidence", "Persistence"])
        st.divider()

    with st.expander("Methodology & validation"):
        st.markdown(
            "Flags are statistical patterns, not confirmed theft — each "
            "comes with a confidence score and a possible-cause label for "
            "human review.\n\n"
            "**Validation:** tuned and tested against synthetically "
            "injected anomalies. The public UCI smart-meter dataset has no "
            "confirmed real-world theft labels, so this measures detection "
            "quality against known injected patterns, not real-world "
            "outcomes.\n\n"
            "**Note:** the current pipeline uses risk score as a stand-in "
            "for confidence — the two will diverge as scoring is refined."
        )


# ---------------------------------------------------------------------------
# QUEUE VIEW
# ---------------------------------------------------------------------------

if st.session_state.view == "queue":
    filtered_df = flagged_df[
        flagged_df["anomaly_type"].isin(selected_types) & (flagged_df["risk_score"] >= min_risk)
    ]
    if search_id:
        filtered_df = filtered_df[filtered_df["account_id"].str.contains(search_id, case=False, na=False)]

    sort_col = {"Risk score": "risk_score", "Confidence": "confidence", "Persistence": None}[sort_by]
    if sort_col:
        filtered_df = filtered_df.sort_values(sort_col, ascending=False)
    else:
        filtered_df = filtered_df.assign(_dur=(filtered_df["window_end"] - filtered_df["window_start"]))
        filtered_df = filtered_df.sort_values("_dur", ascending=False).drop(columns="_dur")

    date_range = f"{DATA_START.date()} → {DATA_END.date()}" if HAS_TIMESERIES else "unavailable"
    st.markdown(
        f'<span class="eyebrow">DATASET: UCI SMART METER · BATCH ANALYSIS &nbsp;·&nbsp; '
        f'WINDOW: {date_range} &nbsp;·&nbsp; ACCOUNTS MONITORED: {N_ACCOUNTS_MONITORED}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"## PRIORITY INVESTIGATIONS")
    st.markdown(
        f'<span class="eyebrow">SHOWING {len(filtered_df)} OF {len(flagged_df)} FLAGGED WINDOWS</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    if filtered_df.empty:
        st.info("No flagged windows match the current filters.")
    else:
        for rank, (idx, row) in enumerate(filtered_df.iterrows(), start=1):
            tier_label, tier_color = risk_tier(row["risk_score"])
            duration = format_duration(row["window_end"] - row["window_start"])
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.3, 3.4, 1.6, 0.9])
                with c1:
                    st.markdown(
                        f'<span class="status-pill" style="color:{tier_color}">{tier_label}</span><br>'
                        f'<span class="row-account">{row["account_id"]}</span><br>'
                        f'<span class="eyebrow">{ANOMALY_LABEL.get(row["anomaly_type"], row["anomaly_type"].upper())} · #{rank}</span>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(f'<div class="row-reason">{row["reason"]}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="row-cause">{POSSIBLE_CAUSE.get(row["anomaly_type"], "Unknown")}</div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f'<span class="metric-label">RISK</span><br>'
                        f'<span class="metric-value" style="color:{tier_color}">{row["risk_score"]}</span> '
                        f'<span class="eyebrow">/100</span><br>'
                        f'<span class="metric-label">CONFIDENCE</span> '
                        f'<span class="mono">{row["confidence"]}%</span><br>'
                        f'<span class="metric-label">PERSISTENCE</span> '
                        f'<span class="mono">{duration}</span>',
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.write("")
                    if st.button("INVESTIGATE →", key=f"open_{idx}"):
                        goto_analysis(idx)
                        st.rerun()


# ---------------------------------------------------------------------------
# ANALYSIS VIEW
# ---------------------------------------------------------------------------

else:
    if st.session_state.selected_window not in flagged_df.index:
        st.session_state.view = "queue"
        st.rerun()

    account_row = flagged_df.loc[st.session_state.selected_window]
    selected_account = account_row["account_id"]
    tier_label, tier_color = risk_tier(account_row["risk_score"])
    duration = format_duration(account_row["window_end"] - account_row["window_start"])

    if account_row["baseline_value"]:
        deviation_pct = (account_row["actual_value"] - account_row["baseline_value"]) / account_row["baseline_value"] * 100
    else:
        deviation_pct = None

    if st.button("← BACK TO QUEUE"):
        st.session_state.view = "queue"
        st.rerun()

    st.markdown(
        f'<span class="status-pill" style="color:{tier_color}">{tier_label}</span> '
        f'<span class="eyebrow">{ANOMALY_LABEL.get(account_row["anomaly_type"], account_row["anomaly_type"].upper())}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'## ACCOUNT {selected_account}')
    st.markdown(
        f'<span class="eyebrow">WINDOW: {account_row["window_start"]} → {account_row["window_end"]} '
        f'&nbsp;·&nbsp; DURATION: {duration}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    account_series = (
        consumption_df[consumption_df["account_id"] == selected_account].sort_values("timestamp")
        if HAS_TIMESERIES
        else pd.DataFrame()
    )

    chart_col, evidence_col = st.columns([2.3, 1])

    # -- Evidence panel -----------------------------------------------------
    with evidence_col:
        st.markdown('<div class="eyebrow">WHY WAS THIS FLAGGED?</div>', unsafe_allow_html=True)
        st.write("")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<span class="metric-label">OBSERVED</span><br>'
                f'<span class="metric-value">{account_row["actual_value"]:.2f}</span> '
                f'<span class="eyebrow">kWh</span>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<span class="metric-label">BASELINE</span><br>'
                f'<span class="metric-value">{account_row["baseline_value"]:.2f}</span> '
                f'<span class="eyebrow">kWh</span>',
                unsafe_allow_html=True,
            )

        st.write("")
        m3, m4 = st.columns(2)
        with m3:
            dev_str = f"{deviation_pct:+.0f}%" if deviation_pct is not None else "n/a"
            st.markdown(
                f'<span class="metric-label">DEVIATION</span><br>'
                f'<span class="metric-value" style="color:{tier_color}">{dev_str}</span>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<span class="metric-label">CONFIDENCE</span><br>'
                f'<span class="metric-value">{account_row["confidence"]}%</span>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="evidence-quote">{account_row["reason"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="metric-label">POSSIBLE EXPLANATION</span><br>'
            f'<span class="row-cause">{POSSIBLE_CAUSE.get(account_row["anomaly_type"], "Unknown")}</span>',
            unsafe_allow_html=True,
        )

    # -- Consumption chart, zoomed to the anomaly --------------------------
    with chart_col:
        st.markdown('<div class="eyebrow">CONSUMPTION — FOCUSED ON FLAGGED WINDOW</div>', unsafe_allow_html=True)

        if not HAS_TIMESERIES:
            st.info(
                "Full consumption history isn't available from the detection "
                "pipeline for this account — showing actual vs. reported "
                "baseline instead."
            )
            fallback_fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Baseline", "Actual"],
                        y=[account_row["baseline_value"], account_row["actual_value"]],
                        marker_color=["#7E8B96", tier_color],
                    )
                ]
            )
            fallback_fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E7EDF2"),
                yaxis_title="Consumption (kWh)",
                showlegend=False,
            )
            st.plotly_chart(fallback_fig, width="stretch")
        elif account_series.empty:
            st.warning("No consumption series found for this account.")
        else:
            window_start, window_end = account_row["window_start"], account_row["window_end"]
            span = window_end - window_start
            pad = max(pd.Timedelta(hours=12), span)
            x_min = max(window_start - pad, account_series["timestamp"].min())
            x_max = min(window_end + pad, account_series["timestamp"].max())

            zoomed = account_series[(account_series["timestamp"] >= x_min) & (account_series["timestamp"] <= x_max)]
            profile = compute_baseline_profile(account_series, window_start, window_end)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=zoomed["timestamp"],
                    y=zoomed["consumption_kwh"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="#38BDF8", width=1.6),
                )
            )
            if not profile.empty:
                expected_y = time_of_day_minutes(zoomed["timestamp"]).map(profile)
                fig.add_trace(
                    go.Scatter(
                        x=zoomed["timestamp"],
                        y=expected_y,
                        mode="lines",
                        name="Typical (this account's own history)",
                        line=dict(color="#7E8B96", width=1.2, dash="dot"),
                    )
                )
            fig.add_hline(
                y=account_row["baseline_value"],
                line=dict(color="#F2B84B", width=1, dash="dash"),
                annotation_text="Reported baseline",
                annotation_font_color="#F2B84B",
            )
            fig.add_vrect(
                x0=window_start,
                x1=window_end,
                fillcolor=tier_color,
                opacity=0.14,
                line_width=0,
                annotation_text="Flagged window",
                annotation_position="top left",
            )
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E7EDF2"),
                xaxis=dict(gridcolor="#1E262E"),
                yaxis=dict(gridcolor="#1E262E", title="Consumption (kWh)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # -- Behavioral fingerprint ----------------------------------------------
    st.markdown('<div class="eyebrow">BEHAVIORAL FINGERPRINT — THIS ACCOUNT\'S OWN NORMAL VS. THE FLAGGED DAY</div>', unsafe_allow_html=True)

    if not HAS_TIMESERIES or account_series.empty:
        st.info("Full consumption history isn't available for this account, so a daily fingerprint can't be computed.")
    else:
        profile = compute_baseline_profile(account_series, account_row["window_start"], account_row["window_end"])
        if profile.empty:
            st.info("Not enough history outside the flagged window to compute a typical-day profile.")
        else:
            flagged_day = account_row["window_start"].normalize()
            day_rows = account_series[
                (account_series["timestamp"] >= flagged_day)
                & (account_series["timestamp"] < flagged_day + pd.Timedelta(days=1))
            ].copy()

            if day_rows.empty:
                st.info("No same-day readings available to compare against the typical profile.")
            else:
                day_rows["tod"] = time_of_day_minutes(day_rows["timestamp"])
                profile_sorted = profile.sort_index()

                fp_fig = go.Figure()
                fp_fig.add_trace(
                    go.Scatter(
                        x=[minutes_to_label(m) for m in profile_sorted.index],
                        y=profile_sorted.values,
                        mode="lines",
                        name="Typical day (baseline)",
                        line=dict(color="#7E8B96", width=1.4, dash="dot"),
                    )
                )
                fp_fig.add_trace(
                    go.Scatter(
                        x=[minutes_to_label(m) for m in day_rows["tod"]],
                        y=day_rows["consumption_kwh"],
                        mode="lines",
                        name="Flagged day (actual)",
                        line=dict(color=tier_color, width=1.8),
                    )
                )
                fp_fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E7EDF2"),
                    xaxis=dict(gridcolor="#1E262E", title="Time of day", nticks=12),
                    yaxis=dict(gridcolor="#1E262E", title="Consumption (kWh)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
                )
                st.plotly_chart(fp_fig, width="stretch")
                st.markdown(
                    '<span class="row-cause">Typical day is this account\'s own average consumption at each '
                    'time of day, computed from its history with the flagged window excluded. Flagged day is '
                    'this account\'s actual readings on the day the window falls in.</span>',
                    unsafe_allow_html=True,
                )
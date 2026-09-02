# Grid Sentinel — Interface Contract

Locked at hour 0. Do not change without syncing with the whole team —
this is what lets all three of us build in parallel without blocking
on each other.

## Raw Data Schema (Person A's input)

Source: UCI ElectricityLoadDiagrams2011-2014 (370 clients, 15-min intervals)

- `account_id`: string
- `timestamp`: datetime (15-min intervals)
- `consumption_kwh`: float
  - **Note:** UCI raw values are in kW. Convert to kWh by dividing by 4
    (15-min interval = 1/4 hour). Do this once at ingestion, document it,
    don't re-convert downstream.

## Detection Output Schema (Person A produces this → Person B consumes this)

```json
{
  "account_id": "string",
  "window_start": "datetime",
  "window_end": "datetime",
  "risk_score": "int (0-100)",
  "confidence": "int (0-100)",
  "anomaly_type": "spike | drop | pattern_break | flatline",
  "reason": "string — human-readable, e.g. 'Consumption is 2.7x higher than normal Saturday evening baseline'",
  "actual_value": "float",
  "baseline_value": "float"
}
```

Person A should export this as a list of these objects — either a JSON
file (`flagged_accounts.json`) or a pandas DataFrame with these exact
column names saved as CSV. Person B builds against mock data matching
this shape until real data is ready, then swaps the source in one line.

## Detector Architecture (for Person C's slide + everyone's understanding)

```
            CONSUMPTION DATA
                   |
     +-------------+-------------+
     |             |             |
  Z-score    Isolation Forest  Flatline
     |             |             |
     +-------------+-------------+
                   |
            COMBINED SCORE
                   |
           PERSISTENCE FILTER
        (must persist 3+ windows)
                   |
          RISK / CONFIDENCE
                   |
           "WHY FLAGGED?"
```

- **Z-score** → catches sudden extreme spikes/drops
- **Isolation Forest** → catches unusual pattern combinations, not
  necessarily extreme values
- **Flatline detector** → catches suspiciously constant readings
  (meter/data issue, not necessarily theft)

## Pitch Framing (locked — do not deviate)

- Never say "detects theft." Say: **"anomaly detection and risk
  assessment"** with a possible-cause label (e.g. "possible unexpected
  load / equipment issue").
- Never imply the model trained on Delhi-specific data. It's a public
  UCI smart-meter dataset. Delhi's grid is the problem's framing and
  deployment context, not the training source. Use: *"The challenge is
  framed around Delhi's growing electricity demand, but our prototype
  demonstrates the detection methodology on a public smart-meter
  dataset."*
- Be upfront that validation is against **synthetically injected**
  anomalies (since UCI has no real theft/anomaly labels) — state this
  as a known limitation, not something to hide.

## Repo structure (suggested)

```
grid-sentinel/
├── SCHEMA.md              <- this file
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                <- UCI dataset (gitignored if large)
│   └── flagged_accounts.json  <- Person A's output
├── detection/               <- Person A's code
│   └── pipeline.py
├── dashboard/                <- Person B's code
│   └── app.py
└── docs/                     <- Person C's slides/validation
    ├── validation.py
    └── pitch_outline.md
```

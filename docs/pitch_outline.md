# Grid Sentinel — Pitch Outline

## 1. Problem framing (Delhi context)
- Delhi's grid reacts to overloads/losses after the fact; rooftop solar adds
  volatility DISCOMs can't yet forecast for.
- Narrow framing: undetected abnormal consumption (possible theft, faulty
  meters, unauthorized load) distorts demand forecasting and costs revenue —
  this is the piece we're solving.

## 2. Approach — architecture slide
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
- **Z-score** → sudden extreme spikes/drops vs. a weekday+hour baseline
- **Isolation Forest** → unusual pattern combinations, not necessarily extreme
- **Flatline** → suspiciously constant readings (meter/data issue)
- Say once, clearly: *"The challenge is framed around Delhi's growing
  electricity demand, but our prototype demonstrates the detection
  methodology on a public smart-meter dataset (UCI ElectricityLoadDiagrams,
  370 clients)."*

## 3. Live demo
- Dashboard shows flagged accounts sorted by risk_score: confidence,
  anomaly_type, human-readable reason, actual vs. baseline value.
- Walk through 2–3 live cases: one high-confidence spike, one flatline, one
  filtered-out blip (shows the persistence filter working).

## 4. Results — with honest limitations
- Report precision / recall / F1 / false-positive rate, before vs. after the
  persistence filter (numbers come from validation.py's console output).
- State plainly: *"Validated against synthetically injected anomalies with
  known ground truth, not confirmed real-world theft cases — the UCI dataset
  has no theft labels. This is a known limitation, not something we're
  hiding."*

## 5. Scalability answer
- Per-account baseline + detection is embarrassingly parallel — each
  account's history is independent, so it scales horizontally by
  partitioning on account_id.
- Isolation Forest is fit per-account in the demo; at production scale, fit
  a shared model on engineered features across accounts (or cluster similar
  consumption profiles and fit per-cluster).
- Persistence filter and z-score are cheap, near-real-time computations;
  the expensive part (IF retraining) can run on a slower cadence (e.g.
  nightly) decoupled from real-time scoring.

---

## Anticipated judge questions + honest answers

**"You're claiming X% accuracy — how do we know that's real?"**
Precision/recall are measured against synthetic anomalies we injected
ourselves with known ground truth, not real confirmed cases — so these
numbers show the methodology works, not that we've caught real theft. We're
upfront about that distinction.

**"Isn't this trained on Delhi data?"**
No — trained and tested on a public UCI smart-meter dataset. Delhi is the
deployment context and problem framing, not the training source.

**"How does this scale to millions of meters?"**
See scalability answer above: per-account parallelism, decoupled retraining
cadence, cheap real-time detectors.

**"Why is this different from a generic anomaly-detection demo?"**
Most demos stop at "we ran Isolation Forest and got a score." We measured a
specific operational cost — false positives — and quantified how much a
persistence filter reduces them, with a stated recall trade-off. That's the
difference between a model and a decision system a utility could act on.

---

## Framing rules (do not deviate)
- Never say "detects theft" — say "anomaly detection and risk assessment"
  with a possible-cause label.
- Never imply Delhi-specific training data — it's the public UCI dataset;
  Delhi is framing/deployment context only.
- Always disclose that validation uses synthetic/injected anomalies, not
  confirmed real-world theft cases.

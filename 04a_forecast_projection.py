#!/usr/bin/env python3
"""
04_forecast_projection.py  --  MAD-Markov Chain pipeline, step 4.

Predict the NEXT REGIME CHANGE -- direction (up/down) and timing (days-to-change)
-- and score three predictors head-to-head, out-of-sample:

  base_rate      : unconditional jump chain from 03 (majority direction per
                   regime). The benchmark. Coin-flip-ish at the hub.
  z_conditional  : jump chain split by whether z sits in the LOWER or UPPER half
                   of its band (are we near the reversion edge or the
                   continuation edge?). Tests "does where-in-the-band help?"
  forecast       : project the flat-price MAD path forward, hold sigma at its
                   current value, and take the FIRST threshold crossing as the
                   predicted direction & timing. Uses the continuous position
                   the discrete chain can't see. No fitted parameters.

Honesty:
  * All features (regime, z, sigma, days-in-regime, flat-price forecast) use only
    data available at day t. Trades/labels look forward only to define the actual
    next change.
  * Chronological split: base rates and z-splits are fit on TRAIN only; every
    accuracy number is on the held-out TEST set.

Outputs: results/04a_regime_predictions.csv (per test day), results/04a_prediction_scores.csv (summary).

Usage:
    pip install pandas numpy
    python3 04_forecast_projection.py
"""

from __future__ import annotations
from collections import defaultdict
import os

import numpy as np
import pandas as pd


INPUT_FILE = "results/02_spy_mad.csv"
ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
SMA_WINDOW = 20
HORIZON = 40          # forecast horizon for threshold crossing (covers long dwells)
TRAIN_FRAC = 0.6      # chronological split


# ------------------------------------------------------------------ features
def load(path):
    df = pd.read_csv(path)
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df = df.reset_index(drop=True)
    return df


def flat_price_mad(close: np.ndarray, t: int, horizon: int, w: int) -> np.ndarray:
    p0 = close[t]
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        lo = t + h - (w - 1)
        known = close[max(lo, 0):t + 1]
        sma = (known.sum() + (w - len(known)) * p0) / w
        out[h - 1] = (p0 - sma) / p0 * 100.0
    return out


def band_bounds(regime: int, sigma: float):
    """MAD-unit (lower, upper) thresholds of a regime's z-band. None = open side."""
    zlo = {-3: None, -2: -2, -1: -1, 1: 0, 2: 1, 3: 2}[regime]
    zhi = {-3: -2, -2: -1, -1: 0, 1: 1, 2: 2, 3: None}[regime]
    return (None if zlo is None else zlo * sigma,
            None if zhi is None else zhi * sigma)


def forecast_direction(close, t, regime, sigma, mad_now):
    """First threshold crossing of the flat-price path -> (dir, days) or (drift, HORIZON)."""
    lo, hi = band_bounds(regime, sigma)
    path = flat_price_mad(close, t, HORIZON, SMA_WINDOW)
    for h, m in enumerate(path, start=1):
        if hi is not None and m > hi:
            return "up", h
        if lo is not None and m < lo:
            return "down", h
    return ("up" if path[-1] > mad_now else "down"), HORIZON  # no crossing -> net drift


def main():
    os.makedirs("results", exist_ok=True)
    df = load(INPUT_FILE)
    close = df["Close"].to_numpy(dtype=float)
    reg = df["regime"].to_numpy()
    z = df["z"].to_numpy(dtype=float)
    sig = df["sigma"].to_numpy(dtype=float)
    n = len(df)

    # runs -> for every in-regime day, actual (direction, days-to-change) + days-in
    samples = []  # dict per day
    i = 0
    # find first classified index
    while i < n and (pd.isna(reg[i])):
        i += 1
    while i < n:
        j = i
        while j + 1 < n and reg[j + 1] == reg[i]:
            j += 1
        nxt = reg[j + 1] if j + 1 < n else None
        if nxt is not None:
            r = int(reg[i])
            direction = "up" if RANK[int(nxt)] > RANK[r] else "down"
            for t in range(i, j + 1):
                samples.append({"t": t, "regime": r, "z": z[t], "sigma": sig[t],
                                "days_in": t - i + 1, "actual_dir": direction,
                                "actual_days": (j + 1) - t})
        i = j + 1

    sdf = pd.DataFrame(samples)
    split = int(len(sdf) * TRAIN_FRAC)
    train, test = sdf.iloc[:split], sdf.iloc[split:]

    # ---- fit base rate + z-conditional on TRAIN ----
    base_dir, zmed, zcond_dir, med_remaining = {}, {}, {}, {}
    for r in ORDER:
        tr = train[train["regime"] == r]
        if len(tr) == 0:
            base_dir[r] = "up"; zmed[r] = 0.0
            zcond_dir[r] = {"low": "up", "high": "up"}; med_remaining[r] = 1.0
            continue
        base_dir[r] = "up" if (tr["actual_dir"] == "up").mean() >= 0.5 else "down"
        m = tr["z"].median(); zmed[r] = m
        for half, sub in (("low", tr[tr["z"] <= m]), ("high", tr[tr["z"] > m])):
            zcond_dir.setdefault(r, {})[half] = (
                "up" if len(sub) and (sub["actual_dir"] == "up").mean() >= 0.5 else "down")
        med_remaining[r] = float(tr["actual_days"].median())

    # ---- predict on TEST ----
    rows = []
    for _, s in test.iterrows():
        r = s["regime"]; t = int(s["t"])
        p_base = base_dir[r]
        p_zc = zcond_dir[r]["low" if s["z"] <= zmed[r] else "high"]
        p_fc, fc_days = forecast_direction(close, t, r, s["sigma"], df["mad"].iloc[t])
        rows.append({"date": df["Date"].iloc[t], "regime": int(r), "z": round(s["z"], 3),
                     "days_in": int(s["days_in"]), "actual_dir": s["actual_dir"],
                     "actual_days": int(s["actual_days"]), "base_dir": p_base,
                     "zcond_dir": p_zc, "fcst_dir": p_fc, "fcst_days": int(fc_days),
                     "base_days": round(med_remaining[r], 1)})
    pred = pd.DataFrame(rows)
    pred.to_csv("results/04a_regime_predictions.csv", index=False)

    # ---- score (persist EVERY metric, not just stdout) ----
    preds = (("base_rate", "base_dir"), ("z_conditional", "zcond_dir"), ("forecast", "fcst_dir"))

    def acc(col, mask=None):
        d = pred if mask is None else pred[mask]
        if len(d) == 0:
            return float("nan"), 0
        return 100 * (d[col] == d["actual_dir"]).mean(), len(d)

    scores = []
    for name, col in preds:                               # overall direction accuracy
        a, nn = acc(col)
        scores.append({"metric": "direction_acc_pct", "predictor": name,
                       "scope": "overall", "value": round(a, 1), "n": nn})
    for r in ORDER:                                       # per-regime direction accuracy
        for name, col in preds:
            a, nn = acc(col, pred["regime"] == r)
            scores.append({"metric": "direction_acc_pct", "predictor": name,
                           "scope": f"regime_{r}",
                           "value": None if nn == 0 else round(a, 1), "n": nn})
    for name, col in (("base_rate", "base_days"), ("forecast", "fcst_days")):  # timing
        mae = (pred[col] - pred["actual_days"]).abs().mean()
        scores.append({"metric": "timing_mae_days", "predictor": name,
                       "scope": "overall", "value": round(float(mae), 2), "n": len(pred)})
    pd.DataFrame(scores).to_csv("results/04a_prediction_scores.csv", index=False)

    # ---- stdout summary (human-readable view of what's now on disk) ----
    print(f"Test set: {len(pred):,} in-regime days "
          f"({pred['date'].iloc[0]} -> {pred['date'].iloc[-1]})\n")
    print("DIRECTION ACCURACY (predict next change up/down):")
    for name, col in preds:
        a, _ = acc(col)
        print(f"  {name:<15} {a:5.1f}%")
    print("\n  from regime 1 (the hub, base rate ~coin-flip):")
    for name, col in preds:
        a, nn = acc(col, pred["regime"] == 1)
        print(f"    {name:<15} {a:5.1f}%   (n={nn})")
    print(f"\nTIMING (mean abs error, days-to-change):")
    print(f"  base_rate (median dwell) {(pred['base_days']-pred['actual_days']).abs().mean():.2f}")
    print(f"  forecast (1st crossing)  {(pred['fcst_days']-pred['actual_days']).abs().mean():.2f}")
    print("\nWrote results/04a_regime_predictions.csv, results/04a_prediction_scores.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
05_transition_model.py  --  MAD-Markov Chain pipeline, step 5: the forecaster.

Turns the validated finding into a live, walk-forward regime-change model:

  DIRECTION  P(next change up | regime, z), calibrated, with an ADAPTIVE baseline:
    at each day we fit per-regime logistic P(up) ~ z on a trailing window of that
    regime's *already-resolved* visits. The window's up-rate is the slowly-updating
    regime bias (the level that drifts with the era); the z-slope is the structural
    gradient that replicated across eras in 04b. Extremes (-3,+3) are ~deterministic
    so they use the trailing up-rate directly.

  TIMING  expected days-to-change, from the semi-Markov dwell conditioned on how
    long you've ALREADY been in the regime (the decreasing hazard from 03):
    median(L - days_in | L > days_in) over recent visits.

  SCORING  everything is walk-forward (a day is predicted using only visits that
    resolved before it -- no look-ahead). On the held-out tail we compare the
    z-conditioned model to the SAME baseline without z, using Brier score and
    log-loss (which reward sharper, calibrated probabilities -- the real edge,
    since hard up/down accuracy ties by construction).

Outputs:
  05_regime_forecast.csv  -- per day: p_up, p_down, predicted next regime, exp days
  05_forecast_scores.csv  -- Brier / log-loss / accuracy / timing MAE, model vs base

Usage:
    pip install pandas numpy
    python3 05_transition_model.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


INPUT_FILE = "results/02_spy_mad.csv"
FORECAST_OUTPUT = "results/05_regime_forecast.csv"
SCORE_OUTPUT = "results/05_forecast_scores.csv"

ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
INTERIOR = {-2, -1, 1, 2}
LABELS = {-3: "ext_neg", -2: "negative", -1: "mild_neg",
          1: "mild_pos", 2: "positive", 3: "ext_pos"}

WINDOW_OBS = 500        # trailing day-observations per regime for the direction fit
WINDOW_VISITS = 150     # trailing visits for the timing model
MIN_N = 60              # minimum history before we emit a forecast
TEST_FRAC = 0.40        # hold out the most recent 40% for scoring
PCLIP = (0.02, 0.98)


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 30):
    """2-param logistic via Newton/IRLS: logit p = a + b*x."""
    a = 0.0
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a + b * x)))
        w = np.clip(p * (1 - p), 1e-6, None)
        ga, gb = np.sum(y - p), np.sum((y - p) * x)
        Haa, Hab, Hbb = np.sum(w), np.sum(w * x), np.sum(w * x * x)
        det = Haa * Hbb - Hab * Hab
        if abs(det) < 1e-9:
            break
        da = (Hbb * ga - Hab * gb) / det
        db = (Haa * gb - Hab * ga) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-7:
            break
    return a, b


def clip(p: float) -> float:
    return float(min(max(p, PCLIP[0]), PCLIP[1]))


def build_visits(path: str):
    df = pd.read_csv(path)
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    reg, z, dates = df["regime"].to_numpy(), df["z"].to_numpy(), df["Date"].to_numpy()
    n = len(df)

    visits = []
    i = 0
    while i < n and pd.isna(reg[i]):
        i += 1
    while i < n:
        j = i
        while j + 1 < n and reg[j + 1] == reg[i]:
            j += 1
        nxt = reg[j + 1] if j + 1 < n else None
        exit_dir = None
        if nxt is not None and not pd.isna(nxt):
            exit_dir = "up" if RANK[int(nxt)] > RANK[int(reg[i])] else "down"
        days = [{"date": dates[t], "z": float(z[t]), "days_in": t - i + 1,
                 "remaining": (j + 1) - t} for t in range(i, j + 1)]
        visits.append({"regime": int(reg[i]), "exit": exit_dir,
                       "length": j - i + 1, "days": days})
        i = j + 1
    return visits


def predict_direction(regime, z, obs):
    """obs: list of (z, up01). Returns (base_p, model_p) or (None, None)."""
    n = len(obs)
    if n < MIN_N:
        return None, None
    ups = sum(u for _, u in obs)
    base_p = (ups + 1) / (n + 2)                       # Laplace-smoothed, ignores z
    if regime in INTERIOR and 0 < ups < n:
        zs = np.array([zz for zz, _ in obs], dtype=float)
        ys = np.array([u for _, u in obs], dtype=float)
        a, b = logistic_fit(zs, ys)
        model_p = 1.0 / (1.0 + np.exp(-(a + b * z)))
    else:
        model_p = base_p
    return clip(base_p), clip(model_p)


def predict_time(days_in, lengths):
    if not lengths:
        return None, None
    base_t = float(np.median(lengths))                 # ignores days_in
    rem = [L - days_in for L in lengths if L > days_in]
    model_t = float(np.median(rem)) if rem else 1.0     # duration-conditioned
    return model_t, base_t


def main():
    os.makedirs("results", exist_ok=True)
    visits = build_visits(INPUT_FILE)
    pool = {r: [] for r in ORDER}       # resolved (z, up01) day-obs
    lengths = {r: [] for r in ORDER}    # resolved visit lengths
    rows = []

    for v in visits:
        r = v["regime"]
        for d in v["days"]:
            base_p, model_p = predict_direction(r, d["z"], pool[r][-WINDOW_OBS:])
            model_t, base_t = predict_time(d["days_in"], lengths[r][-WINDOW_VISITS:])
            if model_p is None:
                continue
            up_nb = ORDER[RANK[r] + 1] if RANK[r] + 1 < len(ORDER) else None
            dn_nb = ORDER[RANK[r] - 1] if RANK[r] - 1 >= 0 else None
            nxt = up_nb if model_p >= 0.5 else dn_nb
            rows.append({"date": d["date"], "regime": r, "label": LABELS[r],
                         "z": round(d["z"], 3), "days_in": d["days_in"],
                         "p_up": round(model_p, 3), "p_down": round(1 - model_p, 3),
                         "pred_next_regime": nxt, "exp_days_to_change": round(model_t, 1),
                         "base_p_up": round(base_p, 3), "base_days": round(base_t, 1),
                         "actual_dir": v["exit"], "actual_days": d["remaining"]})
        if v["exit"] is not None:                       # add only resolved visits
            up = 1 if v["exit"] == "up" else 0
            pool[r].extend((d["z"], up) for d in v["days"])
            lengths[r].append(v["length"])

    fc = pd.DataFrame(rows)
    fc.to_csv(FORECAST_OUTPUT, index=False)

    # ---- score on held-out tail (rows with a known actual) ----
    scored = fc[fc["actual_dir"].notna()].reset_index(drop=True)
    cut = int(len(scored) * (1 - TEST_FRAC))
    test = scored.iloc[cut:].copy()
    y = (test["actual_dir"] == "up").astype(float).to_numpy()

    def brier(p): return float(np.mean((p - y) ** 2))
    def logloss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    def hard_acc(p): return float(np.mean((p >= 0.5) == (y >= 0.5)) * 100)

    mp, bp = test["p_up"].to_numpy(), test["base_p_up"].to_numpy()
    tim_model = float((test["exp_days_to_change"] - test["actual_days"]).abs().mean())
    tim_base = float((test["base_days"] - test["actual_days"]).abs().mean())

    scores = [
        {"metric": "brier", "model": round(brier(mp), 4), "base": round(brier(bp), 4)},
        {"metric": "log_loss", "model": round(logloss(mp), 4), "base": round(logloss(bp), 4)},
        {"metric": "hard_accuracy_pct", "model": round(hard_acc(mp), 1), "base": round(hard_acc(bp), 1)},
        {"metric": "timing_mae_days", "model": round(tim_model, 2), "base": round(tim_base, 2)},
    ]
    pd.DataFrame(scores).to_csv(SCORE_OUTPUT, index=False)

    # ---- report ----
    print(f"Forecast rows: {len(fc):,}  |  scored (out-of-sample) tail: {len(test):,} "
          f"({test['date'].iloc[0]} -> {test['date'].iloc[-1]})\n")
    print("SCORECARD  (model = z-conditioned; base = same baseline, no z):")
    print(f"{'metric':<20}{'model':>10}{'base':>10}   winner")
    for s in scores:
        better = "model" if (s["metric"] == "hard_accuracy_pct" and s["model"] > s["base"]) \
            or (s["metric"] != "hard_accuracy_pct" and s["model"] < s["base"]) else "base/tie"
        print(f"{s['metric']:<20}{s['model']:>10}{s['base']:>10}   {better}")
    gain = 100 * (1 - brier(mp) / brier(bp))
    print(f"\nBrier improvement from z: {gain:.1f}%  (lower Brier = sharper, still-calibrated probs)")

    last = fc.iloc[-1]
    print(f"\n--- Live forecast, {last['date']} ---")
    print(f"  state: regime {last['regime']} ({last['label']}), z={last['z']}, "
          f"in regime {last['days_in']} days")
    print(f"  next change: {last['p_up']:.0%} UP -> {last['pred_next_regime'] if last['p_up']>=0.5 else ORDER[RANK[last['regime']]+1] if RANK[last['regime']]+1<len(ORDER) else 'n/a'}"
          f"   {last['p_down']:.0%} DOWN")
    print(f"  expected time to change: ~{last['exp_days_to_change']:.0f} trading days")
    print(f"\nWrote {FORECAST_OUTPUT}, {SCORE_OUTPUT}")


if __name__ == "__main__":
    main()

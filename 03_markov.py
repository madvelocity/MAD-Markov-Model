#!/usr/bin/env python3
"""
03_markov.py  --  MAD-Markov Chain pipeline, step 3: the empirical (semi-)Markov chain.

Reads spy_mad.csv (needs the `regime` column) and builds:

  1. transition_matrix.csv (+ transition_counts.csv)
        1-day P(regime_{t+1} | regime_t). Diagonal-heavy (regimes are persistent).

  2. jump_chain.csv (+ jump_counts.csv)
        The EMBEDDED chain: given the regime changes, P(next distinct regime | current).
        This answers "from 1, are we going up to 2 or back to -1?"  (zero diagonal.)

  3. dwell_times.csv
        Per regime: number of visits and the run-length distribution (how many
        consecutive days per visit). Answers "how long do we stay?"

  4. dwell_by_direction.csv
        Dwell per regime split by exit direction (up-exit vs down-exit). Answers
        "if we're going up to 2, how long do we sit in 1 first?"

  5. stationary_dist.csv
        Long-run share of time in each regime.

Also prints a memorylessness check: is the daily hazard of leaving a regime flat
(geometric / true Markov) or does it depend on how long you've been there?

Usage:
    pip install pandas numpy
    python3 03_markov.py
"""

from __future__ import annotations
from collections import defaultdict
import os

import numpy as np
import pandas as pd


INPUT_FILE = "results/02_spy_mad.csv"
ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
LABELS = {-3: "ext_neg", -2: "negative", -1: "mild_neg",
          1: "mild_pos", 2: "positive", 3: "ext_pos"}


def load_regimes(path: str) -> list[int]:
    df = pd.read_csv(path)
    reg = pd.to_numeric(df["regime"], errors="coerce").dropna().astype(int)
    return reg.tolist()


def runs_of(regs: list[int]):
    """Run-length encode into visits: (regime, length, next_regime_or_None)."""
    out = []
    i = 0
    while i < len(regs):
        j = i
        while j + 1 < len(regs) and regs[j + 1] == regs[i]:
            j += 1
        nxt = regs[j + 1] if j + 1 < len(regs) else None
        out.append((regs[i], j - i + 1, nxt))
        i = j + 1
    return out


def matrix_df(mat: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(mat, index=[f"{r} {LABELS[r]}" for r in ORDER],
                        columns=[str(r) for r in ORDER])


def main() -> None:
    os.makedirs("results", exist_ok=True)
    regs = load_regimes(INPUT_FILE)
    runs = runs_of(regs)
    n = len(ORDER)

    # ---- 1-day transition matrix ----
    counts = np.zeros((n, n))
    for a, b in zip(regs[:-1], regs[1:]):
        counts[RANK[a], RANK[b]] += 1
    P = np.divide(counts, counts.sum(axis=1, keepdims=True),
                  out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True) > 0)

    # ---- jump / embedded chain (transitions between distinct regimes) ----
    jc = np.zeros((n, n))
    for r, _, nxt in runs:
        if nxt is not None:
            jc[RANK[r], RANK[nxt]] += 1
    J = np.divide(jc, jc.sum(axis=1, keepdims=True),
                  out=np.zeros_like(jc), where=jc.sum(axis=1, keepdims=True) > 0)

    # ---- dwell times (completed visits only) ----
    dwell = defaultdict(list)
    dwell_dir = defaultdict(lambda: defaultdict(list))
    for r, ln, nxt in runs:
        if nxt is None:
            continue  # last (right-censored) visit
        dwell[r].append(ln)
        d = "up" if RANK[nxt] > RANK[r] else "down"
        dwell_dir[r][d].append(ln)

    # ---- stationary distribution (empirical time-share) ----
    vc = pd.Series(regs).value_counts()
    stat = {r: vc.get(r, 0) / len(regs) for r in ORDER}

    # =========================== write artifacts ===========================
    matrix_df(P).round(4).to_csv("results/03_transition_matrix.csv")
    matrix_df(counts).astype(int).to_csv("results/03_transition_counts.csv")
    matrix_df(J).round(4).to_csv("results/03_jump_chain.csv")
    matrix_df(jc).astype(int).to_csv("results/03_jump_counts.csv")

    dwell_rows = []
    for r in ORDER:
        d = np.array(dwell[r], dtype=float)
        dwell_rows.append({"regime": r, "label": LABELS[r], "n_visits": len(d),
                           "mean_days": round(d.mean(), 2) if len(d) else np.nan,
                           "median_days": round(float(np.median(d)), 1) if len(d) else np.nan,
                           "std_days": round(d.std(ddof=0), 2) if len(d) else np.nan,
                           "max_days": int(d.max()) if len(d) else 0})
    pd.DataFrame(dwell_rows).to_csv("results/03_dwell_times.csv", index=False)

    dir_rows = []
    for r in ORDER:
        for d in ("up", "down"):
            arr = np.array(dwell_dir[r][d], dtype=float)
            dir_rows.append({"regime": r, "label": LABELS[r], "exit": d,
                             "n_visits": len(arr),
                             "mean_days": round(arr.mean(), 2) if len(arr) else np.nan,
                             "median_days": round(float(np.median(arr)), 1) if len(arr) else np.nan})
    pd.DataFrame(dir_rows).to_csv("results/03_dwell_by_direction.csv", index=False)

    pd.DataFrame([{"regime": r, "label": LABELS[r], "pct": round(100 * stat[r], 2)}
                  for r in ORDER]).to_csv("results/03_stationary_dist.csv", index=False)

    # =========================== print summary ===========================
    pd.set_option("display.width", 120)
    print("1-DAY TRANSITION MATRIX  P(next | current):")
    print(matrix_df(P).round(3).to_string())
    print("\nJUMP CHAIN  P(next distinct regime | current change):")
    print(matrix_df(J).round(3).to_string())

    print("\nDWELL TIMES (completed visits):")
    print(pd.DataFrame(dwell_rows).to_string(index=False))

    print("\nDWELL BY EXIT DIRECTION:")
    print(pd.DataFrame(dir_rows).to_string(index=False))

    # focused answer for the regime-1 example
    r = 1
    up_p, dn_p = J[RANK[r], RANK[2]], J[RANK[r], RANK[-1]]
    up_d = np.median(dwell_dir[r]["up"]) if dwell_dir[r]["up"] else float("nan")
    dn_d = np.median(dwell_dir[r]["down"]) if dwell_dir[r]["down"] else float("nan")
    print(f"\n--- Your example, from regime 1 ---")
    print(f"  next change is UP to 2   : {100*up_p:.0f}%   (median {up_d:.0f} days in 1 first)")
    print(f"  next change is DOWN to -1: {100*dn_p:.0f}%   (median {dn_d:.0f} days in 1 first)")
    skip = 1 - up_p - dn_p
    if skip > 0.005:
        print(f"  (skips to a non-adjacent regime: {100*skip:.0f}%)")

    # memorylessness check: hazard of leaving vs days already spent
    print("\nMEMORYLESS CHECK  (P(leave next day | already in regime k days)):")
    print("  regime   1-5d   6-10d  11-20d   >20d   (flat => geometric/true-Markov)")
    for r in ORDER:
        lens = dwell[r]
        if len(lens) < 20:
            continue
        buckets = {"1-5": [], "6-10": [], "11-20": [], ">20": []}
        for L in lens:                       # a visit of length L => (L-1) 'stayed', 1 'left'
            for day in range(1, L + 1):
                left = 1 if day == L else 0
                b = ("1-5" if day <= 5 else "6-10" if day <= 10 else "11-20" if day <= 20 else ">20")
                buckets[b].append(left)
        hz = {k: (np.mean(v) if v else float("nan")) for k, v in buckets.items()}
        print(f"  {r:>3} {LABELS[r]:<9}"
              + "".join(f"{hz[k]:>7.2f}" for k in ("1-5", "6-10", "11-20", ">20")))

    cur = runs[-1]
    print(f"\nCurrent visit: regime {cur[0]} ({LABELS[cur[0]]}), {cur[1]} days and counting "
          f"(right-censored).")
    print("\nWrote 7 files to results/: 03_transition_matrix.csv, 03_transition_counts.csv, "
          "03_jump_chain.csv, 03_jump_counts.csv, 03_dwell_times.csv, "
          "03_dwell_by_direction.csv, 03_stationary_dist.csv")


if __name__ == "__main__":
    main()

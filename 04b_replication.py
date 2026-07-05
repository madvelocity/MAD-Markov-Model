#!/usr/bin/env python3
"""
04b_replication.py  --  MAD-Markov Chain pipeline, step 4b: does it replicate?

Stress-tests the core finding from 04a -- that P(next regime change is UP) rises
with where z sits inside the band -- two ways, and PERSISTS both to disk:

  1. Replication across independent eras.
     Split history into N_PERIODS equal chronological blocks. For each interior
     regime (-2,-1,1,2) and z-bin, compute P(up-exit) in each block. If the
     gradient rises monotonically in every block, it's structural, not a fluke.
     -> replication.csv  (regime, period, z-bin, p_up, n)

  2. Calibration, out-of-sample.
     Fit a P(up | regime, z-bin) lookup on the first TRAIN_FRAC of history, apply
     it to the held-out tail, and bucket predictions by predicted probability.
     If mean predicted ~= actual frequency in every bucket, the probabilities are
     numerically honest.
     -> calibration.csv  (pred_bucket, mean_pred, actual_up, n)

No look-ahead: z is known at t; exit direction only labels completed regime runs.

Usage:
    pip install pandas numpy
    python3 04b_replication.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


INPUT_FILE = "results/02_spy_mad.csv"
REP_OUTPUT = "results/04b_replication.csv"
CAL_OUTPUT = "results/04b_calibration.csv"
PLOT_OUTPUT = "results/04b_replication_calibration.png"
PERIOD_COLORS = ["#2a78d6", "#1baf7a", "#eb6834"]

ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
BANDS = {-2: (-2, -1), -1: (-1, 0), 1: (0, 1), 2: (1, 2)}   # interior regimes only
N_PERIODS = 3
N_BINS = 5           # z-bins per band for the replication table
TRAIN_FRAC = 0.6
N_CAL_BINS = 10      # fine z-bins for the calibration lookup
MIN_BIN_N = 8


def build_samples(path: str) -> pd.DataFrame:
    """Per-day (date, regime, z, exit_dir) for completed regime runs."""
    df = pd.read_csv(path)
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    reg = df["regime"].to_numpy()
    zz = df["z"].to_numpy()
    dates = df["Date"].to_numpy()
    n = len(df)

    rows = []
    i = 0
    while i < n and pd.isna(reg[i]):
        i += 1
    while i < n:
        j = i
        while j + 1 < n and reg[j + 1] == reg[i]:
            j += 1
        nxt = reg[j + 1] if j + 1 < n else None
        if nxt is not None and not pd.isna(nxt):
            d = "up" if RANK[int(nxt)] > RANK[int(reg[i])] else "down"
            for t in range(i, j + 1):
                rows.append({"date": dates[t], "regime": int(reg[i]), "z": zz[t], "exit": d})
        i = j + 1
    return pd.DataFrame(rows)


def replication_table(samples: pd.DataFrame) -> pd.DataFrame:
    blocks = np.array_split(samples, N_PERIODS)
    out = []
    for p, blk in enumerate(blocks):
        lo_date, hi_date = blk["date"].iloc[0], blk["date"].iloc[-1]
        for r, (lo, hi) in BANDS.items():
            sub = blk[blk["regime"] == r]
            edges = np.linspace(lo, hi, N_BINS + 1)
            for k in range(N_BINS):
                a, b = edges[k], edges[k + 1]
                seg = sub[(sub["z"] >= a) & ((sub["z"] < b) | ((k == N_BINS - 1) & (sub["z"] <= b)))]
                p_up = (seg["exit"] == "up").mean() if len(seg) >= MIN_BIN_N else np.nan
                out.append({"period": p + 1, "period_start": lo_date, "period_end": hi_date,
                            "regime": r, "z_lo": round(a, 3), "z_hi": round(b, 3),
                            "z_mid": round((a + b) / 2, 3),
                            "p_up": None if pd.isna(p_up) else round(float(p_up), 3),
                            "n": int(len(seg))})
    return pd.DataFrame(out)


def calibration_table(samples: pd.DataFrame) -> pd.DataFrame:
    split = int(len(samples) * TRAIN_FRAC)
    train, test = samples.iloc[:split], samples.iloc[split:]

    lookup = {}  # (regime) -> array of P(up) per fine z-bin
    for r, (lo, hi) in BANDS.items():
        tr = train[train["regime"] == r]
        edges = np.linspace(lo, hi, N_CAL_BINS + 1)
        tbl = []
        for k in range(N_CAL_BINS):
            a, b = edges[k], edges[k + 1]
            seg = tr[(tr["z"] >= a) & ((tr["z"] < b) | ((k == N_CAL_BINS - 1) & (tr["z"] <= b)))]
            tbl.append((seg["exit"] == "up").mean() if len(seg) else np.nan)
        lookup[r] = (edges, np.array(tbl, dtype=float))

    preds = []
    for _, s in test.iterrows():
        r = s["regime"]
        if r not in BANDS:
            continue
        edges, tbl = lookup[r]
        k = min(N_CAL_BINS - 1, max(0, int(np.searchsorted(edges, s["z"], side="right") - 1)))
        p = tbl[k]
        if not np.isnan(p):
            preds.append((p, 1 if s["exit"] == "up" else 0))
    pred = pd.DataFrame(preds, columns=["p", "y"])

    out = []
    for k in range(N_CAL_BINS):
        lo, hi = k / N_CAL_BINS, (k + 1) / N_CAL_BINS
        b = pred[(pred["p"] >= lo) & ((pred["p"] < hi) | ((k == N_CAL_BINS - 1) & (pred["p"] <= hi)))]
        if len(b):
            out.append({"pred_bucket": f"{int(lo*100)}-{int(hi*100)}%",
                        "mean_pred": round(float(b["p"].mean()), 3),
                        "actual_up": round(float(b["y"].mean()), 3), "n": int(len(b))})
    return pd.DataFrame(out)


def plot_results(rep: pd.DataFrame, cal: pd.DataFrame, path: str = PLOT_OUTPUT) -> None:
    """Small-multiples replication panels + calibration curve. Skips if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except Exception as exc:
        print(f"Skipping plot (matplotlib unavailable): {exc}")
        return

    labels = {}
    for p in sorted(rep["period"].unique()):
        row = rep[rep["period"] == p].iloc[0]
        labels[p] = f"P{p}: {str(row['period_start'])[:4]}–{str(row['period_end'])[:4]}"

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 1.15], hspace=0.4, wspace=0.32)

    for idx, r in enumerate([1, -1, 2, -2]):
        ax = fig.add_subplot(gs[0, idx])
        sub = rep[rep["regime"] == r]
        for p in sorted(sub["period"].unique()):
            d = sub[sub["period"] == p].dropna(subset=["p_up"])
            ax.plot(d["z_mid"], d["p_up"], "-o", ms=4, lw=1.6,
                    color=PERIOD_COLORS[(int(p) - 1) % 3],
                    label=labels[p] if idx == 0 else None)
        ax.axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"regime {r}", fontsize=11)
        ax.set_xlabel("z (position in band)", fontsize=8)
        if idx == 0:
            ax.set_ylabel("P(next change up)", fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.legend(loc="center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.52), frameon=False)

    axc = fig.add_subplot(gs[1, 1:3])
    axc.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    axc.plot(cal["mean_pred"], cal["actual_up"], "-", color="#7a3aa7", lw=1.5, zorder=2)
    axc.scatter(cal["mean_pred"], cal["actual_up"], s=cal["n"] / 2.0,
                color="#7a3aa7", alpha=0.75, zorder=3, label="model (size = n)")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_xlabel("predicted P(up)")
    axc.set_ylabel("actual P(up)")
    axc.set_title("Calibration (out-of-sample)", fontsize=11)
    axc.grid(True, alpha=0.25)
    axc.legend(fontsize=8, loc="upper left")

    fig.suptitle("MAD regime transition — P(up-exit) vs z: replication across eras + calibration",
                 fontsize=13, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    os.makedirs("results", exist_ok=True)
    samples = build_samples(INPUT_FILE)

    rep = replication_table(samples)
    rep.to_csv(REP_OUTPUT, index=False)
    cal = calibration_table(samples)
    cal.to_csv(CAL_OUTPUT, index=False)

    print(f"Samples: {len(samples):,} in-regime days "
          f"({samples['date'].iloc[0]} -> {samples['date'].iloc[-1]})\n")

    print("REPLICATION  P(up-exit) by z-bin across periods (monotone rise = holds):")
    for r in (1, -1, 2, -2):
        sub = rep[rep["regime"] == r]
        print(f"\n regime {r}  (z {BANDS[r][0]}..{BANDS[r][1]})")
        piv = sub.pivot(index="period", columns="z_mid", values="p_up")
        print(piv.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nCALIBRATION  (fit on first 60%, scored on last 40%; want mean_pred ~= actual_up):")
    print(cal.to_string(index=False))

    ok = all(
        rep[(rep["regime"] == r)].groupby("period")["p_up"].apply(
            lambda s: pd.Series(s.dropna().values).is_monotonic_increasing).all()
        for r in BANDS
    )
    err = float((cal["mean_pred"] - cal["actual_up"]).abs().mean())
    print(f"\nAll regimes monotone in every period: {ok}")
    print(f"Mean |predicted - actual| across calibration buckets: {err:.3f}")

    plot_results(rep, cal)
    print(f"\nWrote {REP_OUTPUT}, {CAL_OUTPUT}, {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()

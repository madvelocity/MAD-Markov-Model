#!/usr/bin/env python3
"""
02_mad_regime.py  --  MAD-Markov Chain pipeline, step 2: indicator + regimes.

Reads spy.csv, computes:
    sma_20  = 20-day simple moving average of Close
    mad     = (Close - sma_20) / Close * 100        (signed % deviation from trend)
    sigma   = 255-day trailing std of mad            (rolling scale, no look-ahead)
    z       = mad / sigma                             (standardized displacement)
    regime  = 6-state discretization of z:
                -3 : z <= -2      (extended below trend)
                -2 : -2 < z <= -1
                -1 : -1 < z <=  0
                 1 :  0 < z <=  1
                 2 :  1 < z <=  2
                 3 : z >  2       (extended above trend)

All windows are trailing, so each row uses only current and prior data.
Writes spy_mad.csv (Date, Open, Close, sma_20, mad, sigma, z, regime, regime_label).

Usage:
    pip install pandas numpy
    python3 02_mad_regime.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


INPUT_FILE = "results/01_spy.csv"
OUTPUT_FILE = "results/02_spy_mad.csv"
SUMMARY_FILE = "results/02_regime_summary.csv"
SMA_WINDOW = 20
SIGMA_WINDOW = 255

REGIME_LABELS = {
    -3: "extreme_negative", -2: "negative", -1: "mild_negative",
     1: "mild_positive",     2: "positive",  3: "extreme_positive",
}


def read_spy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # drop any stray repeated header rows, coerce types
    df = df[df["Date"].astype(str).str.lower() != "date"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ("Open", "Close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Date", "Open", "Close"]]


def add_mad_regime(df: pd.DataFrame,
                   sma_window: int = SMA_WINDOW,
                   sigma_window: int = SIGMA_WINDOW) -> pd.DataFrame:
    close = df["Close"]
    df["sma_20"] = close.rolling(sma_window, min_periods=sma_window).mean()
    df["mad"] = (close - df["sma_20"]) / close * 100.0
    df["sigma"] = df["mad"].rolling(sigma_window, min_periods=sigma_window).std()
    df["z"] = df["mad"] / df["sigma"]

    z = df["z"]
    reg = pd.Series(pd.NA, index=df.index, dtype="Int64")
    reg[z <= -2] = -3
    reg[(z > -2) & (z <= -1)] = -2
    reg[(z > -1) & (z <= 0)] = -1
    reg[(z > 0) & (z <= 1)] = 1
    reg[(z > 1) & (z <= 2)] = 2
    reg[z > 2] = 3
    df["regime"] = reg
    df["regime_label"] = reg.map(REGIME_LABELS).astype("string")
    return df


def main() -> None:
    os.makedirs("results", exist_ok=True)
    df = read_spy(INPUT_FILE)
    df = add_mad_regime(df)

    out = df[["Date", "Open", "Close", "sma_20", "mad", "sigma", "z",
              "regime", "regime_label"]].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.round({"sma_20": 4, "mad": 4, "sigma": 4, "z": 4})
    out.to_csv(OUTPUT_FILE, index=False)

    valid = df["regime"].notna()
    print(f"Wrote {OUTPUT_FILE}: {len(out):,} rows, "
          f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
    print(f"Warm-up rows with no regime (need {SMA_WINDOW}+{SIGMA_WINDOW} history): "
          f"{int((~valid).sum()):,}")
    print(f"Classified rows: {int(valid.sum()):,} "
          f"({df['Date'][valid].iloc[0].date()} -> {df['Date'][valid].iloc[-1].date()})")
    print("\nRegime distribution:")
    counts = df.loc[valid, "regime"].value_counts().sort_index()
    summary_rows = []
    total = int(valid.sum())
    for r in (-3, -2, -1, 1, 2, 3):
        n = int(counts.get(r, 0))
        pct = 100 * n / total if total else 0.0
        print(f"  {r:>2} {REGIME_LABELS[r]:<17} {n:>6}  ({pct:5.1f}%)")
        summary_rows.append({"regime": r, "label": REGIME_LABELS[r],
                             "count": n, "pct": round(pct, 2)})

    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)
    print(f"\nWrote {SUMMARY_FILE} (regime distribution).")
    print(f"\nMost recent: {df['Date'].iloc[-1].date()}  "
          f"MAD {df['mad'].iloc[-1]:.2f}  z {df['z'].iloc[-1]:.2f}  "
          f"regime {int(df['regime'].iloc[-1])} ({REGIME_LABELS[int(df['regime'].iloc[-1])]})")


if __name__ == "__main__":
    main()

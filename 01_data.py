#!/usr/bin/env python3
"""
01_data.py  --  MAD-Markov Chain pipeline, step 1: data.

Download ~26 years of daily SPY from Yahoo Finance and save to spy.csv.

Raw OHLC + Adj Close are kept separate (auto_adjust=False) so downstream steps
can pick the price field they need. Actions (dividends/splits) are included for
optional total-return work later. This is the clean foundation the regime and
Markov-chain steps build on.

Usage:
    pip install yfinance pandas
    python3 01_data.py
"""

import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf


TICKER = "SPY"
YEARS_BACK = 26
OUTPUT_FILE = "results/01_spy.csv"


def years_ago(today: date, years: int) -> date:
    """Date `years` years before `today`, handling the Feb-29 edge case."""
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


def main() -> None:
    os.makedirs("results", exist_ok=True)
    end_date = date.today() + timedelta(days=1)   # yfinance end date is exclusive
    start_date = years_ago(date.today(), YEARS_BACK)

    df = yf.download(
        TICKER,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=False,   # keep Open/High/Low/Close and Adj Close separate
        actions=True,        # include dividends and splits when available
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {TICKER}")

    # flatten columns if yfinance returns a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.to_csv(OUTPUT_FILE)

    print(f"Downloaded {len(df):,} rows for {TICKER}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Saved to:   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

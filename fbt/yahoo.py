"""Free historical data from Yahoo Finance (via `yfinance`) for research backtests.

Yahoo serves roughly the last 30 days of 1-minute bars (at most 7 days per
request) and years of daily bars, for continuous front-month futures such as
ES=F. Continuous series are NOT back-adjusted: expect a jump at each roll.

Usage:
  python -m fbt.yahoo --symbols ES NQ YM GC --out data/yahoo
writes data/yahoo/<SYM>_1m.csv.gz and data/yahoo/<SYM>_daily.csv.gz, which the
backtester reads with:
  python -m fbt.backtest --data data/yahoo/ES_1m.csv.gz --daily-data data/yahoo/ES_daily.csv.gz --symbol ES
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from .costs import get_instrument
from .data import DataError, validate_ohlcv
from .feeds import current_trading_date

YAHOO_TICKERS = {"ES": "ES=F", "MES": "ES=F", "NQ": "NQ=F", "MNQ": "NQ=F",
                 "YM": "YM=F", "MYM": "YM=F", "GC": "GC=F", "MGC": "GC=F",
                 "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "GLD": "GLD"}


def _yf():
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise DataError("Yahoo downloads need `pip install yfinance`.") from None
    return yf


def _normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise DataError(f"{source}: Yahoo returned no rows")
    cols = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=cols)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise DataError(f"{source}: Yahoo response missing {missing}")
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    # Yahoo stores prices as float32 (3969.9 arrives as 3969.900048828125). float32 spacing
    # is <= 0.004 below 60,000 and every supported tick is >= 0.01, so rounding to 2
    # decimals restores the exact exchange price; it cannot move a price by a tick.
    if (out[["open", "high", "low", "close"]].abs() >= 60_000).any().any():
        raise DataError("prices too large for exact float32 recovery; refusing to round")
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].round(2)
    return out


def download_1m(symbol: str, days: int = 29, now: pd.Timestamp = None, pause: float = 1.0) -> pd.DataFrame:
    """Last `days` days of 1-minute bars (UTC open times), completed bars only, validated."""
    yf = _yf()
    ticker = YAHOO_TICKERS[get_instrument(symbol).symbol]
    now = now or pd.Timestamp.now(tz="UTC")
    end = now.normalize() + pd.Timedelta(days=1)
    first = now - pd.Timedelta(days=days)
    frames = []
    while end > first:
        start = max(first.normalize(), end - pd.Timedelta(days=7))
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1m",
                          progress=False, auto_adjust=False, prepost=True, multi_level_index=False)
        if raw is not None and len(raw):
            frames.append(raw)
        end = start
        time.sleep(pause)
    if not frames:
        raise DataError(f"Yahoo returned no 1-minute data for {ticker}")
    df = _normalise(pd.concat(frames), ticker)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index + pd.Timedelta(minutes=1) <= now]  # drop the minute still forming
    df.index.name = "timestamp"
    validate_ohlcv(df, source=f"Yahoo {ticker}")
    return df


def download_daily(symbol: str, years: int = 1, now: pd.Timestamp = None) -> pd.DataFrame:
    """Daily bars indexed by trading date, excluding today's unfinished session.

    One year by default: the strategy reads at most 60 daily bars, and older Yahoo
    history contains some malformed rows (e.g. YM in early 2025)."""
    yf = _yf()
    ticker = YAHOO_TICKERS[get_instrument(symbol).symbol]
    now = now or pd.Timestamp.now(tz="UTC")
    raw = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=False,
                      multi_level_index=False)
    df = _normalise(raw, ticker)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[df.index.date < current_trading_date(now)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download free Yahoo Finance data for research backtests")
    ap.add_argument("--symbols", nargs="+", default=["ES", "NQ", "YM", "GC"])
    ap.add_argument("--days", type=int, default=29)
    ap.add_argument("--out", default="data/yahoo")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    code = 0
    for sym in args.symbols:
        try:
            m1 = download_1m(sym, args.days)
            daily = download_daily(sym)
        except DataError as exc:
            print(f"DATA ERROR {sym}: {exc}", file=sys.stderr)
            code = 2
            continue
        root = get_instrument(sym).symbol
        m1.to_csv(os.path.join(args.out, f"{root}_1m.csv.gz"))
        daily.to_csv(os.path.join(args.out, f"{root}_daily.csv.gz"))
        print(f"{root}: {len(m1)} one-minute bars {m1.index[0]} -> {m1.index[-1]}, {len(daily)} daily bars")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

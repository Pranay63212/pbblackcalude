"""Historical 1-minute OHLCV loading, validation and multi-timeframe bars.

Lookahead protection lives here:

* Every bar on every timeframe carries `close_time` (UTC ns). A bar exists
  for the strategy only when `close_time <= now`.
* `MarketView` is the ONLY object the strategy receives. It hands out
  read-only numpy views truncated to completed bars, so future data cannot
  be read even by accident.

Higher-timeframe bars are built from 1-minute bars aligned to the CME Globex
session that opens at 17:00 America/Chicago. Daily bars therefore span
17:00 -> 17:00 CT and only become available after that session ends.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

CHICAGO = "America/Chicago"
NS_PER_MIN = 60 * 1_000_000_000

TIMEFRAME_MINUTES = {
    "1m": 1, "2m": 2, "3m": 3, "4m": 4, "5m": 5,
    "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}
ALL_TIMEFRAMES = list(TIMEFRAME_MINUTES)
SESSION_OPEN_HOUR = 17  # CME Globex opens 17:00 CT


class DataError(ValueError):
    """Raised for missing, malformed or insufficient historical data."""


def timeframe_minutes(tf: str) -> int:
    try:
        return TIMEFRAME_MINUTES[tf]
    except KeyError:
        raise ValueError(f"Unknown timeframe '{tf}'. Valid: {ALL_TIMEFRAMES}") from None


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
_TS_CANDIDATES = ("timestamp", "ts_event", "datetime", "date_time", "time", "date")
REQUIRED = ("open", "high", "low", "close", "volume")


def load_ohlcv(path: str, symbol: Optional[str] = None, tz: str = "UTC",
               start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load 1-minute OHLCV bars. Timestamps are bar OPEN times.

    Supports .csv, .csv.gz and (if the `databento` package is installed) .dbn.zst.
    Naive timestamps are interpreted in `tz`. Returns a DataFrame indexed by a
    UTC DatetimeIndex with float columns open/high/low/close/volume.
    """
    if not os.path.exists(path):
        raise DataError(f"Data file not found: {path}")
    if os.path.getsize(path) == 0:
        raise DataError(f"Data file is empty: {path}")

    lower = path.lower()
    if lower.endswith(".dbn.zst") or lower.endswith(".dbn"):
        df = _load_dbn(path)
    elif lower.endswith(".csv") or lower.endswith(".csv.gz"):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # parsing errors -> clear message
            raise DataError(f"Could not parse CSV {path}: {exc}") from exc
    else:
        raise DataError(f"Unsupported data format for {path}. Use .csv, .csv.gz or .dbn.zst")

    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in _TS_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise DataError(f"{path}: no timestamp column found (expected one of {_TS_CANDIDATES})")

    if "symbol" in df.columns:
        symbols = sorted(map(str, df["symbol"].dropna().unique()))
        if symbol is not None:
            df = df[df["symbol"].astype(str) == symbol]
            if df.empty:
                raise DataError(f"{path}: symbol '{symbol}' not present. Found: {symbols[:20]}")
        elif len(symbols) > 1:
            raise DataError(
                f"{path}: contains {len(symbols)} symbols {symbols[:10]}. Pass symbol= to select "
                "one continuous series; mixing contracts would fabricate prices.")

    try:
        raw_ts = pd.to_datetime(df[ts_col], utc=False, format="mixed")
    except Exception as exc:
        raise DataError(f"{path}: could not parse timestamps in column '{ts_col}': {exc}") from exc
    idx = pd.DatetimeIndex(raw_ts)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    idx = idx.tz_convert("UTC").as_unit("ns")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing required columns {missing}")
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in REQUIRED},
                       index=idx)
    out.index.name = "timestamp"
    out = out.sort_index()
    if start is not None or end is not None:
        out = _slice(out, start, end)
    validate_ohlcv(out, source=path)
    return out


def _load_dbn(path: str) -> pd.DataFrame:
    try:
        import databento as db  # type: ignore
    except ImportError:
        raise DataError(f"{path}: reading DBN files requires `pip install databento`, "
                        "or convert the file to CSV first.") from None
    store = db.DBNStore.from_file(path)
    df = store.to_df(price_type="float", pretty_ts=True).reset_index()
    return df


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df.empty:
        raise DataError("No rows loaded.")
    first, last = df.index[0], df.index[-1]
    if start is not None:
        s = _to_utc(start)
        if s < first - pd.Timedelta(days=4):
            raise DataError(f"Requested start {start} is before the first bar {first}. Data does not cover it.")
        df = df[df.index >= s]
    if end is not None:
        e = _to_utc(end)
        if e > last + pd.Timedelta(days=4):
            raise DataError(f"Requested end {end} is after the last bar {last}. Data does not cover it.")
        df = df[df.index < e]
    if df.empty:
        raise DataError(f"No bars between start={start} and end={end}.")
    return df


def _to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(CHICAGO)
    return ts.tz_convert("UTC")


def validate_ohlcv(df: pd.DataFrame, source: str = "data") -> None:
    """Raise DataError if the frame cannot be trusted. Never repairs data."""
    if df is None or len(df) == 0:
        raise DataError(f"{source}: no bars.")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise DataError(f"{source}: index must be a timezone-aware DatetimeIndex.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"{source}: missing required columns {missing}")
    vals = df[list(REQUIRED)].to_numpy(dtype=float)
    bad = ~np.isfinite(vals)
    if bad.any():
        row = int(np.argmax(bad.any(axis=1)))
        raise DataError(f"{source}: {int(bad.any(axis=1).sum())} rows have missing/non-numeric values "
                        f"(first at {df.index[row]}).")
    o, h, l, c, v = vals.T
    checks = [
        ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0), "non-positive prices"),
        (h < l, "high < low"),
        (h < np.maximum(o, c), "high below open/close"),
        (l > np.minimum(o, c), "low above open/close"),
        (v < 0, "negative volume"),
    ]
    for mask, what in checks:
        if mask.any():
            row = int(np.argmax(mask))
            raise DataError(f"{source}: {int(mask.sum())} rows with {what} (first at {df.index[row]}).")
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()][0]
        raise DataError(f"{source}: duplicate timestamps (first {dup}). Data may contain multiple contracts.")
    if not df.index.is_monotonic_increasing:
        raise DataError(f"{source}: timestamps are not sorted.")
    ns = df.index.as_unit("ns").asi8
    if (ns % NS_PER_MIN != 0).any():
        raise DataError(f"{source}: timestamps are not aligned to whole minutes; expected 1-minute bars.")
    if len(ns) > 1:
        step = np.diff(ns)
        if (step < NS_PER_MIN).any():
            raise DataError(f"{source}: bars closer than one minute apart; expected 1-minute bars.")


def data_quality_report(df: pd.DataFrame) -> dict:
    """Informational stats about gaps inside sessions (not errors: halts/holidays exist)."""
    ns = df.index.as_unit("ns").asi8
    gaps = np.diff(ns) // NS_PER_MIN
    return {
        "bars": int(len(df)),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "gaps_2_to_60_min": int(((gaps > 1) & (gaps <= 60)).sum()),
        "gaps_over_60_min": int((gaps > 60).sum()),
    }


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
@dataclass
class Bars:
    """Bars of one timeframe as read-only numpy arrays (UTC ns times)."""
    tf: str
    open_time: np.ndarray
    close_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self):
        for name in ("open_time", "close_time", "open", "high", "low", "close", "volume"):
            arr = np.ascontiguousarray(getattr(self, name))
            arr.flags.writeable = False
            setattr(self, name, arr)

    def __len__(self) -> int:
        return len(self.close)

    @property
    def minutes(self) -> int:
        return timeframe_minutes(self.tf)

    def available_count(self, now_ns: int) -> int:
        """Number of bars completed at or before now_ns."""
        return int(np.searchsorted(self.close_time, now_ns, side="right"))

    def head(self, n: int) -> "Bars":
        """First n bars as views (no copy)."""
        return Bars(self.tf, self.open_time[:n], self.close_time[:n], self.open[:n], self.high[:n],
                    self.low[:n], self.close[:n], self.volume[:n])


def _session_open_ns(trading_dates: np.ndarray) -> np.ndarray:
    """UTC ns of 17:00 CT on the calendar day before each trading date."""
    uniq, inv = np.unique(trading_dates, return_inverse=True)
    opens = (pd.DatetimeIndex(pd.to_datetime(uniq)) - pd.Timedelta(days=1) + pd.Timedelta(hours=SESSION_OPEN_HOUR))
    opens = opens.tz_localize(CHICAGO, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    return opens.as_unit("ns").asi8[inv]


def _trading_dates(ts_ns: np.ndarray) -> np.ndarray:
    local = pd.DatetimeIndex(ts_ns).tz_localize("UTC").tz_convert(CHICAGO)
    shifted = local.tz_localize(None) + pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)
    return shifted.normalize().to_numpy(dtype="datetime64[D]")


def bars_from_frame(df: pd.DataFrame) -> Bars:
    ns = df.index.as_unit("ns").asi8
    return Bars("1m", ns, ns + NS_PER_MIN, df["open"].to_numpy(float), df["high"].to_numpy(float),
                df["low"].to_numpy(float), df["close"].to_numpy(float), df["volume"].to_numpy(float))


def resample(m1: Bars, tf: str, trading_dates: Optional[np.ndarray] = None,
             session_open: Optional[np.ndarray] = None) -> Bars:
    """Aggregate 1-minute bars into `tf` bars aligned to the 17:00 CT session."""
    if tf == "1m":
        return m1
    minutes = timeframe_minutes(tf)
    if len(m1) == 0:
        raise DataError("Cannot resample empty data.")
    if trading_dates is None:
        trading_dates = _trading_dates(m1.open_time)
    if session_open is None:
        session_open = _session_open_ns(trading_dates)
    offset_min = (m1.open_time - session_open) // NS_PER_MIN
    if minutes >= 1440:
        bucket = np.zeros_like(offset_min)
    else:
        bucket = offset_min // minutes
    day_id = trading_dates.astype("int64")
    key_change = np.ones(len(m1), dtype=bool)
    key_change[1:] = (day_id[1:] != day_id[:-1]) | (bucket[1:] != bucket[:-1])
    starts = np.flatnonzero(key_change)

    o = m1.open[starts]
    h = np.maximum.reduceat(m1.high, starts)
    l = np.minimum.reduceat(m1.low, starts)
    ends = np.append(starts[1:], len(m1)) - 1
    c = m1.close[ends]
    v = np.add.reduceat(m1.volume, starts)

    s_open = session_open[starts]
    next_open = _session_open_ns(trading_dates[starts] + np.timedelta64(1, "D"))
    if minutes >= 1440:
        open_time = s_open
        close_time = next_open
    else:
        open_time = s_open + bucket[starts] * minutes * NS_PER_MIN
        close_time = np.minimum(open_time + minutes * NS_PER_MIN, next_open)
    return Bars(tf, open_time, close_time, o, h, l, c, v)


def daily_bars_from_frame(daily: pd.DataFrame) -> Bars:
    """Daily OHLCV indexed by trading date -> Bars spanning 17:00 CT (prior day) to 17:00 CT."""
    if daily is None or len(daily) == 0:
        raise DataError("daily history is empty")
    missing = [c for c in REQUIRED if c not in daily.columns]
    if missing:
        raise DataError(f"daily history missing columns {missing}")
    idx = pd.DatetimeIndex(daily.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    td = idx.normalize().to_numpy(dtype="datetime64[D]")
    if len(td) > 1 and not (np.diff(td.astype("int64")) > 0).all():
        raise DataError("daily history dates must be unique and increasing")
    vals = daily[list(REQUIRED)].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise DataError("daily history has missing values")
    o, h, l, c, v = vals.T
    if (h < l).any() or (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any() or (vals[:, :4] <= 0).any():
        raise DataError("daily history has inconsistent OHLC values")
    open_t = _session_open_ns(td)
    close_t = _session_open_ns(td + np.timedelta64(1, "D"))
    return Bars("1d", open_t, close_t, o, h, l, c, v)


def _merge_daily(daily: pd.DataFrame, derived: Bars, first_trading_date, symbol: str) -> Bars:
    """External daily bars for dates before the first FULL 1m session, derived bars after."""
    ext = daily_bars_from_frame(daily)
    cutoff = _session_open_ns(np.array([first_trading_date + np.timedelta64(1, "D")]))[0]
    keep_ext = ext.open_time < cutoff
    keep_der = derived.open_time >= cutoff
    cat = lambda a, b: np.concatenate([a[keep_ext], b[keep_der]])  # noqa: E731
    merged = Bars("1d", cat(ext.open_time, derived.open_time), cat(ext.close_time, derived.close_time),
                  cat(ext.open, derived.open), cat(ext.high, derived.high), cat(ext.low, derived.low),
                  cat(ext.close, derived.close), cat(ext.volume, derived.volume))
    if len(merged) > 1 and not (np.diff(merged.open_time) > 0).all():
        raise DataError(f"{symbol}: daily history overlaps the 1-minute data inconsistently")
    return merged


class MultiTimeframeData:
    """1-minute data plus every derived timeframe, built once up front."""

    def __init__(self, df: pd.DataFrame, symbol: str, timeframes: Iterable[str] = ALL_TIMEFRAMES,
                 daily_history: Optional[pd.DataFrame] = None):
        """`daily_history` (optional): daily OHLCV indexed by trading date, used only for
        trading dates before the 1-minute data fully covers a session (live warm-up)."""
        validate_ohlcv(df, source=symbol)
        self.symbol = symbol
        self.m1 = bars_from_frame(df)
        dates = _trading_dates(self.m1.open_time)
        opens = _session_open_ns(dates)
        self.bars: Dict[str, Bars] = {}
        for tf in timeframes:
            self.bars[tf] = resample(self.m1, tf, dates, opens)
        self.bars["1m"] = self.m1
        if daily_history is not None and "1d" in self.bars:
            self.bars["1d"] = _merge_daily(daily_history, self.bars["1d"], dates[0], symbol)
        local = pd.DatetimeIndex(self.m1.close_time).tz_localize("UTC").tz_convert(CHICAGO)
        # minute-of-day (local) of each 1m bar's CLOSE, and its local calendar date
        self.close_minute_of_day = (local.hour * 60 + local.minute).to_numpy(dtype=np.int32)
        self.close_local_date = local.tz_localize(None).normalize().to_numpy(dtype="datetime64[D]")

    def view(self, now_ns: int) -> "MarketView":
        return MarketView(self, now_ns)


class MarketView:
    """Read-only window onto completed bars as of `now_ns` (inclusive)."""

    def __init__(self, data: MultiTimeframeData, now_ns: int):
        self._data = data
        self.now = int(now_ns)
        self._counts: Dict[str, int] = {}

    @property
    def symbol(self) -> str:
        return self._data.symbol

    def count(self, tf: str) -> int:
        n = self._counts.get(tf)
        if n is None:
            n = self._data.bars[tf].available_count(self.now)
            self._counts[tf] = n
        return n

    def bars(self, tf: str) -> Bars:
        if tf not in self._data.bars:
            raise DataError(f"Timeframe {tf} not loaded.")
        return self._data.bars[tf].head(self.count(tf))

    @property
    def last_close(self) -> float:
        m1 = self.bars("1m")
        if len(m1) == 0:
            raise DataError("No completed 1m bar at this time.")
        return float(m1.close[-1])


def ns_to_chicago(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), tz="UTC").tz_convert(CHICAGO)


def minutes_to_ns(minutes: float) -> int:
    return int(minutes * NS_PER_MIN)


def parse_hhmm(value: str) -> int:
    hh, mm = value.split(":")
    return int(hh) * 60 + int(mm)


__all__ = [
    "DataError", "Bars", "MultiTimeframeData", "MarketView", "load_ohlcv", "validate_ohlcv",
    "resample", "timeframe_minutes", "TIMEFRAME_MINUTES", "data_quality_report", "ns_to_chicago",
    "parse_hhmm", "minutes_to_ns", "NS_PER_MIN", "CHICAGO",
]

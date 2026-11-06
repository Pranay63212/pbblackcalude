"""Market-data feeds for the live scanner. Read-only: nothing here can place orders.

Every feed returns 1-minute OHLCV frames indexed by UTC bar OPEN time, with
the same columns as data.load_ohlcv. The scanner keeps only bars that have
fully closed (`completed_only`), so a still-forming bar can never reach the
strategy.

Feeds:
* CSVFolderFeed: reads <folder>/<SYMBOL>.csv(.gz), e.g. files kept up to date
  by your charting platform's exporter. Fully tested.
* ReplayFeed: serves a historical frame as if it were arriving live (tests,
  rehearsals). Fully tested.
* AlpacaFeed: Alpaca's free IEX stock/ETF data (SPY, QQQ, DIA, GLD), keys
  from environment variables. Tested with a fake HTTP client only.
* IBKRFeed: Interactive Brokers TWS / IB Gateway via `ib_async`, opened
  read-only. Its conversions are unit-tested with a fake client, but it has
  NOT been run against a real TWS session from this project's test suite.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import pandas as pd

from .costs import EXCHANGES, get_instrument
from .data import CHICAGO, DataError, load_ohlcv, validate_ohlcv


def completed_only(df: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Keep bars whose close (open + 1 minute) is at or before `now_utc`."""
    if df is None or len(df) == 0:
        return df
    return df[df.index + pd.Timedelta(minutes=1) <= now_utc]


def merge_bars(store: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """Append new bars; a re-sent minute replaces the earlier copy."""
    if store is None or len(store) == 0:
        out = new.copy()
    elif new is None or len(new) == 0:
        return store
    else:
        out = pd.concat([store, new])
        out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


class ReplayFeed:
    """Serves historical bars as if live. `now` is set by the caller (or a fake clock)."""

    def __init__(self, frames: Dict[str, pd.DataFrame], include_forming_bar: bool = True):
        self.frames = frames
        self.include_forming = include_forming_bar
        self.now: Optional[pd.Timestamp] = None

    def history(self, symbol: str, days: int) -> pd.DataFrame:
        df = self._frame(symbol)
        return df[df.index < self.now]

    def daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def recent(self, symbol: str) -> pd.DataFrame:
        df = self._frame(symbol)
        lo = self.now - pd.Timedelta(minutes=30)
        # a live feed typically also sends the bar that is still forming
        hi = self.now + (pd.Timedelta(minutes=1) if self.include_forming else pd.Timedelta(0))
        return df[(df.index >= lo) & (df.index < hi)]

    def _frame(self, symbol):
        if symbol not in self.frames:
            raise DataError(f"replay feed has no data for {symbol}")
        if self.now is None:
            raise RuntimeError("ReplayFeed.now is not set")
        return self.frames[symbol]


class CSVFolderFeed:
    """Reads <folder>/<SYMBOL>.csv or .csv.gz on every poll (validated each time)."""

    def __init__(self, folder: str, tz: str = "UTC"):
        if not os.path.isdir(folder):
            raise DataError(f"live data folder not found: {folder}")
        self.folder = folder
        self.tz = tz

    def _path(self, symbol: str) -> str:
        for ext in (".csv", ".csv.gz"):
            p = os.path.join(self.folder, symbol + ext)
            if os.path.exists(p):
                return p
        raise DataError(f"no file for {symbol} in {self.folder} (expected {symbol}.csv or {symbol}.csv.gz)")

    def history(self, symbol: str, days: int) -> pd.DataFrame:
        df = load_ohlcv(self._path(symbol), tz=self.tz)
        cut = df.index[-1] - pd.Timedelta(days=days)
        return df[df.index >= cut]

    def daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def recent(self, symbol: str) -> pd.DataFrame:
        df = load_ohlcv(self._path(symbol), tz=self.tz)
        return df.iloc[-60:]


# ---------------------------------------------------------------------------
# Interactive Brokers (read-only)
# ---------------------------------------------------------------------------
def ib_bars_to_frame(bars) -> pd.DataFrame:
    """Convert ib_async BarData (1-minute, formatDate=2 -> UTC) to our frame."""
    rows = []
    for b in bars:
        ts = pd.Timestamp(b.date)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        rows.append((ts, float(b.open), float(b.high), float(b.low), float(b.close), max(float(b.volume), 0.0)))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
    if len(df):
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def ib_daily_to_frame(bars, today_trading_date) -> pd.DataFrame:
    """Daily BarData -> frame indexed by trading date, excluding today's unfinished day."""
    rows = []
    for b in bars:
        d = pd.Timestamp(b.date).tz_localize(None) if pd.Timestamp(b.date).tzinfo else pd.Timestamp(b.date)
        d = d.normalize()
        if d.date() >= today_trading_date:
            continue
        rows.append((d, float(b.open), float(b.high), float(b.low), float(b.close), max(float(b.volume), 0.0)))
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).set_index("date")


def current_trading_date(now_utc: pd.Timestamp):
    local = now_utc.tz_convert(CHICAGO).tz_localize(None)
    return (local + pd.Timedelta(hours=7)).normalize().date()


class IBKRFeed:
    """Interactive Brokers market data through TWS or IB Gateway (paper port 7497 by default).

    Connects with readonly=True: this object has no order methods and IB will
    refuse orders on a read-only connection. Requires `pip install ib_async`
    and real-time CME/CBOT/COMEX market-data permissions on the IB account
    (without them IB returns delayed or no data, which the scanner flags).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17, ib=None,
                 pacing_seconds: float = 2.0):
        if ib is None:
            try:
                from ib_async import IB  # type: ignore
            except ImportError:
                raise DataError("IBKRFeed needs `pip install ib_async`.") from None
            ib = IB()
            ib.connect(host, port, clientId=client_id, readonly=True)
        self.ib = ib
        self.pacing = pacing_seconds
        self._contracts: Dict[str, object] = {}

    def contract(self, symbol: str):
        """Front-month future for the root (resolved from IB's continuous-future definition)."""
        if symbol not in self._contracts:
            from ib_async import ContFuture, Future  # type: ignore
            root = get_instrument(symbol).symbol
            cont = ContFuture(root, exchange=EXCHANGES[root], currency="USD")
            qualified = self.ib.qualifyContracts(cont)
            if not qualified:
                raise DataError(f"IB could not resolve a futures contract for {root}")
            self._contracts[symbol] = Future(conId=qualified[0].conId, exchange=EXCHANGES[root])
            self.ib.qualifyContracts(self._contracts[symbol])
        return self._contracts[symbol]

    def _request(self, contract, end, duration: str, bar_size: str):
        bars = self.ib.reqHistoricalData(contract, endDateTime=end, durationStr=duration, barSizeSetting=bar_size,
                                         whatToShow="TRADES", useRTH=False, formatDate=2)
        if self.pacing:
            time.sleep(self.pacing)  # stay well inside IB's historical-data pacing limits
        return bars

    def history(self, symbol: str, days: int) -> pd.DataFrame:
        """`days` calendar days of 1-minute bars, fetched one day per request."""
        c = self.contract(symbol)
        frames = []
        end = pd.Timestamp.now(tz="UTC")
        for _ in range(days):
            bars = self._request(c, end.to_pydatetime(), "1 D", "1 min")
            df = ib_bars_to_frame(bars)
            if len(df):
                frames.append(df)
                end = df.index[0]
            else:
                end = end - pd.Timedelta(days=1)
        if not frames:
            raise DataError(f"IB returned no 1-minute history for {symbol} (check market-data permissions)")
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        validate_ohlcv(out, source=f"IB {symbol}")
        return out

    def daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        bars = self._request(self.contract(symbol), "", "1 Y", "1 day")
        df = ib_daily_to_frame(bars, current_trading_date(pd.Timestamp.now(tz="UTC")))
        return df if len(df) else None

    def recent(self, symbol: str) -> pd.DataFrame:
        bars = self._request(self.contract(symbol), "", "1800 S", "1 min")
        return ib_bars_to_frame(bars)


# ---------------------------------------------------------------------------
# Alpaca market data (read-only, free IEX feed)
# ---------------------------------------------------------------------------
class AlpacaFeed:
    """Alpaca stock/ETF bars from data.alpaca.markets. Keys come ONLY from the environment
    variables APCA_API_KEY_ID and APCA_API_SECRET_KEY; they are never logged or stored.

    The free plan's real-time feed is IEX (a single exchange): prices track the full
    market closely, but some minutes have no IEX trade, so expect occasional gaps.
    This class only reads market data; it has no order methods.
    """

    BASE = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, feed: str = "iex", http=None, env=None):
        env = os.environ if env is None else env
        self._key = env.get("APCA_API_KEY_ID")
        self._secret = env.get("APCA_API_SECRET_KEY")
        if not self._key or not self._secret:
            raise DataError("Set the APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables "
                            "(Alpaca dashboard -> API Keys). Never put them in a file in this project.")
        self.feed = feed
        self._http = http or self._urllib_get

    def __repr__(self) -> str:  # never show keys
        return f"AlpacaFeed(feed={self.feed!r}, keys=<hidden>)"

    def _urllib_get(self, url: str, headers: dict) -> dict:
        import json
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise DataError("Alpaca rejected the API keys (check APCA_API_KEY_ID / APCA_API_SECRET_KEY, "
                                "and that they are the paper-account keys).") from None
            raise DataError(f"Alpaca data request failed: HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise DataError(f"Could not reach Alpaca: {exc.reason}") from None

    def _bars(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> list:
        from urllib.parse import urlencode
        headers = {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}
        params = {"timeframe": timeframe, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "feed": self.feed,
                  "adjustment": "raw", "sort": "asc"}
        out, token = [], None
        for _ in range(200):
            q = dict(params, **({"page_token": token} if token else {}))
            data = self._http(f"{self.BASE}/{symbol}/bars?{urlencode(q)}", headers)
            out.extend(data.get("bars") or [])
            token = data.get("next_page_token")
            if not token:
                break
        return out

    @staticmethod
    def bars_to_frame(bars: list) -> pd.DataFrame:
        rows = [(pd.Timestamp(b["t"]), float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]),
                 max(float(b.get("v", 0)), 0.0)) for b in bars]
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
        if len(df):
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df.index = idx.as_unit("ns")
            df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def history(self, symbol: str, days: int) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        df = self.bars_to_frame(self._bars(symbol, "1Min", now - pd.Timedelta(days=days), now))
        if len(df) == 0:
            raise DataError(f"Alpaca returned no 1-minute history for {symbol}")
        validate_ohlcv(df, source=f"Alpaca {symbol}")
        return df

    def daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        now = pd.Timestamp.now(tz="UTC")
        bars = self._bars(symbol, "1Day", now - pd.Timedelta(days=400), now)
        rows = []
        today = current_trading_date(now)
        for b in bars:
            d = pd.Timestamp(b["t"]).tz_convert("America/New_York").tz_localize(None).normalize()
            if d.date() >= today:
                continue  # today's session is unfinished
            rows.append((d, float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]), max(float(b.get("v", 0)), 0.0)))
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).set_index("date")
        return df if len(df) else None

    def recent(self, symbol: str) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        return self.bars_to_frame(self._bars(symbol, "1Min", now - pd.Timedelta(minutes=30), now))


def minutes_since_last_bar(df: pd.DataFrame, now_utc: pd.Timestamp) -> float:
    if df is None or len(df) == 0:
        return float("inf")
    return (now_utc - (df.index[-1] + pd.Timedelta(minutes=1))) / pd.Timedelta(minutes=1)


__all__ = ["completed_only", "merge_bars", "ReplayFeed", "CSVFolderFeed", "IBKRFeed", "AlpacaFeed", "ib_bars_to_frame",
           "ib_daily_to_frame", "current_trading_date", "minutes_since_last_bar"]

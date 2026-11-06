"""Live strategy scanner: watches markets and ALERTS. This module never places orders.
With --trade, signals are handed to fbt/trader.py, which sends PAPER orders only
after RiskManager approval.

Once per minute, a few seconds after each 1-minute candle closes, it runs the
same PBBlakeStrategy the backtester uses on completed bars only. It reports:
* SIGNAL: every condition in docs/STRATEGY.md is met (entry, stop, target, R:R);
* SETUP FORMING: price traded into a key level and the setup is waiting for
  the highest-timeframe IFVG to invert (a heads-up, not a trade);
* a pre-session summary (bias, draw on liquidity, key levels);
* STALE / DATA ERROR when data stops or is invalid (it never guesses).

Signals are only reported inside the 08:30-10:00 CT window. The scanner does
not know your fills, so it cannot enforce the daily limits (spec 4.5); it
prints them as a reminder.

Usage:
  python -m fbt.scanner live --feed alpaca                      (SPY QQQ DIA GLD, free data)
  python -m fbt.scanner live --feed ibkr --symbols ES NQ YM GC [--port 7497]
  python -m fbt.scanner live --feed csv --folder data/live --symbols ES NQ
  python -m fbt.scanner replay --data data/ES_1m.csv.gz --symbol ES --date 2024-03-12
Optional phone/desktop alerts: set FBT_ALERT_WEBHOOK to a webhook URL
(ntfy.sh topic, Discord webhook, ...). Never put it in code or commit it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional

import pandas as pd

from .costs import ETFS, SMT_PAIRS, CostModel, get_instrument
from .data import CHICAGO, DataError, MultiTimeframeData, load_ohlcv, ns_to_chicago, parse_hhmm
from .feeds import completed_only, merge_bars, minutes_since_last_bar
from .risk import RiskManager
from .strategy import Decision, PBBlakeStrategy

MICRO = {"ES": "MES", "NQ": "MNQ", "YM": "MYM", "GC": "MGC"}
DISCLAIMER = ("Alerts show setups that FIT the written rules. They are not predictions and not advice. "
              "No orders are placed.")


@dataclass
class ScanEvent:
    time: int
    symbol: str
    kind: str            # SIGNAL / SETUP_FORMING / PREMARKET / STALE / DATA_ERROR / STATUS
    text: str
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def _px(x, tick):
    if x is None:
        return "-"
    decimals = max(0, -Decimal(str(tick)).normalize().as_tuple().exponent)
    return f"{x:.{decimals}f}"


def _hm(ns):
    return ns_to_chicago(ns).strftime("%H:%M") if ns else "-"


def format_bias(d: Decision) -> str:
    b = d.detail.get("bias", {})
    short = {"BULLISH": "bull", "BEARISH": "bear", "NEUTRAL": "neutral", None: "-"}
    parts = f"D {short[b.get('daily_bias')]} / 4H {short[b.get('four_hour_bias')]} / 1H {short[b.get('one_hour_bias')]}"
    extra = []
    if b.get("fifteen_minute_confirmation") is not None:
        extra.append("15m confirms" if b["fifteen_minute_confirmation"] else "15m does not confirm")
    if b.get("smt_confirmation") is not None:
        extra.append("SMT yes" if b["smt_confirmation"] else "SMT no")
    return f"{d.bias_direction} ({parts}{', ' + ', '.join(extra) if extra else ''})"


def format_levels(d: Decision, tick) -> List[str]:
    out = []
    for lv in d.detail.get("levels", []):
        touched = f"touched {_hm(lv['touch_time'])}" if lv.get("touch_time") else "untouched"
        out.append(f"{lv['type']} {lv['timeframe']} {_px(lv['lower'], tick)}-{_px(lv['upper'], tick)} ({touched})")
    return out


def sizing_lines(signal, symbol: str, config: dict) -> List[str]:
    lines = []
    for sym in (symbol, MICRO.get(symbol)):
        if not sym:
            continue
        inst = get_instrument(sym)
        rm = RiskManager(config, CostModel.default(sym))
        per = abs(signal.entry_price - signal.stop) * inst.point_value
        unit = inst.unit
        lines.append(f"{sym}: ${per:,.2f} risk per {unit}, {rm.size(signal)} {unit}(s) at "
                     f"{config['risk']['risk_per_trade_pct']}% of ${config['risk']['initial_equity']:,.0f}")
    return lines


def format_signal(symbol: str, d: Decision, config: dict) -> str:
    s = d.signal
    tick = get_instrument(symbol).tick_size
    side = "LONG" if s.direction == "BULLISH" else "SHORT"
    lv = s.key_level
    lines = [
        f"*** {symbol} {side} SIGNAL {_hm(d.time)} CT ***",
        f"Entry ~{_px(s.entry_price, tick)} ({s.entry_method}, after the {s.ifvg['timeframe']} candle close)",
        f"Stop  {_px(s.stop, tick)}   Target {_px(s.target, tick)} ({s.target_label})   R:R {s.risk_reward:.2f}",
        f"Bias  {format_bias(d)}; draw {_px(s.draw_on_liquidity['price'], tick)} "
        f"{s.draw_on_liquidity['liquidity_type']}",
        f"Level {lv['type']} {lv['timeframe']} {_px(lv['lower_boundary'], tick)}-{_px(lv['upper_boundary'], tick)}, "
        f"touched {_hm(lv['touch_time'])}",
        f"Leg   {_px(s.leg['start_price'], tick)} -> {_px(s.leg['end_price'], tick)}; IFVG {s.ifvg['timeframe']} "
        f"{_px(s.ifvg['lower'], tick)}-{_px(s.ifvg['upper'], tick)} inverted",
    ]
    if s.break_even_reference is not None:
        lines.append(f"Break-even reference {_px(s.break_even_reference, tick)} (configurable rule)")
    lines += sizing_lines(s, symbol, config)
    lim = config["execution"]["daily_limits"]
    lines.append(f"Your daily limits: max {lim['max_trades']} trades; stop after a win; after a loss only a new A+ setup.")
    return "\n".join(lines)


def format_forming(symbol: str, d: Decision) -> str:
    tick = get_instrument(symbol).tick_size
    st = d.detail.get("setup", {})
    g = st.get("ifvg")
    if g:
        need = "above" if d.bias_direction == "BULLISH" else "below"
        edge = g["upper"] if d.bias_direction == "BULLISH" else g["lower"]
        what = f"waiting for a {g['timeframe']} candle body close {need} {_px(edge, tick)}"
    else:
        what = d.reason.replace("_", " ").lower()
    return f"{symbol} setup forming {_hm(d.time)} CT: {d.bias_direction}, price in key level; {what}."


def format_premarket(symbol: str, d: Decision) -> str:
    tick = get_instrument(symbol).tick_size
    lines = [f"{symbol} pre-session {_hm(d.time)} CT: bias {format_bias(d)}"]
    draw = d.detail.get("draw")
    if draw:
        lines.append(f"  draw on liquidity {_px(draw['price'], tick)} {draw['liquidity_type']} ({draw['kind']})")
    lv = format_levels(d, tick)
    lines += [f"  key level: {x}" for x in lv] or ["  no key levels yet"]
    if d.status == "NO_TRADE":
        lines.append(f"  -> no trade conditions: {d.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
class Scanner:
    """Evaluates symbols at a point in time and turns decisions into events."""

    def __init__(self, config: dict, symbols: List[str],
                 strategy_factory: Optional[Callable[[str], PBBlakeStrategy]] = None):
        self.cfg = config
        self.symbols = [get_instrument(s).symbol for s in symbols]
        self.factory = strategy_factory or (lambda sym: PBBlakeStrategy(config, get_instrument(sym)))
        sess = config["execution"]["session"]
        self.start = parse_hhmm(sess["start"])
        self.end = parse_hhmm(sess["end"])
        self._last_reason: Dict[str, str] = {}
        self._premarket_done: Dict[str, object] = {}
        self._alerted: set = set()

    def active(self, now_ns: int) -> bool:
        """Minutes when the scanner does work: 5 minutes before the session through its end."""
        t = ns_to_chicago(now_ns)
        m = t.hour * 60 + t.minute
        return self.start - 5 <= m <= self.end

    def in_session(self, now_ns: int) -> bool:
        t = ns_to_chicago(now_ns)
        m = t.hour * 60 + t.minute
        return self.start <= m <= self.end

    def evaluate(self, symbol: str, data: Dict[str, MultiTimeframeData], now_ns: int) -> Decision:
        view = data[symbol].view(now_ns)
        pair = SMT_PAIRS.get(symbol)
        smt_view = data[pair].view(now_ns) if pair in data and self.cfg["strategy"].get("smt_enabled") else None
        return self.factory(symbol).evaluate(view, smt_view)

    def scan(self, data: Dict[str, MultiTimeframeData], now_ns: int) -> List[ScanEvent]:
        events: List[ScanEvent] = []
        local = ns_to_chicago(now_ns)
        minute = local.hour * 60 + local.minute
        premarket_at = self.start - 5
        for sym in self.symbols:
            if sym not in data:
                continue
            if premarket_at <= minute < self.start and self._premarket_done.get(sym) != local.date():
                d = self.evaluate(sym, data, now_ns)
                self._premarket_done[sym] = local.date()
                events.append(ScanEvent(now_ns, sym, "PREMARKET", format_premarket(sym, d)))
            if not self.in_session(now_ns):
                continue
            d = self.evaluate(sym, data, now_ns)
            if d.status == "SIGNAL" and d.signal is not None:
                key = (sym, d.signal.key_level_id, d.time)
                if key not in self._alerted:
                    self._alerted.add(key)
                    events.append(ScanEvent(now_ns, sym, "SIGNAL", format_signal(sym, d, self.cfg),
                                            {"signal": d.signal.to_dict()}))
            elif d.reason.startswith("WAITING_IFVG") and self._last_reason.get(sym) != d.reason:
                events.append(ScanEvent(now_ns, sym, "SETUP_FORMING", format_forming(sym, d)))
            self._last_reason[sym] = d.reason
        return events


# ---------------------------------------------------------------------------
# alert sinks
# ---------------------------------------------------------------------------
class AlertSink:
    PUSH_KINDS = ("SIGNAL", "SETUP_FORMING", "STALE", "DATA_ERROR", "ORDER", "FILL", "EXIT", "ORDER_ERROR", "SKIPPED")

    def __init__(self, log_path: Optional[str] = None, webhook: Optional[str] = None,
                 send: Optional[Callable[[str, str], None]] = None, echo: Callable[[str], None] = print,
                 push_summary: bool = False):
        self.push_kinds = self.PUSH_KINDS + (("PREMARKET", "HEARTBEAT") if push_summary else ())
        self.log_path = log_path
        self.webhook = webhook if webhook is not None else os.environ.get("FBT_ALERT_WEBHOOK")
        self.send = send or _post_webhook
        self.echo = echo

    def emit(self, ev: ScanEvent) -> None:
        self.echo(ev.text)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            new = not os.path.exists(self.log_path)
            row = pd.DataFrame([{"time_ct": str(ns_to_chicago(ev.time)), "symbol": ev.symbol, "kind": ev.kind,
                                 "text": ev.text.replace("\n", " | ")}])
            row.to_csv(self.log_path, mode="a", header=new, index=False)
        if self.webhook and ev.kind in self.push_kinds:
            try:
                self.send(self.webhook, ev.text)
            except Exception as exc:  # an alert failure must not stop the scanner
                self.echo(f"(webhook failed: {type(exc).__name__})")


def _post_webhook(url: str, text: str) -> None:
    if "discord" in url:
        body, ctype = json.dumps({"content": text}).encode(), "application/json"
    else:
        body, ctype = text.encode(), "text/plain; charset=utf-8"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=10):
        pass


# ---------------------------------------------------------------------------
# live loop
# ---------------------------------------------------------------------------
class LiveRunner:
    """Polls a feed once a minute, keeps completed bars, rebuilds timeframes, scans."""

    def __init__(self, scanner: Scanner, feed, sink: AlertSink, history_days: int = 20,
                 stale_minutes: float = 3.0, clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
                 trader=None):
        self.trader = trader  # optional PaperTrader (paper orders through the RiskManager)
        self.scanner = scanner
        self.feed = feed
        self.sink = sink
        self.history_days = history_days
        self.stale = stale_minutes
        self.clock = clock
        self.store: Dict[str, pd.DataFrame] = {}
        self.daily: Dict[str, Optional[pd.DataFrame]] = {}
        self._stale_flag: Dict[str, bool] = {}
        self._warmed: set = set()

    def _warm_symbol(self, sym: str, now: pd.Timestamp) -> bool:
        """Load history for one symbol. Retried each minute until it succeeds."""
        try:
            hist = completed_only(self.feed.history(sym, self.history_days), now)
            self.daily[sym] = self.feed.daily_history(sym)
            self.store[sym] = merge_bars(hist, self.store.get(sym)) if sym in self.store else hist
            self._warmed.add(sym)
            return True
        except Exception as exc:  # unattended: report, retry next minute, never crash
            self.sink.emit(ScanEvent(int(now.value), sym, "DATA_ERROR",
                                     f"{sym} DATA ERROR loading history (will retry): {_describe(exc)}"))
            return False

    def warm_up(self) -> None:
        now = self.clock()
        for sym in self.scanner.symbols:
            self._warm_symbol(sym, now)

    def _trim(self, sym: str, now: pd.Timestamp) -> None:
        keep_from = now - pd.Timedelta(days=self.history_days + 10)
        df = self.store.get(sym)
        if df is not None and len(df) and df.index[0] < keep_from:
            self.store[sym] = df[df.index >= keep_from]

    def step(self) -> List[ScanEvent]:
        now = self.clock()
        # Evaluate at the close of the last completed minute (e.g. 09:00:00 when polled at 09:00:03):
        # a signal is defined by the candle that closed exactly then.
        now_ns = int(now.floor("min").value)
        data: Dict[str, MultiTimeframeData] = {}
        events: List[ScanEvent] = []
        active = self.scanner.active(now_ns)
        for sym in self.scanner.symbols:
            if sym not in self._warmed and not self._warm_symbol(sym, now):
                continue
            if not active:
                continue  # nights and weekends: nothing to scan, keep quiet
            try:
                new = completed_only(self.feed.recent(sym), now)
                self.store[sym] = merge_bars(self.store.get(sym), new)
                lag = minutes_since_last_bar(self.store[sym], now)
                if self.scanner.in_session(now_ns) and lag > self.stale:
                    if not self._stale_flag.get(sym):
                        events.append(ScanEvent(now_ns, sym, "STALE",
                                                f"{sym} STALE DATA: last completed bar is {lag:.0f} min old; "
                                                "not scanning until fresh bars arrive."))
                    self._stale_flag[sym] = True
                    continue
                self._stale_flag[sym] = False
                self._trim(sym, now)
                data[sym] = MultiTimeframeData(self.store[sym], sym, daily_history=self.daily.get(sym))
            except Exception as exc:  # unattended: report and keep running
                events.append(ScanEvent(now_ns, sym, "DATA_ERROR", f"{sym} DATA ERROR: {_describe(exc)}"))
        try:
            events += self.scanner.scan(data, now_ns)
        except Exception as exc:
            events.append(ScanEvent(now_ns, "-", "DATA_ERROR", f"Scanner error (still running): {_describe(exc)}"))
        if self.trader is not None:
            try:
                for te in self.trader.on_minute(now, list(events)):
                    events.append(ScanEvent(te.time, te.symbol, te.kind, te.text, te.data))
            except Exception as exc:  # report loudly, keep running; open stops are held at Alpaca
                events.append(ScanEvent(now_ns, "-", "ORDER_ERROR", f"TRADER ERROR (still running): {_describe(exc)}"))
        for ev in events:
            self.sink.emit(ev)
        return events

    def run_forever(self, delay_after_close: float = 3.0) -> None:  # pragma: no cover - wall-clock loop
        self.warm_up()
        print(DISCLAIMER)
        while True:
            now = pd.Timestamp.now(tz="UTC")
            nxt = now.floor("min") + pd.Timedelta(minutes=1, seconds=delay_after_close)
            time.sleep(max(0.0, (nxt - now).total_seconds()))
            try:
                self.step()
            except Exception as exc:  # last line of defence: never let one bad minute stop the scanner
                print(f"Unexpected error (still running): {_describe(exc)}")


def _describe(exc: Exception) -> str:
    return str(exc) if isinstance(exc, DataError) else f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# replay (historical rehearsal; same code path as live)
# ---------------------------------------------------------------------------
def replay_day(scanner: Scanner, data: Dict[str, MultiTimeframeData], date: str) -> List[ScanEvent]:
    """Scan every completed minute of `date` (CT) from 08:00 to the session end."""
    events: List[ScanEvent] = []
    start = pd.Timestamp(f"{date} 08:00", tz=CHICAGO)
    end = pd.Timestamp(date, tz=CHICAGO) + pd.Timedelta(minutes=scanner.end)
    t = start
    while t <= end:
        events += scanner.scan(data, int(t.tz_convert("UTC").value))
        t += pd.Timedelta(minutes=1)
    return events


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import default_config

    ap = argparse.ArgumentParser(description="PB Blake strategy scanner (alerts only, never trades)")
    sub = ap.add_subparsers(dest="mode", required=True)
    lv = sub.add_parser("live")
    lv.add_argument("--feed", choices=["alpaca", "ibkr", "csv"], required=True)
    lv.add_argument("--symbols", nargs="+", default=None,
                    help="default: SPY QQQ DIA GLD for alpaca, ES NQ YM GC otherwise")
    lv.add_argument("--stale-minutes", type=float, default=None,
                    help="minutes without a new bar before a symbol is flagged (default 5 alpaca, 3 otherwise)")
    lv.add_argument("--folder", default="data/live", help="csv feed: folder with <SYMBOL>.csv files")
    lv.add_argument("--tz", default="UTC")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=7497, help="7497 = TWS paper, 4002 = IB Gateway paper")
    lv.add_argument("--client-id", type=int, default=17)
    lv.add_argument("--history-days", type=int, default=20)
    lv.add_argument("--log", default="results/alerts.csv")
    lv.add_argument("--trade", action="store_true",
                    help="auto PAPER trade signals on Alpaca with bracket orders (entry + stop-loss + target)")
    lv.add_argument("--live", action="store_true", help="refused: live trading is not implemented")
    lv.add_argument("--flatten-time", default="14:50", help="CT time to close open paper trades (ETFs close 15:00)")
    lv.add_argument("--push-summary", action="store_true",
                    help="also send the 08:25 CT pre-session summary to your phone (daily 'still running' check)")
    rp = sub.add_parser("replay")
    rp.add_argument("--data", required=True)
    rp.add_argument("--symbol", required=True)
    rp.add_argument("--smt-data", default=None)
    rp.add_argument("--date", required=True, help="CT date, e.g. 2024-03-12")
    rp.add_argument("--tz", default="UTC")
    args = ap.parse_args(argv)
    cfg = default_config()

    if args.mode == "replay":
        try:
            sym = get_instrument(args.symbol).symbol
            data = {sym: MultiTimeframeData(load_ohlcv(args.data, tz=args.tz), sym)}
            if args.smt_data and SMT_PAIRS.get(sym):
                data[SMT_PAIRS[sym]] = MultiTimeframeData(load_ohlcv(args.smt_data, tz=args.tz), SMT_PAIRS[sym])
        except DataError as exc:
            print(f"DATA ERROR: {exc}", file=sys.stderr)
            return 2
        events = replay_day(Scanner(cfg, [sym]), data, args.date)
        for ev in events:
            print(ev.text)
        if not events:
            print(f"No pre-session summary or setups for {sym} on {args.date} (check the date has data).")
        print(DISCLAIMER)
        return 0

    from .feeds import AlpacaFeed, CSVFolderFeed, IBKRFeed
    symbols = args.symbols or (list(ETFS) if args.feed == "alpaca" else ["ES", "NQ", "YM", "GC"])
    stale = args.stale_minutes or (5.0 if args.feed == "alpaca" else 3.0)
    try:
        if args.feed == "csv":
            feed = CSVFolderFeed(args.folder, args.tz)
        elif args.feed == "alpaca":
            feed = AlpacaFeed()
        else:
            feed = IBKRFeed(args.host, args.port, args.client_id)
    except DataError as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # e.g. TWS not running
        print(f"Could not connect to the data feed: {exc}", file=sys.stderr)
        return 3
    trader = None
    if args.trade or args.live:
        # All order code lives in fbt/trader.py behind the RiskManager; this module only watches.
        from .trader import TradingError, build_paper_trader
        try:
            trader = build_paper_trader(cfg, symbols, feed_name=args.feed, live_flag=args.live,
                                        flatten_time=args.flatten_time)
        except TradingError as exc:
            print(f"TRADING ERROR: {exc}", file=sys.stderr)
            return 4
        print("AUTO PAPER TRADING ON: signals become bracket orders in your Alpaca PAPER account.")
    runner = LiveRunner(Scanner(cfg, symbols), feed, AlertSink(log_path=args.log, push_summary=args.push_summary),
                        history_days=args.history_days, stale_minutes=stale, trader=trader)
    print(f"Scanning {', '.join(runner.scanner.symbols)}; signals only {cfg['execution']['session']['start']}-"
          f"{cfg['execution']['session']['end']} CT. Ctrl+C to stop.")
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

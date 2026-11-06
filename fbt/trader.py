"""Automatic PAPER trading on Alpaca from scanner signals.

AGENTS.md rules implemented here:
* Every order goes through the RiskManager: a signal must be approved
  (session window, daily limits, sizing, stop sanity) and the single-use
  approval is consumed by AlpacaPaperBroker.submit() before any HTTP call.
* Paper only. `resolve_trading_mode` returns "paper" by default; `--live`
  requires LIVE_TRADING_CONFIRMED=yes AND is still refused, because no live
  trading is implemented in this project. Orders go only to
  paper-api.alpaca.markets.
* Keys come only from APCA_API_KEY_ID / APCA_API_SECRET_KEY and are never
  printed; the account number is never logged.

Each signal becomes ONE bracket order (market entry + stop-loss + take-profit,
good-till-cancelled), so the stop sits at Alpaca even if this computer goes
off. The trader:
* never opens a trade while any position or open order exists in a traded symbol;
* tracks the bracket every minute, records the exit and applies the spec 4.5
  daily limits across all symbols;
* flattens any open trade at the flatten time (default 14:50 CT, before the
  15:00 CT ETF close);
* persists its day state to results/trader_state.json so a restart the same
  day does not forget trades.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from .costs import CostModel, get_instrument
from .data import CHICAGO, ns_to_chicago, parse_hhmm
from .indicators import BULLISH
from .risk import Approval, RiskManager
from .strategy import Signal

PAPER_URL = "https://paper-api.alpaca.markets/v2"
ORDER_PREFIX = "pbb-"


class TradingError(RuntimeError):
    pass


def resolve_trading_mode(live_flag: bool, env=None) -> str:
    """AGENTS.md rule 6: paper unless --live AND LIVE_TRADING_CONFIRMED=yes. Live is not implemented."""
    env = os.environ if env is None else env
    if not live_flag:
        return "paper"
    if env.get("LIVE_TRADING_CONFIRMED") != "yes":
        raise TradingError("--live also requires the environment variable LIVE_TRADING_CONFIRMED=yes.")
    raise TradingError("Live trading is not implemented in this project. Only paper trading is available.")


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------
class AlpacaPaperBroker:
    """Minimal Alpaca paper Trading API client. Orders require a RiskManager approval."""

    def __init__(self, risk: RiskManager, http: Optional[Callable] = None, env=None):
        env = os.environ if env is None else env
        self._key = env.get("APCA_API_KEY_ID")
        self._secret = env.get("APCA_API_SECRET_KEY")
        if not self._key or not self._secret:
            raise TradingError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (your Alpaca PAPER keys).")
        self.risk = risk
        self.base = PAPER_URL  # the only endpoint this class can talk to
        self._http = http or self._urllib

    def __repr__(self) -> str:
        return "AlpacaPaperBroker(paper, keys=<hidden>)"

    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret,
                "Content-Type": "application/json"}

    def _urllib(self, method: str, url: str, headers: dict, body: Optional[dict]):
        import urllib.error
        import urllib.request
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode()).get("message", "")
            except Exception:
                pass
            if exc.code in (401, 403) and not detail:
                detail = "keys rejected (are they the PAPER keys?)"
            raise TradingError(f"Alpaca {method} failed: HTTP {exc.code} {detail}".strip()) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise TradingError(f"Could not reach Alpaca trading API: {type(exc).__name__}") from None

    def _call(self, method: str, path: str, body: Optional[dict] = None):
        return self._http(method, f"{self.base}{path}", self._headers(), body)

    # --- reads
    def account(self) -> dict:
        return self._call("GET", "/account")

    def positions(self) -> List[dict]:
        return self._call("GET", "/positions") or []

    def open_orders(self) -> List[dict]:
        return self._call("GET", "/orders?status=open&nested=true&limit=500") or []

    def order(self, order_id: str) -> dict:
        return self._call("GET", f"/orders/{order_id}?nested=true")

    # --- writes (risk-approved only)
    def submit_bracket(self, approval: Approval, symbol: str) -> dict:
        self.risk.consume(approval)  # raises RiskViolation if not approved or reused
        s = approval.signal
        side = "buy" if s.direction == BULLISH else "sell"
        body = {
            "symbol": symbol, "qty": str(int(approval.contracts)), "side": side, "type": "market",
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": f"{s.target:.2f}"},
            "stop_loss": {"stop_price": f"{s.stop:.2f}"},
            "client_order_id": f"{ORDER_PREFIX}{symbol}-{s.signal_time}",
        }
        return self._call("POST", "/orders", body)

    def cancel(self, order_id: str) -> None:
        self._call("DELETE", f"/orders/{order_id}")

    def close_position(self, symbol: str) -> dict:
        return self._call("DELETE", f"/positions/{symbol}")


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
@dataclass
class OpenTrade:
    symbol: str
    order_id: str
    direction: str
    qty: int
    planned_entry: float
    stop: float
    target: float
    signal_time: int
    key_level_id: str
    filled: bool = False
    entry_price: Optional[float] = None
    close_order_id: Optional[str] = None


@dataclass
class TradeEvent:
    time: int
    symbol: str
    kind: str      # ORDER / FILL / EXIT / ORDER_ERROR / SKIPPED
    text: str
    data: dict = field(default_factory=dict)


class PaperTrader:
    def __init__(self, config: dict, broker_factory: Callable[[RiskManager], AlpacaPaperBroker],
                 symbols: List[str], state_path: Optional[str] = "results/trader_state.json",
                 trades_log: Optional[str] = "results/paper_trades.csv", flatten_time: str = "14:50"):
        self.cfg = config
        self.symbols = [get_instrument(s).symbol for s in symbols]
        self.risk = RiskManager(config, CostModel.default(self.symbols[0]))
        self.broker = broker_factory(self.risk)
        self.state_path = state_path
        self.trades_log = trades_log
        self.flatten = parse_hhmm(flatten_time)
        self.open: Optional[OpenTrade] = None
        self._synced_day = None
        self._first_sync = True
        self._load_state()

    # --- persistence (no keys, no account numbers)
    def _load_state(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            st = json.load(open(self.state_path))
        except Exception:
            return
        today = str(pd.Timestamp.now(tz=CHICAGO).date())
        d = self.risk.day
        if st.get("date") == today:
            d.date = pd.Timestamp(today).date()
            d.trades, d.wins, d.losses = st["trades"], st["wins"], st["losses"]
            d.realized = st["realized"]
            d.last_exit_ns = st.get("last_exit_ns")
            d.used_level_ids = set(st.get("used_level_ids", []))
        if st.get("open"):
            self.open = OpenTrade(**st["open"])
            self.risk.position_open = self.open.filled

    def _save_state(self) -> None:
        if not self.state_path:
            return
        d = self.risk.day
        st = {"date": str(d.date) if d.date is not None else None, "trades": d.trades, "wins": d.wins,
              "losses": d.losses, "realized": d.realized, "last_exit_ns": d.last_exit_ns,
              "used_level_ids": sorted(d.used_level_ids),
              "open": self.open.__dict__ if self.open else None}
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh, indent=2)
        os.replace(tmp, self.state_path)

    # --- daily sync
    def _sync_day(self, now: pd.Timestamp) -> List[TradeEvent]:
        local = now.tz_convert(CHICAGO)
        if self._synced_day == local.date():
            return []
        self.risk.new_day(local.date())
        acct = self.broker.account()
        equity = float(acct.get("equity") or acct.get("portfolio_value") or 0)
        if equity <= 0:
            raise TradingError("Alpaca account reports no equity.")
        self.risk.set_equity(equity, reset_peak=self._first_sync)
        self._first_sync = False
        self._synced_day = local.date()
        self._save_state()
        return []

    def _busy(self) -> Optional[str]:
        """Broker-side truth: any position or open order in our symbols blocks a new entry."""
        if self.open is not None:
            return "TRADE_ALREADY_OPEN"
        for p in self.broker.positions():
            if p.get("symbol") in self.symbols:
                return f"POSITION_ALREADY_OPEN ({p.get('symbol')})"
        for o in self.broker.open_orders():
            if o.get("symbol") in self.symbols:
                return f"OPEN_ORDER_EXISTS ({o.get('symbol')})"
        return None

    # --- main hook, called every minute by LiveRunner
    def on_minute(self, now: pd.Timestamp, scan_events) -> List[TradeEvent]:
        out: List[TradeEvent] = []
        now_ns = int(now.floor("min").value)
        local = now.tz_convert(CHICAGO)
        minute = local.hour * 60 + local.minute
        try:
            out += self._sync_day(now)
            if self.open is not None:
                out += self._track(now_ns, minute)
            for ev in scan_events:
                if ev.kind == "SIGNAL" and ev.symbol in self.symbols:
                    out += self._enter(ev, now_ns, minute)
        except TradingError as exc:
            out.append(TradeEvent(now_ns, "-", "ORDER_ERROR", f"TRADING ERROR: {exc}"))
        self._save_state()
        return out

    def _enter(self, ev, now_ns: int, minute: int) -> List[TradeEvent]:
        sig = Signal(**ev.data["signal"])
        busy = self._busy()
        if busy:
            return [TradeEvent(now_ns, ev.symbol, "SKIPPED", f"{ev.symbol} signal not traded: {busy}")]
        self.risk.use_costs(CostModel.default(ev.symbol))
        res = self.risk.approve(sig, now_ns, minute, has_position=False)
        if not isinstance(res, Approval):
            return [TradeEvent(now_ns, ev.symbol, "SKIPPED", f"{ev.symbol} signal not traded: {res.reason}")]
        order = self.broker.submit_bracket(res, ev.symbol)
        self.open = OpenTrade(ev.symbol, order["id"], sig.direction, int(res.contracts), sig.entry_price, sig.stop,
                              sig.target, sig.signal_time, sig.key_level_id)
        side = "BUY" if sig.direction == BULLISH else "SELL SHORT"
        return [TradeEvent(now_ns, ev.symbol, "ORDER",
                           f"PAPER ORDER {side} {res.contracts} {ev.symbol} at market; stop {sig.stop:.2f}, "
                           f"target {sig.target:.2f} (bracket order, risk ${res.risk_dollars:,.2f})")]

    def _track(self, now_ns: int, minute: int) -> List[TradeEvent]:
        t = self.open
        out: List[TradeEvent] = []
        o = self.broker.order(t.close_order_id or t.order_id)
        if t.close_order_id is None and not t.filled:
            status = o.get("status")
            if status == "filled":
                t.filled = True
                t.entry_price = float(o["filled_avg_price"])
                t.qty = int(float(o.get("filled_qty") or t.qty))
                self.risk.record_entry(t.key_level_id)
                out.append(TradeEvent(now_ns, t.symbol, "FILL",
                                      f"PAPER FILL {t.symbol} {t.qty} @ {t.entry_price:.2f}; stop {t.stop:.2f} "
                                      f"and target {t.target:.2f} are live at Alpaca."))
            elif status in ("canceled", "expired", "rejected", "done_for_day"):
                self.risk.day.used_level_ids.add(t.key_level_id)
                self.open = None
                return [TradeEvent(now_ns, t.symbol, "ORDER_ERROR", f"{t.symbol} entry order {status}; no trade.")]
            else:
                return out
        if t.close_order_id is not None:
            if o.get("status") == "filled":
                return out + [self._closed(t, float(o["filled_avg_price"]), "FLATTEN_TIME", now_ns)]
            return out
        for leg in o.get("legs") or []:
            if leg.get("status") == "filled":
                reason = "TARGET" if leg.get("type") == "limit" else "STOP"
                return out + [self._closed(t, float(leg["filled_avg_price"]), reason, now_ns)]
        if minute >= self.flatten:
            for leg in o.get("legs") or []:
                if leg.get("status") in ("new", "accepted", "held", "pending_new", "partially_filled"):
                    self.broker.cancel(leg["id"])
            close = self.broker.close_position(t.symbol)
            t.close_order_id = close.get("id")
            out.append(TradeEvent(now_ns, t.symbol, "ORDER", f"{t.symbol}: closing paper position at "
                                  f"{ns_to_chicago(now_ns).strftime('%H:%M')} CT (end-of-day flatten)."))
        return out

    def _closed(self, t: OpenTrade, exit_px: float, reason: str, now_ns: int) -> TradeEvent:
        sign = 1 if t.direction == BULLISH else -1
        costs = CostModel.default(t.symbol)
        gross = sign * (exit_px - t.entry_price) * t.qty * costs.instrument.point_value
        net = gross - costs.round_trip_commission(t.qty)
        risk_dollars = abs(t.entry_price - t.stop) * t.qty * costs.instrument.point_value
        r = net / risk_dollars if risk_dollars > 0 else 0.0
        self.risk.on_trade_closed(net, now_ns)
        self.open = None
        if self.trades_log:
            os.makedirs(os.path.dirname(self.trades_log) or ".", exist_ok=True)
            new = not os.path.exists(self.trades_log)
            pd.DataFrame([{"signal_time_ct": str(ns_to_chicago(t.signal_time)), "exit_time_ct": str(ns_to_chicago(now_ns)),
                           "symbol": t.symbol, "direction": t.direction, "qty": t.qty, "entry": t.entry_price,
                           "stop": t.stop, "target": t.target, "exit": exit_px, "reason": reason,
                           "net_pnl_est": round(net, 2), "r_multiple": round(r, 3)}]).to_csv(
                self.trades_log, mode="a", header=new, index=False)
        word = "WIN" if net > 0 else "LOSS"
        return TradeEvent(now_ns, t.symbol, "EXIT",
                          f"PAPER EXIT {t.symbol} {reason} @ {exit_px:.2f}: {word} ${net:,.2f} ({r:+.2f}R). "
                          f"Today: {self.risk.day.wins} win(s), {self.risk.day.losses} loss(es).")


def build_paper_trader(config: dict, symbols: List[str], feed_name: str, live_flag: bool = False,
                       flatten_time: str = "14:50", env=None) -> "PaperTrader":
    """Validate the request and build the paper trader (used by `fbt.scanner live --trade`)."""
    resolve_trading_mode(live_flag, env)
    if feed_name != "alpaca":
        raise TradingError("--trade needs --feed alpaca (orders go to your Alpaca PAPER account).")
    bad = [s for s in symbols if get_instrument(s).unit != "share"]
    if bad:
        raise TradingError(f"Alpaca cannot trade futures {bad}; use SPY QQQ DIA GLD.")
    return PaperTrader(config, lambda rm: AlpacaPaperBroker(rm, env=env), symbols, flatten_time=flatten_time)


__all__ = ["PaperTrader", "AlpacaPaperBroker", "build_paper_trader", "resolve_trading_mode", "TradingError", "TradeEvent", "PAPER_URL"]

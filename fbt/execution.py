"""Simulated (paper) execution against historical 1-minute bars.

There is no live broker connection in this project. `PaperBroker` refuses any
order that does not carry a single-use Approval from the RiskManager.

Fill model (conservative):
* Market entry: signal at bar t's close -> filled at bar t+1's OPEN + slippage.
* Limit entry: filled only if a later bar trades THROUGH the limit; fill at
  the limit (or the open if it gapped through). Cancelled at expiry, at the
  session end, or if the target is reached first (do not chase).
* Stop: filled at stop - slippage (long); if the bar opens beyond the stop,
  at the open - slippage.
* Target (limit): requires a trade-through (high > target for longs).
* Stop and target in the same bar -> the stop is assumed first.
* On a limit-entry fill bar only the stop is checked (intrabar order unknown).
* Break-even moves take effect from the next bar and can only tighten.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .costs import CostModel
from .indicators import BULLISH
from .risk import Approval, RiskManager


@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str
    contracts: int
    signal_time: int
    entry_method: str
    planned_entry: float
    entry_time: int = 0
    entry_price: float = 0.0
    initial_stop: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    target_label: str = ""
    planned_rr: float = 0.0
    break_even_reference: Optional[float] = None
    break_even_moved: bool = False
    exit_time: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    risk_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    key_level: dict = field(default_factory=dict)
    ifvg: dict = field(default_factory=dict)
    leg: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PaperBroker:
    MODE = "PAPER"

    def __init__(self, costs: CostModel, risk: RiskManager, symbol: str, mode: str = "PAPER"):
        if mode != "PAPER":
            raise RuntimeError("Only paper mode exists in this project. No live trading is implemented.")
        self.costs = costs
        self.risk = risk
        self.symbol = symbol
        self.tick = costs.instrument.tick_size
        self.pv = costs.instrument.point_value
        self.pending: Optional[dict] = None
        self.position: Optional[Trade] = None
        self.closed: List[Trade] = []
        self._ids = 0

    @property
    def flat(self) -> bool:
        return self.position is None and self.pending is None

    # ------------------------------------------------------------------
    def submit(self, approval: Approval, expiry_ns: Optional[int] = None) -> Trade:
        self.risk.consume(approval)  # raises RiskViolation if not approved
        if not self.flat:
            raise RuntimeError("Broker already has a position or working order; adding is not allowed.")
        s = approval.signal
        self._ids += 1
        trade = Trade(self._ids, self.symbol, s.direction, approval.contracts, s.signal_time, s.entry_method,
                      s.entry_price, initial_stop=s.stop, stop=s.stop, target=s.target, target_label=s.target_label,
                      planned_rr=s.risk_reward, break_even_reference=s.break_even_reference,
                      key_level=s.key_level, ifvg=s.ifvg, leg=s.leg, bias=s.bias)
        kind = "MARKET" if s.entry_method == "IFVG_BODY_CLOSE" else "LIMIT"
        self.pending = {"trade": trade, "kind": kind, "approval": approval, "expiry": expiry_ns}
        return trade

    # ------------------------------------------------------------------
    def on_bar(self, o: float, h: float, l: float, c: float, open_time: int, close_time: int,
               flatten_now: bool = False) -> List[Trade]:
        """Process one completed 1m bar. Returns trades closed on this bar."""
        done: List[Trade] = []
        fill_bar_kind = None
        if self.pending is not None:
            fill_bar_kind = self._try_fill(o, h, l, open_time)
        if self.position is not None:
            t = self.position
            long = t.direction == BULLISH
            fav = (h - t.entry_price) if long else (t.entry_price - l)
            adv = (t.entry_price - l) if long else (h - t.entry_price)
            t.mfe_points = max(t.mfe_points, fav)
            t.mae_points = max(t.mae_points, adv)
            exit_px, reason = self._check_exit(t, o, h, l, check_target=(fill_bar_kind != "LIMIT"))
            if exit_px is None and flatten_now:
                exit_px, reason = self.costs.apply_slippage(c, "SELL" if long else "BUY", "MARKET"), "FLATTEN_TIME"
                self._slip(t, "MARKET")
            if exit_px is not None:
                done.append(self._close(t, exit_px, close_time, reason))
            elif t.break_even_reference is not None and not t.break_even_moved and fill_bar_kind is None:
                hit = h >= t.break_even_reference if long else l <= t.break_even_reference
                if hit:
                    new_stop = t.entry_price
                    if (long and new_stop > t.stop) or (not long and new_stop < t.stop):
                        RiskManager.validate_stop_change(t.direction, t.stop, new_stop)
                        t.stop = new_stop
                    t.break_even_moved = True
        return done

    def _try_fill(self, o, h, l, open_time) -> Optional[str]:
        p = self.pending
        t: Trade = p["trade"]
        long = t.direction == BULLISH
        if p["kind"] == "MARKET":
            t.entry_price = self.costs.apply_slippage(o, "BUY" if long else "SELL", "MARKET")
            self._slip(t, "MARKET")
            self._open(t, open_time)
            return "MARKET"
        # limit order
        if p["expiry"] is not None and open_time >= p["expiry"]:
            self._cancel("EXPIRED")
            return None
        limit = t.planned_entry
        through = l < limit if long else h > limit
        if through:
            px = min(o, limit) if long else max(o, limit)
            t.entry_price = self.costs.apply_slippage(px, "BUY" if long else "SELL", "LIMIT")
            self._slip(t, "LIMIT")
            self._open(t, open_time)
            return "LIMIT"
        reached_target = h >= t.target if long else l <= t.target
        if reached_target:
            self._cancel("TARGET_REACHED_BEFORE_FILL")
        return None

    def _open(self, t: Trade, open_time: int) -> None:
        approval = self.pending["approval"]
        t.entry_time = open_time
        t.commission += self.costs.commission(t.contracts)
        t.risk_dollars = abs(t.entry_price - t.initial_stop) * self.pv * t.contracts
        self.position = t
        self.pending = None
        self.risk.on_entry_filled(approval)

    def _cancel(self, reason: str) -> None:
        self.risk.on_entry_cancelled(self.pending["approval"])
        self.pending = None

    def _slip(self, t: Trade, kind: str) -> None:
        t.slippage_cost += self.costs.slippage_points(kind) * self.pv * t.contracts

    def _check_exit(self, t: Trade, o, h, l, check_target: bool):
        long = t.direction == BULLISH
        if long:
            if o <= t.stop:
                self._slip(t, "STOP")
                return self.costs.apply_slippage(o, "SELL", "STOP"), self._stop_reason(t)
            if l <= t.stop:
                self._slip(t, "STOP")
                return self.costs.apply_slippage(t.stop, "SELL", "STOP"), self._stop_reason(t)
            if check_target and (o > t.target or h > t.target):
                return max(o, t.target), "TARGET"
        else:
            if o >= t.stop:
                self._slip(t, "STOP")
                return self.costs.apply_slippage(o, "BUY", "STOP"), self._stop_reason(t)
            if h >= t.stop:
                self._slip(t, "STOP")
                return self.costs.apply_slippage(t.stop, "BUY", "STOP"), self._stop_reason(t)
            if check_target and (o < t.target or l < t.target):
                return min(o, t.target), "TARGET"
        return None, ""

    @staticmethod
    def _stop_reason(t: Trade) -> str:
        return "BREAK_EVEN_STOP" if t.break_even_moved else "STOP"

    def _close(self, t: Trade, px: float, close_time: int, reason: str) -> Trade:
        long = t.direction == BULLISH
        t.exit_price = px
        t.exit_time = close_time
        t.exit_reason = reason
        t.commission += self.costs.commission(t.contracts)
        t.gross_pnl = ((px - t.entry_price) if long else (t.entry_price - px)) * self.pv * t.contracts
        t.net_pnl = t.gross_pnl - t.commission
        t.r_multiple = t.net_pnl / t.risk_dollars if t.risk_dollars > 0 else 0.0
        self.position = None
        self.closed.append(t)
        self.risk.on_trade_closed(t.net_pnl, close_time)
        return t

    def force_close(self, price: float, close_time: int, reason: str = "END_OF_DATA") -> Optional[Trade]:
        if self.pending is not None:
            self._cancel(reason)
        if self.position is None:
            return None
        t = self.position
        side = "SELL" if t.direction == BULLISH else "BUY"
        self._slip(t, "MARKET")
        return self._close(t, self.costs.apply_slippage(price, side, "MARKET"), close_time, reason)

"""Risk manager. Every order must be approved here before the broker accepts it.

Enforces (docs/STRATEGY.md 4.5, 4.6 and RISK_CONFIG):
* entries only inside the session window (08:30-10:00 America/Chicago);
* one position at a time, never adding to a position;
* daily limits: stop after one win, stop after one loss unless a new A+ setup
  (different key level touched after the losing exit), stop after two
  losses, max two trades;
* daily realized-loss limit and a max-drawdown kill switch;
* fixed-fractional position sizing that includes estimated costs;
* stops may only be tightened, never moved farther away.

This module is research / paper-trading only. It does not talk to a broker.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional, Set

from .costs import CostModel
from .data import parse_hhmm
from .indicators import BULLISH
from .strategy import Signal, TradeContext


class RiskViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class Approval:
    token: int
    signal: Signal
    contracts: int
    risk_dollars: float
    approved_at: int


@dataclass
class Rejection:
    reason: str


@dataclass
class DayState:
    date: object = None
    trades: int = 0
    wins: int = 0
    losses: int = 0
    realized: float = 0.0
    start_equity: float = 0.0
    last_exit_ns: Optional[int] = None
    used_level_ids: Set[str] = field(default_factory=set)


class RiskManager:
    def __init__(self, config: dict, costs: CostModel):
        self.risk = config["risk"]
        self.limits = config["execution"]["daily_limits"]
        sess = config["execution"]["session"]
        if sess["timezone"] != "America/Chicago":
            raise ValueError("Session times are configured for America/Chicago.")
        if sess.get("allow_entries_after_end"):
            raise ValueError("allow_entries_after_end must stay False (spec 4.6).")
        self.session_start = parse_hhmm(sess["start"])
        self.session_end = parse_hhmm(sess["end"])
        self.costs = costs
        self.inst = costs.instrument
        self.equity = float(self.risk["initial_equity"])
        self.peak = self.equity
        self.halted = False
        self.day = DayState(start_equity=self.equity)
        self._tokens = itertools.count(1)
        self._outstanding: Set[int] = set()
        self.position_open = False

    # ------------------------------------------------------------------ day
    def new_day(self, date) -> None:
        if self.day.date != date:
            self.day = DayState(date=date, start_equity=self.equity)

    def in_session(self, local_minute: int) -> bool:
        return self.session_start <= local_minute <= self.session_end

    def day_blocked_reason(self) -> Optional[str]:
        d, lim = self.day, self.limits
        if self.halted:
            return "MAX_DRAWDOWN_HALT"
        if d.trades >= lim["max_trades"]:
            return "MAX_TRADES_REACHED"
        if lim["stop_after_one_win"] and d.wins >= 1:
            return "STOP_AFTER_WIN"
        if lim["stop_after_two_losses"] and d.losses >= 2:
            return "STOP_AFTER_TWO_LOSSES"
        if d.losses >= lim["max_daily_losses"]:
            return "MAX_DAILY_LOSSES"
        if lim["stop_after_one_loss"] and d.losses >= 1 and not lim["allow_new_A_PLUS_setup"]:
            return "STOP_AFTER_LOSS"
        max_loss = d.start_equity * self.risk["max_daily_loss_pct"] / 100.0
        if d.realized <= -max_loss:
            return "DAILY_LOSS_LIMIT"
        return None

    def trade_context(self) -> TradeContext:
        """What the strategy may still look for today."""
        d = self.day
        require_after = None
        if self.limits["stop_after_one_loss"] and d.losses >= 1:
            # A+ exception: only a setup touched after the losing exit, on a new level
            require_after = d.last_exit_ns
        return TradeContext(exclude_level_ids=set(d.used_level_ids), require_touch_after_ns=require_after)

    # ------------------------------------------------------------------ approvals
    def size(self, signal: Signal) -> int:
        risk_dollars = self.equity * self.risk["risk_per_trade_pct"] / 100.0
        per_contract = abs(signal.entry_price - signal.stop) * self.inst.point_value \
            + self.costs.estimated_round_trip_cost(1)
        if per_contract <= 0:
            return 0
        n = math.floor(risk_dollars / per_contract)
        cap = self.inst.max_units if self.inst.max_units is not None else self.risk["max_contracts"]
        return int(max(0, min(n, cap)))

    def approve(self, signal: Signal, now_ns: int, local_minute: int, has_position: bool):
        if has_position or self.position_open:
            return Rejection("POSITION_ALREADY_OPEN")  # never add to a position
        if not self.in_session(local_minute):
            return Rejection("OUTSIDE_SESSION")
        blocked = self.day_blocked_reason()
        if blocked:
            return Rejection(blocked)
        ctx = self.trade_context()
        if signal.key_level_id in ctx.exclude_level_ids:
            return Rejection("KEY_LEVEL_ALREADY_TRADED")
        if ctx.require_touch_after_ns is not None and signal.key_level_touch_time <= ctx.require_touch_after_ns:
            return Rejection("NO_NEW_A_PLUS_SETUP")
        bull = signal.direction == BULLISH
        if (bull and signal.stop >= signal.entry_price) or (not bull and signal.stop <= signal.entry_price):
            return Rejection("STOP_ON_WRONG_SIDE")
        if (bull and signal.target <= signal.entry_price) or (not bull and signal.target >= signal.entry_price):
            return Rejection("TARGET_ON_WRONG_SIDE")
        stop_ticks = abs(signal.entry_price - signal.stop) / self.inst.tick_size
        min_ticks = self.inst.min_stop_ticks if self.inst.min_stop_ticks is not None else self.risk["min_stop_ticks"]
        if stop_ticks + 1e-9 < min_ticks:
            return Rejection("STOP_TOO_TIGHT")
        contracts = self.size(signal)
        if contracts < 1:
            return Rejection("RISK_TOO_SMALL_FOR_ONE_CONTRACT")
        token = next(self._tokens)
        self._outstanding.add(token)
        risk_dollars = abs(signal.entry_price - signal.stop) * self.inst.point_value * contracts
        return Approval(token, signal, contracts, risk_dollars, now_ns)

    def consume(self, approval: Approval) -> None:
        """Called by the broker. An approval can be used exactly once."""
        if not isinstance(approval, Approval) or approval.token not in self._outstanding:
            raise RiskViolation("Order rejected: not approved by the risk manager (or approval reused).")
        self._outstanding.discard(approval.token)

    # ------------------------------------------------------------------ lifecycle
    def on_entry_filled(self, approval: Approval) -> None:
        self.position_open = True
        self.day.trades += 1
        self.day.used_level_ids.add(approval.signal.key_level_id)

    def on_entry_cancelled(self, approval: Approval) -> None:
        # an unfilled limit does not count as a trade, but the level is used up
        self.day.used_level_ids.add(approval.signal.key_level_id)

    def on_trade_closed(self, net_pnl: float, exit_ns: int) -> None:
        self.position_open = False
        self.equity += net_pnl
        self.peak = max(self.peak, self.equity)
        self.day.realized += net_pnl
        self.day.last_exit_ns = exit_ns
        # a scratch (break-even) trade loses fees -> counted as a loss (conservative)
        if net_pnl > 0:
            self.day.wins += 1
        else:
            self.day.losses += 1
        dd_pct = (self.peak - self.equity) / self.peak * 100.0 if self.peak > 0 else 100.0
        if dd_pct >= self.risk["max_drawdown_halt_pct"] or self.equity <= 0:
            self.halted = True

    @staticmethod
    def validate_stop_change(direction: str, old_stop: float, new_stop: float) -> None:
        """Spec 4.5 rule 7: never move a stop farther away."""
        if direction == BULLISH and new_stop < old_stop - 1e-9:
            raise RiskViolation(f"Refusing to widen long stop {old_stop} -> {new_stop}")
        if direction != BULLISH and new_stop > old_stop + 1e-9:
            raise RiskViolation(f"Refusing to widen short stop {old_stop} -> {new_stop}")

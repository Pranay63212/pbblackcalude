"""PB Blake strategy: the decision flow of docs/STRATEGY.md section 5.

`PBBlakeStrategy.evaluate(view, ...)` is called once per completed 1-minute
bar inside the trading session. It sees only a MarketView (completed bars),
never broker state or future data, and returns a Decision. A Decision with a
Signal is a *request*; the risk manager decides whether it becomes an order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .costs import InstrumentSpec
from .data import MarketView
from .indicators import BEARISH, BULLISH, NEUTRAL, FVG, find_cisd
from .structure import (KeyLevel, ManipulationLeg, TouchTracker, calculate_risk_reward,
                        determine_higher_timeframe_bias, find_highest_timeframe_ifvg, identify_draw_on_liquidity,
                        identify_key_levels, identify_manipulation_leg, ifvg_inversion_state, liquidity_targets,
                        select_target)

ENTRY_METHODS = ("IFVG_BODY_CLOSE", "IFVG_LIMIT", "CISD_LIMIT")
STOP_METHODS = ("MANIPULATION_SWING", "CANDLE_BODY", "FVG_BOUNDARY", "LEG_FVG", "ORDER_BLOCK")


@dataclass
class Signal:
    direction: str
    entry_method: str          # IFVG_BODY_CLOSE -> market order next bar; *_LIMIT -> limit order
    entry_price: float         # reference close (market) or limit price
    stop: float
    target: float
    target_label: str
    risk_reward: float
    signal_time: int
    stop_method: str
    break_even_reference: Optional[float]
    key_level: dict
    ifvg: dict
    leg: dict
    bias: dict
    draw_on_liquidity: Optional[dict]
    key_level_id: str = ""
    key_level_touch_time: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    time: int
    status: str                # SIGNAL / WAIT / NO_TRADE
    reason: str
    bias_direction: str = NEUTRAL
    key_level_ids: List[str] = field(default_factory=list)
    signal: Optional[Signal] = None
    # Informational only (bias, draw, levels, what the setup waits for). Never used for decisions.
    detail: dict = field(default_factory=dict)

    def fingerprint(self) -> tuple:
        """Compact, comparable summary (used by lookahead tests)."""
        sig = None
        if self.signal is not None:
            s = self.signal
            sig = (s.direction, s.entry_method, round(s.entry_price, 6), round(s.stop, 6), round(s.target, 6))
        return (self.time, self.status, self.reason, self.bias_direction, tuple(self.key_level_ids), sig)


@dataclass
class TradeContext:
    """Constraints from the risk manager (spec 4.5)."""
    exclude_level_ids: set = field(default_factory=set)
    require_touch_after_ns: Optional[int] = None   # A+ exception after a loss


class PBBlakeStrategy:
    def __init__(self, config: dict, instrument: InstrumentSpec):
        self.cfg = config["strategy"]
        self.exe = config["execution"]
        self.inst = instrument
        self.tick = instrument.tick_size
        if self.exe["entry"]["method"] not in ENTRY_METHODS:
            raise ValueError(f"Unknown entry method {self.exe['entry']['method']}")
        if self.exe["stop_loss"]["default"] not in STOP_METHODS:
            raise ValueError(f"Unknown stop method {self.exe['stop_loss']['default']}")
        self.tracker = TouchTracker()
        self._bias_cache = {}
        self._setup: dict = {}
        self._pair_cache: dict = {}

    # ------------------------------------------------------------------
    def _bias(self, view: MarketView, smt_view: Optional[MarketView]) -> dict:
        tfs = list(self.cfg["bias_timeframes"].values())
        key = tuple(view.count(tf) for tf in tfs)
        if smt_view is not None:
            key += tuple(smt_view.count(tf) for tf in tfs)
        hit = self._bias_cache.get(key)
        if hit is None:
            hit = determine_higher_timeframe_bias(view, self.cfg, self.tick, smt_view)
            self._bias_cache = {key: hit}  # only the latest is ever needed
        return dict(hit)

    def _pair_bias(self, smt_view: MarketView) -> str:
        tfs = list(self.cfg["bias_timeframes"].values())
        key = ("pair",) + tuple(smt_view.count(tf) for tf in tfs)
        hit = self._pair_cache.get(key)
        if hit is None:
            hit = determine_higher_timeframe_bias(smt_view, self.cfg, self.tick)["direction"]
            self._pair_cache = {key: hit}
        return hit

    # ------------------------------------------------------------------
    def evaluate(self, view: MarketView, smt_view: Optional[MarketView] = None,
                 context: Optional[TradeContext] = None) -> Decision:
        ctx = context or TradeContext()
        now = view.now

        # STEP 1: higher-timeframe bias
        bias = self._bias(view, smt_view)
        direction = bias["direction"]
        detail = {"bias": bias}
        if direction == NEUTRAL:
            return Decision(now, "NO_TRADE", "BIAS_NEUTRAL", NEUTRAL, detail=detail)
        # Optional market-agreement filter (not in the spec): the paired market must share the bias.
        if self.cfg.get("require_pair_agreement") and smt_view is not None:
            pair_dir = self._pair_bias(smt_view)
            detail["pair_bias"] = pair_dir
            if pair_dir != direction:
                return Decision(now, "NO_TRADE", "PAIR_DISAGREES", direction, detail=detail)
        draw = identify_draw_on_liquidity(view, direction, self.cfg, self.tick)
        if draw is None:
            return Decision(now, "NO_TRADE", "NO_DRAW_ON_LIQUIDITY", direction, detail=detail)
        bias["draw_on_liquidity"] = draw["price"]
        bias["liquidity_type"] = draw["liquidity_type"]
        detail["draw"] = draw

        # STEP 2: key levels
        levels, _ = identify_key_levels(view, direction, self.cfg, self.tick, self.tracker,
                                        exclude_ids=ctx.exclude_level_ids, smt=bias.get("smt_confirmation"))
        ids = [l.id for l in levels]
        detail["levels"] = [{"type": l.type, "timeframe": l.timeframe, "upper": l.upper_boundary,
                             "lower": l.lower_boundary, "touch_time": l.touch_time if l.touch_index >= 0 else None,
                             "id": l.id} for l in levels]
        if not levels:
            return Decision(now, "WAIT", "NO_KEY_LEVELS", direction, detail=detail)

        # STEP 3: price must have traded into a level
        touched = [l for l in levels if l.touch_index >= 0]
        if ctx.require_touch_after_ns is not None:
            touched = [l for l in touched if l.touch_time > ctx.require_touch_after_ns]
        if not touched:
            return Decision(now, "WAIT", "NO_KEY_LEVEL_REACTION", direction, ids, detail=detail)

        reason, final_status = "WAIT", "WAIT"
        for level in touched:
            self._setup = {"active_level": level.id}
            status, why, signal = self._setup_from_level(view, level, direction, bias, draw)
            if signal is not None:
                detail["setup"] = self._setup
                return Decision(now, "SIGNAL", "IFVG_CONFIRMED", direction, ids, signal, detail=detail)
            if reason == "WAIT":
                reason = why
                final_status = status
                detail["setup"] = self._setup
        return Decision(now, final_status, reason, direction, ids, detail=detail)

    # ------------------------------------------------------------------
    def _setup_from_level(self, view: MarketView, level: KeyLevel, direction: str, bias: dict, draw: dict):
        # STEP 4: manipulation leg
        leg = identify_manipulation_leg(view, level, direction, self.cfg)
        if leg is None:
            return "WAIT", "NO_MANIPULATION_LEG", None
        self._setup["leg"] = leg.to_dict()
        # STEP 5: highest-timeframe IFVG inside the leg
        min_size = self.cfg["min_fvg_ticks"] * self.tick
        gap = find_highest_timeframe_ifvg(view, leg, direction, self.cfg["ifvg_timeframes"], min_size)
        if gap is None:
            return "WAIT", "NO_IFVG_IN_LEG", None
        self._setup["ifvg"] = {"timeframe": gap.tf, "upper": gap.top, "lower": gap.bottom}
        # STEP 6: body close through it on its own timeframe, exactly now
        state, inv_idx = ifvg_inversion_state(view, gap, direction)
        if state == "PENDING":
            return "WAIT", f"WAITING_IFVG_{gap.tf}", None
        if state == "ALREADY_INVERTED":
            return "WAIT", f"IFVG_{gap.tf}_INVERTED_EARLIER", None
        bars = view.bars(gap.tf)
        confirm_close = float(bars.close[inv_idx])

        # STEP 7/8: entry, stop, target
        method = self.exe["entry"]["method"]
        attempt = [method]
        if method == "IFVG_BODY_CLOSE" and self.cfg.get("alternative_entry_on_poor_rr"):
            attempt += list(self.exe["entry"]["alternative_methods"])
        last_reason = "RR_BELOW_MINIMUM"
        for m in attempt:
            entry = self._entry_price(view, m, gap, leg, confirm_close, direction)
            if entry is None:
                last_reason = f"{m}_UNAVAILABLE"
                continue
            stop = self._stop(view, gap, leg, level, inv_idx, direction)
            if stop is None or (direction == BULLISH and stop >= entry) or (direction == BEARISH and stop <= entry):
                last_reason = "INVALID_STOP"
                continue
            targets = liquidity_targets(view, direction, entry, self.cfg, self.tick, leg, draw)
            choice = select_target(direction, entry, stop, targets, self.exe["take_profit"]["minimum_rr"])
            if choice is None:
                last_reason = "RR_BELOW_MINIMUM" if targets else "NO_LIQUIDITY_TARGET"
                continue
            target, label, rr = choice
            be = self._break_even_reference(direction, entry, stop, target, leg)
            sig = Signal(direction, m, entry, stop, target, label, rr, view.now,
                         self.exe["stop_loss"]["default"], be, level.to_dict(),
                         {"timeframe": gap.tf, "upper": gap.top, "lower": gap.bottom, "created_time": gap.created_time,
                          "confirm_close": confirm_close},
                         leg.to_dict(), bias, draw, level.id, level.touch_time)
            return "SIGNAL", "IFVG_CONFIRMED", sig
        return "NO_TRADE", last_reason, None

    # ------------------------------------------------------------------
    def _entry_price(self, view, method, gap: FVG, leg: ManipulationLeg, confirm_close, direction) -> Optional[float]:
        if method == "IFVG_BODY_CLOSE":
            return confirm_close
        if method == "IFVG_LIMIT":
            return gap.top if direction == BULLISH else gap.bottom
        if method == "CISD_LIMIT":
            bars = view.bars(gap.tf)
            j = int(np.searchsorted(bars.open_time, leg.end_time, side="left")) - 1
            cisd = find_cisd(bars, direction, j, len(bars))
            if cisd is None:
                return None
            price = cisd.price
            # a limit must be at or better than the current close, never a chase
            if (direction == BULLISH and price > confirm_close) or (direction == BEARISH and price < confirm_close):
                return None
            return price
        raise ValueError(method)

    def _stop(self, view, gap: FVG, leg: ManipulationLeg, level: KeyLevel, inv_idx: int, direction) -> Optional[float]:
        buf = self.cfg["stop_buffer_ticks"] * self.tick
        method = self.exe["stop_loss"]["default"]
        bull = direction == BULLISH
        bars = view.bars(gap.tf)
        if method == "MANIPULATION_SWING":
            ref = leg.end_price
        elif method == "CANDLE_BODY":
            ref = min(bars.open[inv_idx], bars.close[inv_idx]) if bull else max(bars.open[inv_idx], bars.close[inv_idx])
        elif method == "FVG_BOUNDARY":
            ref = level.lower_boundary if bull else level.upper_boundary
        elif method == "LEG_FVG":
            ref = gap.bottom if bull else gap.top
        elif method == "ORDER_BLOCK":
            # last opposing-close candle on the IFVG timeframe before the inversion
            def opposing(i):
                return (bars.close[i] < bars.open[i]) if bull else (bars.close[i] > bars.open[i])
            j = inv_idx - 1
            while j > gap.c1 and not opposing(j):
                j -= 1
            if j < 0 or not opposing(j):
                return None
            ref = bars.low[j] if bull else bars.high[j]
        else:
            raise ValueError(method)
        stop = float(ref) - buf if bull else float(ref) + buf
        # round away from entry onto the tick grid
        t = self.tick
        stop = np.floor(stop / t + 1e-9) * t if bull else np.ceil(stop / t - 1e-9) * t
        return round(float(stop), 10)

    def _break_even_reference(self, direction, entry, stop, target, leg: ManipulationLeg) -> Optional[float]:
        be = self.exe["break_even"]
        if not be.get("enabled"):
            return None
        if be["method"] == "STRUCTURE_BASED":
            ref = leg.start_price  # the manipulation leg's swing (spec 4.4 example)
            if direction == BULLISH and entry < ref < target:
                return ref
            if direction == BEARISH and target < ref < entry:
                return ref
            return None
        if be["method"] == "FIXED_R":
            r = be.get("fixed_trigger")
            if not r:
                return None
            risk = abs(entry - stop)
            return entry + r * risk if direction == BULLISH else entry - r * risk
        return None


__all__ = ["PBBlakeStrategy", "Signal", "Decision", "TradeContext", "calculate_risk_reward"]

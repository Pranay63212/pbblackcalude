"""Market-structure analysis built on indicators.py, following docs/STRATEGY.md.

All functions take a data.MarketView (completed bars only). Touches of key
levels are measured on completed 1-minute bars; respect / invalidation of a
gap is measured with body closes on the gap's own timeframe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import Bars, MarketView, NS_PER_MIN, timeframe_minutes
from .indicators import (BEARISH, BULLISH, NEUTRAL, FVG, find_cisd, find_fvgs, find_rejection_block,
                         first_true, gap_status, opposite, swing_highs, swing_lows)

PRIORITY = {"FVG": 1, "INTERMEDIATE_HIGH_LOW": 1, "CISD": 2, "REJECTION_BLOCK": 3}


# ---------------------------------------------------------------------------
# Forward-only cache for first touches on 1-minute bars
# ---------------------------------------------------------------------------
class TouchTracker:
    """Finds the first completed 1m bar after `after_ns` that reaches `price`.

    Results are cached per query and extended incrementally as time moves
    forward. Only bars inside the supplied (truncated) 1m arrays are scanned,
    so the cache can never contain information from the future. If time moves
    backwards the cache is discarded.
    """

    def __init__(self):
        self._cache: Dict[tuple, Tuple[int, int]] = {}
        self._last_n = 0

    def first(self, m1: Bars, after_ns: int, price: float, direction: str, strict: bool) -> int:
        n = len(m1)
        if n < self._last_n:
            self._cache.clear()
        self._last_n = n
        key = (after_ns, round(price, 8), direction, strict)
        hit = self._cache.get(key)
        if hit is not None:
            scanned, idx = hit
            if idx >= 0:
                return idx
            start = scanned
        else:
            start = int(np.searchsorted(m1.open_time, after_ns, side="left"))
        if start >= n:
            self._cache[key] = (start, -1)
            return -1
        if direction == BULLISH:
            seg = m1.low[start:n]
            mask = seg < price if strict else seg <= price
        else:
            seg = m1.high[start:n]
            mask = seg > price if strict else seg >= price
        k = first_true(mask)
        idx = start + k if k >= 0 else -1
        self._cache[key] = (n, idx)
        return idx


# ---------------------------------------------------------------------------
# Step 1: bias
# ---------------------------------------------------------------------------
def timeframe_bias(bars: Bars, lookback: int, min_size: float = 0.0) -> dict:
    """Bias of one timeframe from FVG respect / disrespect (spec 1.2)."""
    n = len(bars)
    start = max(0, n - lookback)
    bull = bear = 0
    events = []
    for gap in find_fvgs(bars, start, n, min_size):
        state = gap_status(bars, gap, n).state
        if state == "PENDING":
            continue
        if (gap.direction == BULLISH) == (state == "RESPECTED"):
            bull += 1
        else:
            bear += 1
        events.append((gap.direction, state))
    if bull > bear:
        d = BULLISH
    elif bear > bull:
        d = BEARISH
    else:
        d = NEUTRAL
    return {"direction": d, "bullish_points": bull, "bearish_points": bear, "events": len(events)}


def smt_divergence(view_a: MarketView, view_b: MarketView, tf: str, lookback: int, recent: int,
                   direction: str) -> Optional[bool]:
    """Spec 1.4. Bullish SMT: exactly one instrument swept its prior low.

    The prior extreme is taken over the window excluding the last `recent`
    bars; a sweep means the last `recent` bars traded beyond it.
    Returns None if data is insufficient.
    """
    a, b = view_a.bars(tf), view_b.bars(tf)
    if len(a) < lookback or len(b) < lookback or recent >= lookback:
        return None

    def swept(bars: Bars) -> bool:
        if direction == BULLISH:
            lo = bars.low[-lookback:]
            return float(lo[-recent:].min()) < float(lo[:-recent].min())
        hi = bars.high[-lookback:]
        return float(hi[-recent:].max()) > float(hi[:-recent].max())

    return swept(a) != swept(b)


def determine_higher_timeframe_bias(view: MarketView, cfg: dict, tick: float,
                                    smt_view: Optional[MarketView] = None) -> dict:
    tfs = cfg["bias_timeframes"]
    lb = cfg["bias_lookback_bars"]
    min_size = cfg["min_fvg_ticks"] * tick
    per = {}
    for key, tf in tfs.items():
        per[key] = timeframe_bias(view.bars(tf), lb[tf], min_size)
    primary = [per["daily"]["direction"], per["four_hour"]["direction"], per["one_hour"]["direction"]]
    direction = NEUTRAL
    for d in (BULLISH, BEARISH):
        agree = primary.count(d)
        opposed = primary.count(opposite(d))
        if agree >= cfg["bias_min_agreement"] and (cfg["bias_allow_opposing"] or opposed == 0):
            direction = d
    fifteen = per["fifteen_minute"]["direction"]
    confirmation = fifteen == direction if direction != NEUTRAL else None
    if direction != NEUTRAL and cfg["require_15m_confirmation"] and not confirmation:
        direction = NEUTRAL
    smt = None
    if smt_view is not None and cfg.get("smt_enabled") and direction != NEUTRAL:
        smt = smt_divergence(view, smt_view, cfg["smt_timeframe"], cfg["smt_lookback_bars"],
                             cfg["smt_recent_bars"], direction)
    return {
        "direction": direction,
        "draw_on_liquidity": None,
        "liquidity_type": None,
        "daily_bias": per["daily"]["direction"],
        "four_hour_bias": per["four_hour"]["direction"],
        "one_hour_bias": per["one_hour"]["direction"],
        "fifteen_minute_confirmation": confirmation,
        "smt_confirmation": smt,
    }


def _max_high_since(view: MarketView, bars: Bars, idx: int) -> float:
    """Highest high after bar idx of `bars`, including completed 1m bars after its last bar."""
    vals = [-np.inf]
    if idx + 1 < len(bars):
        vals.append(float(bars.high[idx + 1:].max()))
    m1 = view.bars("1m")
    if len(bars):
        j = int(np.searchsorted(m1.open_time, bars.close_time[-1], side="left"))
        if j < len(m1):
            vals.append(float(m1.high[j:].max()))
    return max(vals)


def _min_low_since(view: MarketView, bars: Bars, idx: int) -> float:
    vals = [np.inf]
    if idx + 1 < len(bars):
        vals.append(float(bars.low[idx + 1:].min()))
    m1 = view.bars("1m")
    if len(bars):
        j = int(np.searchsorted(m1.open_time, bars.close_time[-1], side="left"))
        if j < len(m1):
            vals.append(float(m1.low[j:].min()))
    return min(vals)


def unswept_swings(view: MarketView, tf: str, direction: str, strength: int, lookback: int,
                   beyond: float) -> List[Tuple[float, int]]:
    """Untouched swing highs above `beyond` (bullish) / swing lows below it (bearish)."""
    bars = view.bars(tf)
    n = len(bars)
    start = max(0, n - lookback)
    out = []
    if direction == BULLISH:
        for i in swing_highs(bars.high, strength, start, n):
            p = float(bars.high[i])
            if p > beyond and _max_high_since(view, bars, int(i)) < p:
                out.append((p, int(i)))
    else:
        for i in swing_lows(bars.low, strength, start, n):
            p = float(bars.low[i])
            if p < beyond and _min_low_since(view, bars, int(i)) > p:
                out.append((p, int(i)))
    return out


def identify_draw_on_liquidity(view: MarketView, direction: str, cfg: dict, tick: float) -> Optional[dict]:
    """Spec 1.3: most obvious external swing / relative equal highs or lows beyond price."""
    if direction == NEUTRAL:
        return None
    price = view.last_close
    tol = cfg["equal_level_tolerance_ticks"] * tick
    k = cfg["swing_strength"]["htf"]
    cands = []  # (price, tf)
    for tf in cfg["liquidity_timeframes"]:
        for p, _ in unswept_swings(view, tf, direction, k, cfg["liquidity_lookback_bars"][tf], price):
            cands.append((p, tf))
    if not cands:
        return None
    dist = (lambda p: p - price) if direction == BULLISH else (lambda p: price - p)
    # relative equal highs/lows: two or more swings within tolerance
    prices = sorted({p for p, _ in cands})
    equal = []
    for i in range(len(prices) - 1):
        if prices[i + 1] - prices[i] <= tol + 1e-9:
            pool = prices[i + 1] if direction == BULLISH else prices[i]
            equal.append(pool)
    liq_type = "BUY_SIDE" if direction == BULLISH else "SELL_SIDE"
    if equal:
        best = min(equal, key=dist)
        return {"price": best, "timeframe": "multi", "kind": "RELATIVE_EQUAL", "liquidity_type": liq_type}
    # otherwise: nearest unswept swing on the highest timeframe that has one
    tf = max({t for _, t in cands}, key=timeframe_minutes)
    best = min((p for p, t in cands if t == tf), key=dist)
    return {"price": best, "timeframe": tf, "kind": "SWING", "liquidity_type": liq_type}


# ---------------------------------------------------------------------------
# Step 2: key levels
# ---------------------------------------------------------------------------
@dataclass
class KeyLevel:
    type: str
    direction: str
    timeframe: str
    upper_boundary: float
    lower_boundary: float
    midpoint: float
    mitigated: bool = False
    swept: bool = False
    cisd_confirmed: bool = False
    fvg_confluence: bool = False
    smt_confluence: bool = False
    valid: bool = False
    # bookkeeping (not in spec dict)
    id: str = ""
    source_id: str = ""
    created_time: int = 0
    touch_index: int = -1          # 1m index of the touch / sweep
    touch_time: int = 0
    overlaps_intermediate: bool = False
    multi_tf_cisd: bool = False
    ce: Optional[float] = None
    gap: Optional[FVG] = field(default=None, repr=False)

    @property
    def priority(self) -> int:
        return PRIORITY[self.type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("gap", None)
        return d

    def touch_price(self) -> float:
        return self.upper_boundary if self.direction == BULLISH else self.lower_boundary


def _fresh(view: MarketView, m1: Bars, idx: int, max_age_min: float) -> bool:
    return idx >= 0 and (view.now - int(m1.close_time[idx])) <= max_age_min * NS_PER_MIN


def _make_level(type_, direction, tf, upper, lower, created, source_id, gap=None, **kw) -> KeyLevel:
    lvl = KeyLevel(type_, direction, tf, float(upper), float(lower), (float(upper) + float(lower)) / 2.0,
                   id=f"{type_}:{tf}:{direction}:{created}:{round(float(upper), 6)}", source_id=source_id,
                   created_time=int(created), gap=gap, **kw)
    return lvl


def _attach_touch(view, m1, lvl: KeyLevel, tracker: TouchTracker, strict: bool, max_age: float,
                  after_ns: Optional[int] = None) -> Optional[KeyLevel]:
    """Mark the first 1m touch. Returns None if it was touched too long ago (used up)."""
    after = lvl.created_time if after_ns is None else after_ns
    idx = tracker.first(m1, after, lvl.touch_price(), lvl.direction, strict)
    if idx < 0:
        return lvl
    if not _fresh(view, m1, idx, max_age):
        return None
    lvl.touch_index = idx
    lvl.touch_time = int(m1.close_time[idx])
    return lvl


def _levels_for_timeframe(view: MarketView, tf: str, direction: str, cfg: dict, tick: float,
                          tracker: TouchTracker) -> Tuple[List[KeyLevel], List[FVG]]:
    bars = view.bars(tf)
    m1 = view.bars("1m")
    n = len(bars)
    lookback = cfg["key_level_lookback_bars"][tf]
    min_size = cfg["min_fvg_ticks"] * tick
    max_age = cfg["max_touch_age_minutes"]
    k_int = cfg["swing_strength"]["intermediate"]
    levels: List[KeyLevel] = []
    live_gaps: List[FVG] = []
    bull = direction == BULLISH
    for gap in find_fvgs(bars, max(0, n - lookback), n, min_size, direction):
        st = gap_status(bars, gap, n)
        if st.invalidated >= 0:
            continue
        live_gaps.append(gap)
        # --- 2.1A unmitigated FVG (or one whose first touch is the current setup)
        lvl = _make_level("FVG", direction, tf, gap.top, gap.bottom, gap.created_time, gap.id, gap=gap)
        lvl = _attach_touch(view, m1, lvl, tracker, strict=False, max_age=max_age)
        intermediate_price = None
        if lvl is not None:
            lvl.mitigated = lvl.touch_index >= 0
            lvl.valid = True
            levels.append(lvl)
        else:
            # --- 2.1B mitigated: use the intermediate low/high inside the gap
            if bull:
                sw = swing_lows(bars.low, k_int, gap.c3 + 1, n)
                sw = [i for i in sw if gap.bottom <= bars.low[i] <= gap.top]
            else:
                sw = swing_highs(bars.high, k_int, gap.c3 + 1, n)
                sw = [i for i in sw if gap.bottom <= bars.high[i] <= gap.top]
            if sw:
                i = int(sw[-1])
                p = float(bars.low[i] if bull else bars.high[i])
                intermediate_price = p
                confirm_time = int(bars.close_time[min(i + k_int, n - 1)])
                upper, lower = (p, gap.bottom) if bull else (gap.top, p)
                il = _make_level("INTERMEDIATE_HIGH_LOW", direction, tf, upper, lower, confirm_time, gap.id,
                                 gap=gap, mitigated=True)
                il = _attach_touch(view, m1, il, tracker, strict=True, max_age=max_age)
                if il is not None:
                    il.swept = il.touch_index >= 0
                    il.valid = True
                    levels.append(il)
        # --- 2.2 CISD after price traded into this gap (on this timeframe)
        if st.touched >= 0:
            cisd = find_cisd(bars, direction, st.touched, n, st.invalidated)
            if cisd is not None:
                s, e = cisd.series_start, cisd.series_end
                if bull:
                    upper, lower = cisd.price, float(bars.low[s:e + 1].min())
                else:
                    upper, lower = float(bars.high[s:e + 1].max()), cisd.price
                inval = first_true(bars.close[cisd.confirm_index + 1:n] < lower) if bull else \
                    first_true(bars.close[cisd.confirm_index + 1:n] > upper)
                if inval < 0 and upper > lower:
                    cl = _make_level("CISD", direction, tf, upper, lower, cisd.created_time, gap.id, gap=gap,
                                     cisd_confirmed=True)
                    cl.overlaps_intermediate = intermediate_price is not None and lower <= intermediate_price <= upper
                    cl = _attach_touch(view, m1, cl, tracker, strict=False, max_age=max_age)
                    if cl is not None:
                        cl.mitigated = cl.touch_index >= 0
                        cl.valid = True
                        levels.append(cl)
        # --- 2.3 rejection block whose wick traded into this gap
        rb = find_rejection_block(bars, gap, n, cfg["min_wick_ticks"] * tick)
        if rb is not None:
            inval = first_true(bars.close[rb.index + 1:n] < rb.wick_low) if bull else \
                first_true(bars.close[rb.index + 1:n] > rb.wick_high)
            if inval < 0:
                rl = _make_level("REJECTION_BLOCK", direction, tf, rb.wick_high, rb.wick_low, rb.created_time,
                                 gap.id, gap=gap, fvg_confluence=True)
                rl.ce = rb.ce
                rl = _attach_touch(view, m1, rl, tracker, strict=False, max_age=max_age)
                if rl is not None:
                    rl.mitigated = rl.touch_index >= 0
                    rl.valid = True
                    levels.append(rl)
    return levels, live_gaps


def identify_key_levels(view: MarketView, direction: str, cfg: dict, tick: float, tracker: TouchTracker,
                        exclude_ids: Optional[set] = None, smt: Optional[bool] = None
                        ) -> Tuple[List[KeyLevel], List[KeyLevel]]:
    """Returns (selected levels <= max_key_levels, all candidates). Spec 2.4."""
    exclude_ids = exclude_ids or set()
    all_levels: List[KeyLevel] = []
    gaps: List[FVG] = []
    for tf in cfg["key_level_timeframes"]:
        lv, g = _levels_for_timeframe(view, tf, direction, cfg, tick, tracker)
        all_levels.extend(lv)
        gaps.extend(g)
    tol = cfg["cisd_multi_tf_tolerance_ticks"] * tick
    cisds = [l for l in all_levels if l.type == "CISD"]
    for lvl in all_levels:
        if lvl.type == "CISD":
            lvl.fvg_confluence = any(g.bottom <= lvl.upper_boundary and lvl.lower_boundary <= g.top for g in gaps)
            ref = lvl.touch_price()
            lvl.multi_tf_cisd = any(o is not lvl and o.timeframe != lvl.timeframe and abs(o.touch_price() - ref) <= tol
                                    for o in cisds)
        elif lvl.type in ("FVG", "INTERMEDIATE_HIGH_LOW"):
            lvl.fvg_confluence = True
        if smt:
            lvl.smt_confluence = True
    price = view.last_close
    candidates = [l for l in all_levels if l.valid and l.id not in exclude_ids]

    def distance(l: KeyLevel) -> float:
        return max(0.0, price - l.upper_boundary) if direction == BULLISH else max(0.0, l.lower_boundary - price)

    candidates.sort(key=lambda l: (l.priority, distance(l), -timeframe_minutes(l.timeframe), l.id))
    return candidates[: cfg["max_key_levels"]], all_levels


# ---------------------------------------------------------------------------
# Step 3: manipulation leg and IFVG
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ManipulationLeg:
    direction: str
    start_index: int   # 1m index of the swing high (bullish) / swing low (bearish)
    end_index: int     # 1m index of the extreme that traded into the level
    start_price: float
    end_price: float
    start_time: int    # open time of start bar
    end_time: int      # close time of end bar

    def to_dict(self) -> dict:
        return asdict(self)


def identify_manipulation_leg(view: MarketView, level: KeyLevel, direction: str, cfg: dict
                              ) -> Optional[ManipulationLeg]:
    """Spec 3.2: swing high -> swing low (bullish) that traded into the key level.

    End = most extreme 1m bar from the touch up to now. Start = the most recent
    confirmed 1m swing before the end that is the extreme of everything between
    it and the end, and lies beyond the key level.
    """
    if level.touch_index < 0:
        return None
    m1 = view.bars("1m")
    n = len(m1)
    p = level.touch_index
    if p >= n:
        return None
    k = cfg["swing_strength"]["leg"]
    w0 = max(0, p - cfg["leg_max_bars"])
    if direction == BULLISH:
        end = p + int(np.argmin(m1.low[p:n]))
        cands = swing_highs(m1.high, k, w0, n)
        for i in reversed(cands[cands < end]):
            i = int(i)
            hi = float(m1.high[i])
            if hi >= float(m1.high[i:end + 1].max()) and hi > level.upper_boundary:
                return ManipulationLeg(direction, i, end, hi, float(m1.low[end]),
                                       int(m1.open_time[i]), int(m1.close_time[end]))
    else:
        end = p + int(np.argmax(m1.high[p:n]))
        cands = swing_lows(m1.low, k, w0, n)
        for i in reversed(cands[cands < end]):
            i = int(i)
            lo = float(m1.low[i])
            if lo <= float(m1.low[i:end + 1].min()) and lo < level.lower_boundary:
                return ManipulationLeg(direction, i, end, lo, float(m1.high[end]),
                                       int(m1.open_time[i]), int(m1.close_time[end]))
    return None


def fvgs_within_leg(view: MarketView, tf: str, leg: ManipulationLeg, gap_direction: str,
                    min_size: float) -> List[FVG]:
    """FVGs of `gap_direction` formed inside the leg on `tf` (spec 3.2), still valid at the leg's end."""
    bars = view.bars(tf)
    n = len(bars)
    if n < 3:
        return []
    i0 = max(0, int(np.searchsorted(bars.open_time, leg.start_time, side="right")) - 1)
    i1 = int(np.searchsorted(bars.open_time, leg.end_time, side="left")) - 1
    i1 = min(i1, n - 1)
    if i1 - i0 < 2:
        return []
    gaps = find_fvgs(bars, i0, i1 + 1, min_size, gap_direction)
    out = []
    for g in gaps:
        # discard a gap already inverted before the leg finished (it was not the last leg's gap)
        stop = int(np.searchsorted(bars.close_time, leg.end_time, side="right"))
        seg = bars.close[g.c3 + 1:stop]
        inverted = (seg > g.top).any() if gap_direction == BEARISH else (seg < g.bottom).any()
        if not inverted:
            out.append(g)
    return out


def find_highest_timeframe_ifvg(view: MarketView, manipulation_leg: ManipulationLeg, direction: str,
                                timeframes: List[str], min_size: float = 0.0) -> Optional[FVG]:
    """Spec 3.6: the FVG on the highest timeframe that has an opposing FVG inside the leg."""
    valid_ifvgs = []
    for timeframe in timeframes:
        gaps = fvgs_within_leg(view, timeframe, manipulation_leg, opposite(direction), min_size)
        for gap in gaps:
            if direction == BULLISH and gap.type == BEARISH:
                valid_ifvgs.append(gap)
            elif direction == BEARISH and gap.type == BULLISH:
                valid_ifvgs.append(gap)
    if not valid_ifvgs:
        return None
    return max(valid_ifvgs, key=lambda gap: timeframe_minutes(gap.timeframe))


def ifvg_inversion_state(view: MarketView, gap: FVG, direction: str) -> Tuple[str, int]:
    """'CONFIRMED_NOW' if the candle that closed exactly now is the first body close
    through the gap; 'ALREADY_INVERTED' if that happened earlier; else 'PENDING'.
    Returns (state, index of inverting candle on the gap's timeframe)."""
    bars = view.bars(gap.tf)
    n = len(bars)
    seg = bars.close[gap.c3 + 1:n]
    k = first_true(seg > gap.top) if direction == BULLISH else first_true(seg < gap.bottom)
    if k < 0:
        return "PENDING", -1
    j = gap.c3 + 1 + k
    if j == n - 1 and int(bars.close_time[j]) == view.now:
        return "CONFIRMED_NOW", j
    return "ALREADY_INVERTED", j


# ---------------------------------------------------------------------------
# Step 4: targets
# ---------------------------------------------------------------------------
def liquidity_targets(view: MarketView, direction: str, entry: float, cfg: dict, tick: float,
                      leg: Optional[ManipulationLeg], draw: Optional[dict]) -> List[Tuple[float, str]]:
    """Candidate take-profit levels beyond entry, nearest first (spec 4.3)."""
    out: List[Tuple[float, str]] = []
    k = cfg["swing_strength"]["target"]
    for tf in cfg["target_swing_timeframes"]:
        for p, _ in unswept_swings(view, tf, direction, k, cfg["liquidity_lookback_bars"].get(tf, 120), entry):
            out.append((p, f"SWING_{tf}"))
    if leg is not None:
        if (direction == BULLISH and leg.start_price > entry) or (direction == BEARISH and leg.start_price < entry):
            out.append((leg.start_price, "MANIPULATION_LEG_SWING"))
    m1 = view.bars("1m")
    min_size = cfg["min_fvg_ticks"] * tick
    for tf in cfg["target_fvg_timeframes"]:
        bars = view.bars(tf)
        n = len(bars)
        for g in find_fvgs(bars, max(0, n - cfg["key_level_lookback_bars"].get(tf, 96)), n, min_size):
            j = int(np.searchsorted(m1.open_time, g.created_time, side="left"))
            if direction == BULLISH and g.bottom > entry:
                if j >= len(m1) or m1.high[j:].max() < g.bottom:
                    out.append((g.bottom, f"UNFILLED_FVG_{tf}"))
            elif direction == BEARISH and g.top < entry:
                if j >= len(m1) or m1.low[j:].min() > g.top:
                    out.append((g.top, f"UNFILLED_FVG_{tf}"))
    if draw is not None:
        p = draw["price"]
        if (direction == BULLISH and p > entry) or (direction == BEARISH and p < entry):
            out.append((p, "DRAW_ON_LIQUIDITY"))
    out.sort(key=lambda t: abs(t[0] - entry))
    return out


def calculate_risk_reward(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(target - entry) / risk


def select_target(direction: str, entry: float, stop: float, targets: List[Tuple[float, str]],
                  minimum_rr: float) -> Optional[Tuple[float, str, float]]:
    """Closest valid liquidity objective with R:R >= minimum (spec 4.3)."""
    for price, label in targets:
        if direction == BULLISH and price <= entry:
            continue
        if direction == BEARISH and price >= entry:
            continue
        rr = calculate_risk_reward(entry, stop, price)
        if rr + 1e-9 >= minimum_rr:
            return price, label, rr
    return None


__all__ = [
    "TouchTracker", "timeframe_bias", "determine_higher_timeframe_bias", "identify_draw_on_liquidity",
    "KeyLevel", "identify_key_levels", "ManipulationLeg", "identify_manipulation_leg",
    "find_highest_timeframe_ifvg", "fvgs_within_leg", "ifvg_inversion_state", "liquidity_targets",
    "select_target", "calculate_risk_reward", "smt_divergence", "unswept_swings", "session_levels",
]


# ---------------------------------------------------------------------------
# Pre-session mark-out (information for the trader; not used in decisions)
# ---------------------------------------------------------------------------
def session_levels(view: MarketView) -> dict:
    """Classic liquidity levels from completed bars: previous day, current session so far
    (overnight), previous week. `taken_*` = today's session already traded beyond it."""
    from .data import ns_to_chicago
    out = {"price": view.last_close}
    d = view.bars("1d")
    m1 = view.bars("1m")
    # current session = since the most recent 17:00 CT (works even before any daily bar exists)
    local_now = ns_to_chicago(view.now)
    open_local = local_now.normalize() + pd.Timedelta(hours=17)
    if local_now < open_local:
        open_local -= pd.Timedelta(days=1)
    j = int(np.searchsorted(m1.open_time, open_local.tz_convert("UTC").value, side="left"))
    if j < len(m1):
        out["overnight_high"], out["overnight_low"] = float(m1.high[j:].max()), float(m1.low[j:].min())
    if len(d) == 0:
        return out
    out["prev_day_high"], out["prev_day_low"] = float(d.high[-1]), float(d.low[-1])
    if "overnight_high" in out:
        out["taken_prev_day_high"] = out["overnight_high"] > out["prev_day_high"]
        out["taken_prev_day_low"] = out["overnight_low"] < out["prev_day_low"]
    today = local_now.date()
    monday = pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).dayofweek)
    prev_mon, prev_sun = (monday - pd.Timedelta(days=7)).date(), (monday - pd.Timedelta(days=1)).date()
    dates = [ns_to_chicago(int(t)).date() for t in d.close_time]
    idx = [k for k, dt in enumerate(dates) if prev_mon <= dt <= prev_sun]
    if idx:
        out["prev_week_high"] = float(d.high[idx].max())
        out["prev_week_low"] = float(d.low[idx].min())
    return out

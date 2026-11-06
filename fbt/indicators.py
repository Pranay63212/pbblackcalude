"""Pure price-structure primitives: FVGs, swings, gap status, CISD, rejection blocks.

Every function takes arrays that have ALREADY been truncated to completed bars
(see data.MarketView), plus an explicit `end` where relevant. Nothing here
reads beyond `end`.

Definitions (docs/STRATEGY.md):
* Bullish FVG at candles (c1, c2, c3): low[c3] > high[c1]. Gap = [high[c1], low[c3]].
* Bearish FVG: high[c3] < low[c1]. Gap = [high[c3], low[c1]].
  An FVG exists only once c3 has closed (created_time = c3 close).
* Mitigated / touched: a later bar trades into the gap
  (bullish: low <= top; bearish: high >= bottom).
* Disrespected / invalidated: a later candle BODY CLOSE through the relevant
  boundary (bullish: close < bottom; bearish: close > top). Wicks do not count.
* Respected: touched, not invalidated, then a candle closes back outside the gap
  in the gap's direction (bullish: close > top; bearish: close < bottom).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import Bars

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"


def opposite(direction: str) -> str:
    return BEARISH if direction == BULLISH else BULLISH


def first_true(mask: np.ndarray) -> int:
    """Index of the first True, or -1."""
    if mask.size == 0:
        return -1
    i = int(np.argmax(mask))
    return i if mask[i] else -1


# ---------------------------------------------------------------------------
# Fair value gaps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FVG:
    tf: str
    direction: str
    top: float
    bottom: float
    c1: int              # index of first candle in its timeframe's bars
    c3: int              # index of third candle
    c1_open_time: int    # ns
    created_time: int    # ns, close of c3

    @property
    def upper_boundary(self) -> float:
        return self.top

    @property
    def lower_boundary(self) -> float:
        return self.bottom

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def id(self) -> str:
        return f"FVG:{self.tf}:{self.direction}:{self.created_time}"

    # spec 3.6 uses gap.type / gap.timeframe
    @property
    def type(self) -> str:
        return self.direction

    @property
    def timeframe(self) -> str:
        return self.tf


def find_fvgs(bars: Bars, start: int = 0, end: Optional[int] = None, min_size: float = 0.0,
              direction: Optional[str] = None) -> List[FVG]:
    """All FVGs whose three candles lie in bars[start:end], chronological order."""
    n = len(bars) if end is None else min(end, len(bars))
    start = max(0, start)
    if n - start < 3:
        return []
    h = bars.high[start:n]
    l = bars.low[start:n]
    out: List[FVG] = []
    if direction in (None, BULLISH):
        gap = l[2:] - h[:-2]
        for k in np.flatnonzero((gap > 0) & (gap + 1e-12 >= min_size)):
            c1 = start + int(k)
            out.append(FVG(bars.tf, BULLISH, float(l[k + 2]), float(h[k]), c1, c1 + 2,
                           int(bars.open_time[c1]), int(bars.close_time[c1 + 2])))
    if direction in (None, BEARISH):
        gap = l[:-2] - h[2:]
        for k in np.flatnonzero((gap > 0) & (gap + 1e-12 >= min_size)):
            c1 = start + int(k)
            out.append(FVG(bars.tf, BEARISH, float(l[k]), float(h[k + 2]), c1, c1 + 2,
                           int(bars.open_time[c1]), int(bars.close_time[c1 + 2])))
    out.sort(key=lambda g: (g.c3, g.direction))
    return out


@dataclass(frozen=True)
class GapStatus:
    touched: int        # index of first bar trading into the gap, -1 if none
    invalidated: int    # index of first body close through the gap, -1 if none
    respected: int      # index of first reaction close after touch, -1 if none

    @property
    def state(self) -> str:
        """RESPECTED / DISRESPECTED / PENDING, whichever happened first."""
        r, d = self.respected, self.invalidated
        if d >= 0 and (r < 0 or d <= r):
            return "DISRESPECTED"
        if r >= 0:
            return "RESPECTED"
        return "PENDING"


def gap_status(bars: Bars, gap: FVG, end: Optional[int] = None) -> GapStatus:
    """Status of `gap` using this timeframe's bars after c3 up to `end`."""
    n = len(bars) if end is None else min(end, len(bars))
    a = gap.c3 + 1
    if a >= n:
        return GapStatus(-1, -1, -1)
    lo, hi, cl = bars.low[a:n], bars.high[a:n], bars.close[a:n]
    if gap.direction == BULLISH:
        touch = first_true(lo <= gap.top)
        inval = first_true(cl < gap.bottom)
        react_mask = cl > gap.top
    else:
        touch = first_true(hi >= gap.bottom)
        inval = first_true(cl > gap.top)
        react_mask = cl < gap.bottom
    resp = -1
    if touch >= 0:
        r = first_true(react_mask[touch:])
        resp = touch + r if r >= 0 else -1
    conv = lambda x: a + x if x >= 0 else -1  # noqa: E731
    return GapStatus(conv(touch), conv(inval), conv(resp))


# ---------------------------------------------------------------------------
# Swings
# ---------------------------------------------------------------------------
def swing_highs(high: np.ndarray, strength: int, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    """Indices i with high[i] > the `strength` highs before it and >= the `strength` after it.

    A swing at i is only confirmed once bar i+strength has closed, so with
    truncated arrays the last `strength` bars can never be swings (no lookahead).
    """
    return _swings(np.asarray(high, float), strength, start, end, sign=1.0)


def swing_lows(low: np.ndarray, strength: int, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    return _swings(np.asarray(low, float), strength, start, end, sign=-1.0)


def _swings(x: np.ndarray, k: int, start: int, end: Optional[int], sign: float) -> np.ndarray:
    n = len(x) if end is None else min(end, len(x))
    start = max(0, start)
    seg = sign * x[start:n]
    if len(seg) < 2 * k + 1:
        return np.empty(0, dtype=np.int64)
    if k == 0:
        return np.arange(start, n, dtype=np.int64)
    win = sliding_window_view(seg, 2 * k + 1)
    centre = win[:, k]
    left = win[:, :k].max(axis=1)
    right = win[:, k + 1:].max(axis=1)
    idx = np.flatnonzero((centre > left) & (centre >= right))
    return (idx + k + start).astype(np.int64)


# ---------------------------------------------------------------------------
# Change in State of Delivery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CISD:
    tf: str
    direction: str
    price: float          # opening price of the opposing-close series
    series_start: int
    series_end: int
    confirm_index: int
    created_time: int     # close of the confirming candle

    @property
    def id(self) -> str:
        return f"CISD:{self.tf}:{self.direction}:{self.created_time}"


def find_cisd(bars: Bars, direction: str, touch_index: int, end: Optional[int] = None,
              invalid_index: int = -1) -> Optional[CISD]:
    """CISD after price traded into a key level at bar `touch_index` (spec 2.2).

    Bullish: take the series of consecutive down-close candles that includes the
    touching candle (or, if the touching candle closed up, the down-close series
    ending right before it). Its opening price is the CISD level. The CISD is
    confirmed by the first later candle whose body closes above that price.
    Bearish mirrors this with up-close candles. Returns None if not confirmed
    by `end` or if the key level was invalidated (`invalid_index`) first.
    """
    n = len(bars) if end is None else min(end, len(bars))
    if touch_index < 0 or touch_index >= n:
        return None
    o, c = bars.open, bars.close
    opposing = (lambda j: c[j] < o[j]) if direction == BULLISH else (lambda j: c[j] > o[j])
    j = touch_index
    if not opposing(j):
        j -= 1
        if j < 0 or not opposing(j):
            return None
    s = j
    while s - 1 >= 0 and opposing(s - 1):
        s -= 1
    e = j
    while e + 1 < n and opposing(e + 1):
        e += 1
    price = float(o[s])
    a = e + 1
    if a >= n:
        return None
    closes = c[a:n]
    k = first_true(closes > price) if direction == BULLISH else first_true(closes < price)
    if k < 0:
        return None
    confirm = a + k
    if 0 <= invalid_index <= confirm:
        return None
    return CISD(bars.tf, direction, price, s, e, confirm, int(bars.close_time[confirm]))


# ---------------------------------------------------------------------------
# Rejection blocks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RejectionBlock:
    tf: str
    direction: str
    wick_high: float
    wick_low: float
    index: int
    created_time: int
    source_id: str

    @property
    def ce(self) -> float:
        """Consequent encroachment: 50% of the wick range."""
        return (self.wick_high + self.wick_low) / 2.0

    @property
    def id(self) -> str:
        return f"RB:{self.tf}:{self.direction}:{self.created_time}"


def find_rejection_block(bars: Bars, gap: FVG, end: Optional[int] = None, min_wick: float = 0.0,
                         zone_top: Optional[float] = None, zone_bottom: Optional[float] = None,
                         source_id: Optional[str] = None) -> Optional[RejectionBlock]:
    """Most recent rejection block whose wick traded into `gap` (spec 2.3).

    Bullish: a lower wick (low < min(open, close)) that reached into the bullish
    gap (low <= top) without the body closing below the gap bottom. The wick
    range is [low, min(open, close)]. Bearish mirrors with upper wicks.
    `zone_top/zone_bottom` narrow the zone (e.g. to an intermediate low).
    """
    n = len(bars) if end is None else min(end, len(bars))
    a = gap.c3 + 1
    if a >= n:
        return None
    top = gap.top if zone_top is None else zone_top
    bottom = gap.bottom if zone_bottom is None else zone_bottom
    o, h, l, c = bars.open[a:n], bars.high[a:n], bars.low[a:n], bars.close[a:n]
    if gap.direction == BULLISH:
        body = np.minimum(o, c)
        wick = body - l
        ok = (l <= top) & (c >= gap.bottom) & (wick > 0) & (wick + 1e-12 >= min_wick)
        inval = first_true(c < gap.bottom)
    else:
        body = np.maximum(o, c)
        wick = h - body
        ok = (h >= bottom) & (c <= gap.top) & (wick > 0) & (wick + 1e-12 >= min_wick)
        inval = first_true(c > gap.top)
    if inval >= 0:
        ok[inval:] = False
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    k = int(idx[-1])
    j = a + k
    if gap.direction == BULLISH:
        wh, wl = float(body[k]), float(l[k])
    else:
        wh, wl = float(h[k]), float(body[k])
    return RejectionBlock(bars.tf, gap.direction, wh, wl, j, int(bars.close_time[j]), source_id or gap.id)


# ---------------------------------------------------------------------------
# IFVG confirmation (spec 3.7, verbatim logic)
# ---------------------------------------------------------------------------
def confirm_ifvg(gap: FVG, direction: str, candle_close: float) -> bool:
    if direction == BULLISH:
        return candle_close > gap.upper_boundary
    if direction == BEARISH:
        return candle_close < gap.lower_boundary
    return False

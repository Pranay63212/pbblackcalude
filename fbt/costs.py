"""Instrument specifications, commissions and slippage.

AGENTS.md rule: never lower the default fees or slippage in this file.
tests.py locks the defaults below against minimum values.

Defaults are deliberately conservative estimates of all-in retail costs
(broker commission + CME exchange fee + NFA fee) per contract per side.
Override them per run if your broker is more expensive; do not lower them
to make results look better.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Defaults (do not lower)
# ---------------------------------------------------------------------------
DEFAULT_COMMISSION_PER_SIDE = {
    "ES": 2.50,
    "NQ": 2.50,
    "MES": 0.85,
    "MNQ": 0.85,
}
# Slippage in ticks applied to every market order and every stop order fill.
DEFAULT_MARKET_SLIPPAGE_TICKS = 1
DEFAULT_STOP_SLIPPAGE_TICKS = 1
# Limit orders get no price slippage, but only fill when price trades
# THROUGH the limit (touch is not enough). See execution.py.
DEFAULT_LIMIT_SLIPPAGE_TICKS = 0


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float
    tick_value: float  # dollars per tick per contract

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size

    def round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size


INSTRUMENTS = {
    "ES": InstrumentSpec("ES", 0.25, 12.50),
    "NQ": InstrumentSpec("NQ", 0.25, 5.00),
    "MES": InstrumentSpec("MES", 0.25, 1.25),
    "MNQ": InstrumentSpec("MNQ", 0.25, 0.50),
}


def get_instrument(symbol: str) -> InstrumentSpec:
    root = symbol.upper()
    # Accept contract codes such as ESH4 / NQZ24 / ES.c.0 by matching the root.
    for key in sorted(INSTRUMENTS, key=len, reverse=True):
        if root.startswith(key):
            return INSTRUMENTS[key]
    raise ValueError(f"Unknown instrument '{symbol}'. Known roots: {sorted(INSTRUMENTS)}")


@dataclass(frozen=True)
class CostModel:
    instrument: InstrumentSpec
    commission_per_side: float
    market_slippage_ticks: float = DEFAULT_MARKET_SLIPPAGE_TICKS
    stop_slippage_ticks: float = DEFAULT_STOP_SLIPPAGE_TICKS
    limit_slippage_ticks: float = DEFAULT_LIMIT_SLIPPAGE_TICKS

    @classmethod
    def default(cls, symbol: str) -> "CostModel":
        inst = get_instrument(symbol)
        return cls(instrument=inst, commission_per_side=DEFAULT_COMMISSION_PER_SIDE[inst.symbol])

    def slippage_points(self, order_kind: str) -> float:
        ticks = {
            "MARKET": self.market_slippage_ticks,
            "STOP": self.stop_slippage_ticks,
            "LIMIT": self.limit_slippage_ticks,
        }[order_kind]
        return ticks * self.instrument.tick_size

    def apply_slippage(self, price: float, side: str, order_kind: str) -> float:
        """Move a fill price against the trader. side is 'BUY' or 'SELL'."""
        slip = self.slippage_points(order_kind)
        return price + slip if side == "BUY" else price - slip

    def commission(self, contracts: int) -> float:
        """Commission for one side (entry OR exit)."""
        return self.commission_per_side * contracts

    def round_trip_commission(self, contracts: int) -> float:
        return 2 * self.commission(contracts)

    def estimated_round_trip_cost(self, contracts: int) -> float:
        """Commission + slippage for a market entry and a stop exit (worst case for sizing)."""
        slip_pts = self.slippage_points("MARKET") + self.slippage_points("STOP")
        return self.round_trip_commission(contracts) + slip_pts * self.instrument.point_value * contracts

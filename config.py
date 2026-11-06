"""Configuration for the PB Blake backtester.

EXECUTION_CONFIG is copied from docs/STRATEGY.md section 4.7 (source of truth).
STRATEGY_CONFIG holds the timeframes from the spec plus the operational
parameters needed to turn the spec's prose into code. Every parameter that
is an implementation choice (not stated in the spec) is marked
"OPERATIONAL" and documented in README.md.
"""
from __future__ import annotations

import copy

# ---------------------------------------------------------------------------
# docs/STRATEGY.md 4.7 — verbatim
# ---------------------------------------------------------------------------
EXECUTION_CONFIG = {
    "entry": {
        "method": "IFVG_BODY_CLOSE",
        "alternative_methods": ["IFVG_LIMIT", "CISD_LIMIT"],
    },
    "stop_loss": {
        "default": "MANIPULATION_SWING",
        "alternatives": ["CANDLE_BODY", "FVG_BOUNDARY", "LEG_FVG", "ORDER_BLOCK"],
    },
    "take_profit": {
        "minimum_rr": 1.0,
        "preferred_rr_min": 1.0,
        "preferred_rr_max": 3.0,
        "target_method": "NEAREST_VALID_LIQUIDITY",
    },
    "break_even": {
        "enabled": True,
        "method": "STRUCTURE_BASED",
        "fixed_trigger": None,
    },
    "daily_limits": {
        "max_trades": 2,
        "stop_after_one_win": True,
        "stop_after_one_loss": True,
        "stop_after_two_losses": True,
        "allow_new_A_PLUS_setup": True,
        "max_daily_losses": 2,
    },
    "session": {
        "timezone": "America/Chicago",
        "start": "08:30",
        "end": "10:00",
        "allow_entries_after_end": False,
    },
}

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
STRATEGY_CONFIG = {
    # Spec 1.1 / 2 / 3.1
    "bias_timeframes": {"daily": "1d", "four_hour": "4h", "one_hour": "1h", "fifteen_minute": "15m"},
    "key_level_timeframes": ["3m", "5m", "15m", "30m", "1h", "4h"],
    "ifvg_timeframes": ["1m", "2m", "3m", "4m", "5m"],
    "max_key_levels": 3,  # spec 2.4: "no more than two to three"

    # OPERATIONAL: how many completed bars each timeframe looks back.
    "bias_lookback_bars": {"1d": 60, "4h": 90, "1h": 120, "15m": 96},
    "key_level_lookback_bars": {"3m": 160, "5m": 144, "15m": 96, "30m": 96, "1h": 120, "4h": 90},
    "liquidity_timeframes": ["1h", "4h"],          # draw on liquidity (spec 1.3)
    "target_swing_timeframes": ["1m", "5m", "15m", "1h"],  # low-hanging fruit (spec 4.3)
    "target_fvg_timeframes": ["5m", "15m", "1h"],
    "liquidity_lookback_bars": {"1m": 240, "5m": 144, "15m": 96, "1h": 120, "4h": 90},

    # OPERATIONAL: bias combination. Daily/4H/1H are primary (spec 1.1).
    # Direction = X when at least `bias_min_agreement` primary timeframes say X
    # and (unless allowed) none says the opposite.
    "bias_min_agreement": 2,
    "bias_allow_opposing": False,
    "require_15m_confirmation": False,  # spec: 15M is additional confirmation
    "smt_enabled": True,                # used only if a second instrument is loaded
    "smt_timeframe": "15m",
    "smt_lookback_bars": 16,
    "smt_recent_bars": 4,

    # OPERATIONAL: swing (fractal) strengths: bars required on each side.
    "swing_strength": {"htf": 2, "intermediate": 1, "leg": 2, "target": 2},
    "equal_level_tolerance_ticks": 4,   # relative equal highs / lows
    "min_fvg_ticks": 1,                 # gap must be at least one tick
    "min_wick_ticks": 1,                # rejection-block wick must exist
    "cisd_multi_tf_tolerance_ticks": 4,

    # OPERATIONAL: a key-level touch must be recent to start a setup.
    "max_touch_age_minutes": 60,
    # OPERATIONAL: how far back (1m bars) to search for the manipulation swing.
    "leg_max_bars": 240,

    "stop_buffer_ticks": 1,             # stop placed 1 tick beyond the swing
    "limit_order_expiry_minutes": 30,   # for IFVG_LIMIT / CISD_LIMIT entries
    "alternative_entry_on_poor_rr": False,
    # Spec 4.5 exception: after one loss, a second trade only if the new setup
    # uses a different key level that was touched after the losing exit.
}

# ---------------------------------------------------------------------------
# Risk and account (research / paper only)
# ---------------------------------------------------------------------------
RISK_CONFIG = {
    "initial_equity": 50_000.0,
    "risk_per_trade_pct": 0.5,   # of current equity
    "max_contracts": 10,
    "min_stop_ticks": 4,         # reject unrealistically tight stops
    "max_daily_loss_pct": 2.0,   # stop for the day once realized loss reaches this
    "max_drawdown_halt_pct": 25.0,  # kill switch: halt all new trades
}

BACKTEST_CONFIG = {
    # OPERATIONAL safety: flatten any open position before the CME daily halt.
    # Not part of the strategy's signals; it only exits.
    "flatten_time": "15:50",
}


# ---------------------------------------------------------------------------
# Opt-in variations (the defaults stay exactly as docs/STRATEGY.md says)
# ---------------------------------------------------------------------------
DAILY_LIMIT_PROFILES = {
    # spec 4.5 as written
    "spec": copy.deepcopy(EXECUTION_CONFIG["daily_limits"]),
    # NOT the spec: keep trading after wins and losses. Brakes that remain: max 10 trades,
    # stop after 4 losses, the 2% daily loss limit (RISK_CONFIG), one position at a time,
    # never re-using a key level already traded that day.
    "relaxed": {
        "max_trades": 10,
        "stop_after_one_win": False,
        "stop_after_one_loss": False,
        "stop_after_two_losses": False,
        "allow_new_A_PLUS_setup": True,
        "max_daily_losses": 4,
    },
}

SESSION_PROFILES = {
    "spec": {"start": "08:30", "end": "10:00"},       # spec 4.6: 9:30-11:00 ET
    # NOT the spec: new entries until 14:30 CT (30 min before the ETF close, 20 min before
    # the 14:50 paper-trading flatten). Untested by PB Blake's method.
    "extended": {"start": "08:30", "end": "14:30"},
}


def default_config(daily_limits: str = "spec", session: str = "spec") -> dict:
    """Deep copy of all configuration so callers can modify it safely.

    `daily_limits` / `session` select a profile above; "spec" (the default) is the
    strategy document unchanged.
    """
    if daily_limits not in DAILY_LIMIT_PROFILES:
        raise ValueError(f"daily_limits must be one of {sorted(DAILY_LIMIT_PROFILES)}")
    if session not in SESSION_PROFILES:
        raise ValueError(f"session must be one of {sorted(SESSION_PROFILES)}")
    cfg = {
        "execution": copy.deepcopy(EXECUTION_CONFIG),
        "strategy": copy.deepcopy(STRATEGY_CONFIG),
        "risk": copy.deepcopy(RISK_CONFIG),
        "backtest": copy.deepcopy(BACKTEST_CONFIG),
    }
    cfg["execution"]["daily_limits"] = copy.deepcopy(DAILY_LIMIT_PROFILES[daily_limits])
    cfg["execution"]["session"].update(SESSION_PROFILES[session])
    cfg["profiles"] = {"daily_limits": daily_limits, "session": session}
    return cfg

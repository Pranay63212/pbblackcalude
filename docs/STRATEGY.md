# Strategy specification (PB Blake–style ICT model)

This is the strategy the backtester and any future live code must follow.
All logic must obey AGENTS.md (in particular: no lookahead; only data up to and including the current bar's close).

## 1. Higher-timeframe bias and draw on liquidity

**Purpose:** decide which direction the market is most likely to move before looking for an entry.

### 1.1 Timeframes

| Timeframe | Purpose |
|---|---|
| Daily | Overall market direction |
| 4-hour | Major directional structure |
| 1-hour | Confirm trend and FVG respect |
| 15-minute | Refine bias and identify nearby liquidity |

Daily, 4H and 1H are primary. 15M is extra confirmation, especially near the open.

### 1.2 FVG respect / disrespect

Ask: which bullish and bearish FVGs is price respecting, and which is it disrespecting?

- **Bullish:** price respects bullish FVGs, disrespects bearish FVGs, holds bullish gaps and moves higher, closes through bearish gaps.
- **Bearish:** price respects bearish FVGs, disrespects bullish FVGs, holds bearish gaps and moves lower, closes through bullish gaps.

Coding rules:
- A gap is **respected** when price reacts from it without invalidating its relevant boundary.
- A gap is **disrespected** when price closes through its relevant boundary.
- Use candle **body closes**, not wick penetration, to decide whether a gap was closed through.

### 1.3 Draw on liquidity

- **Bullish:** most obvious swing high above current price; external highs and relative equal highs; mark the most relevant buy-side liquidity. This is the HTF target.
- **Bearish:** most obvious swing low below current price; external lows and relative equal lows; mark the most relevant sell-side liquidity. This is the HTF target.

### 1.4 Optional ES/NQ SMT confirmation

- Bullish SMT: one instrument sweeps a low while the other does not.
- Bearish SMT: one instrument sweeps a high while the other does not.

SMT is extra confluence, **not** a mandatory entry condition.

### 1.5 Bias output

```python
BIAS = {
    "direction": "BULLISH",  # BULLISH, BEARISH, NEUTRAL
    "draw_on_liquidity": None,
    "liquidity_type": None,  # BUY_SIDE or SELL_SIDE
    "daily_bias": None,
    "four_hour_bias": None,
    "one_hour_bias": None,
    "fifteen_minute_confirmation": None,
    "smt_confirmation": None,
}
```

## 2. Valid key levels

**Purpose:** find two to three high-quality areas where price could reverse in line with the HTF bias.

Key-level timeframes: 3m, 5m, 15m, 30m, 1h, 4h.

### 2.1 Fair Value Gaps

**A. Unmitigated FVG** — a gap price has not traded back into. Bullish FVG = potential support; bearish FVG = potential resistance.

**B. Intermediate high/low inside an FVG.**
- Bullish: identify a bullish FVG and the intermediate low inside it. If price has already traded into the FVG, do not keep using the whole original gap; wait for the intermediate low to be swept. The swept level can then be a valid bullish key level.
- Bearish: mirror image — identify the intermediate high inside a bearish FVG; if mitigated, wait for the intermediate high to be swept and use the swept level.

### 2.2 Change in State of Delivery (CISD)

- **Bullish CISD:** price trades down into a bullish key level; forms a down-close candle or series of down-close candles; take the opening price of that candle/series; a candle **body close above** that opening price confirms the bullish CISD.
- **Bearish CISD:** price trades up into a bearish key level; forms an up-close candle or series; take its opening price; a candle **body close below** it confirms the bearish CISD.

The CISD can then be marked as a key level for later reactions.

Record whether a CISD: exists independently; overlaps an FVG; overlaps an intermediate high/low inside an FVG; is confirmed on multiple timeframes. CISD + FVG is stronger confluence.

### 2.3 Rejection blocks

- **Bullish:** a bullish wick that traded into a bullish FVG (or an intermediate low inside one). Mark the wick's range and its 50% level (Consequent Encroachment, CE).
- **Bearish:** a bearish wick that traded into a bearish FVG (or an intermediate high inside one). Mark the wick's range and its 50% level.

CE is useful for more precise reactions.

### 2.4 Key-level selection

Priority:
1. FVGs and intermediate highs/lows inside FVGs
2. CISDs
3. Rejection blocks

No more than two to three primary key levels per setup. Each must align with the HTF narrative: a bullish level must not independently trigger a long when HTF bias is bearish (and vice versa).

### 2.5 Key-level data structure

```python
KEY_LEVEL = {
    "type": None,  # FVG, INTERMEDIATE_HIGH_LOW, CISD, REJECTION_BLOCK
    "direction": None,  # BULLISH or BEARISH
    "timeframe": None,
    "upper_boundary": None,
    "lower_boundary": None,
    "midpoint": None,
    "mitigated": False,
    "swept": False,
    "cisd_confirmed": False,
    "fvg_confluence": False,
    "smt_confluence": False,
    "valid": False,
}
```

## 3. IFVG confirmation (critical entry confirmation)

**Purpose:** confirm order flow has actually changed. Do not enter just because price touches a key level; wait for an Inversion Fair Value Gap (IFVG).

### 3.1 IFVG timeframes

1m, 2m, 3m, 4m, 5m. (30-second is used by PB Blake but excluded until it is backtested separately.)

### 3.2 Manipulation leg

The swing high→swing low (bullish) or swing low→swing high (bearish) leg that trades into the key level.

- Bullish: identify the swing high and the swing low that trades into the bullish key level; analyse the whole leg.
- Bearish: identify the swing low and the swing high that trades into the bearish key level; analyse the whole leg.

Only FVGs **created inside that leg** count. Ignore FVGs outside it.

### 3.3 Highest-timeframe IFVG rule (most important)

Wait for the inversion on the **highest timeframe** that has a qualifying FVG inside the manipulation leg.

| Timeframe | IFVG inside leg? |
|---|---|
| 1M | Yes |
| 2M | Yes |
| 3M | Yes |
| 4M | Yes |
| 5M | No |

Here the highest is 4M, so the bot must wait for the 4-minute inversion and must not enter on the 1-minute inversion.

### 3.4 Bullish IFVG confirmation

1. HTF bias bullish.
2. Price trades into a valid bullish key level.
3. Identify the manipulation leg into that level.
4. Find bearish FVGs formed inside the leg.
5. Find the highest timeframe with a valid bearish FVG.
6. Wait for a candle **body close above** that bearish FVG.
7. The bearish FVG is now inverted → bullish IFVG confirmed.
8. Bullish entry permitted.

### 3.5 Bearish IFVG confirmation

Mirror image: bullish FVGs inside the leg, highest timeframe, candle **body close below** it → bearish IFVG confirmed → bearish entry permitted.

### 3.6 Highest-timeframe selection

```python
IFVG_TIMEFRAMES = ["1m", "2m", "3m", "4m", "5m"]

def find_highest_timeframe_ifvg(manipulation_leg, direction):
    valid_ifvgs = []
    for timeframe in IFVG_TIMEFRAMES:
        gaps = find_fvgs(timeframe=timeframe, within=manipulation_leg)
        for gap in gaps:
            if direction == "BULLISH" and gap.type == "BEARISH":
                valid_ifvgs.append(gap)
            elif direction == "BEARISH" and gap.type == "BULLISH":
                valid_ifvgs.append(gap)
    if not valid_ifvgs:
        return None
    return max(valid_ifvgs, key=lambda gap: timeframe_minutes(gap.timeframe))
```

### 3.7 Entry confirmation

```python
def confirm_ifvg(gap, direction, candle):
    if direction == "BULLISH":
        return candle.close > gap.upper_boundary
    if direction == "BEARISH":
        return candle.close < gap.lower_boundary
    return False
```

## 4. Execution and risk management

A trade is permitted only when **all** hold:
- HTF bias established.
- Valid key level identified.
- Price has traded into that key level.
- Manipulation leg identified.
- Highest-timeframe IFVG inside that leg has inverted.
- Entry is within the allowed session.
- Risk/reward is acceptable.

### 4.1 Entry

Default: enter at the **body close of the IFVG confirming candle** (long for bullish, short for bearish), i.e. after that candle closes.

Alternatives when the confirming close is too far and R:R is poor (configurable, not mandatory): limit order at the IFVG, or limit order at the CISD. Do not chase price after a large move.

### 4.2 Stop loss

| Stop location | Description |
|---|---|
| Swing low/high | Main (default) |
| Candle body | Alternative |
| FVG boundary | Alternative |
| FVG created during manipulation leg | Alternative |
| Order block | Alternative |

Default: LONG stop below the relevant swing low of the manipulation leg; SHORT stop above the relevant swing high. Alternatives are configurable for backtesting.

### 4.3 Take profit

Generally 1R–3R (R = initial risk). Target selection is "low-hanging fruit":
- Bullish: nearest obvious swing high, internal high, external high, nearby buy-side liquidity, relevant unfilled FVG.
- Bearish: nearest obvious swing low, internal low, external low, nearby sell-side liquidity, relevant unfilled FVG.

Do not automatically target 3R if the nearest logical liquidity is 1R away. Prefer the closest valid liquidity objective that still gives acceptable R:R (minimum 1.0).

### 4.4 Break-even

Configurable only. PB Blake sometimes moves to break-even after price reaches a relevant internal/external swing (manipulation leg, external swing, stacked liquidity), but gives no fixed formula. Do not invent a universal trigger and attribute it to him.

### 4.5 Daily limits (enforced)

1. One winning trade → stop for the day.
2. One losing trade → generally stop for the day.
3. Exception: a new key level forms or price stop-runs the previous setup and another A+ setup develops.
4. Two losing trades → stop for the day.
5. No repeated re-entry after losses.
6. Never add to losing positions.
7. Never move stops farther away.

### 4.6 Trading hours

9:30–11:00 ET = **8:30–10:00 America/Chicago**. Start looking at the open, prioritise the opening window, no new positions after the window ends. (PB Blake's occasional discretionary extension is **not** implemented; the automated version enforces the strict window.)

### 4.7 Execution configuration

```python
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
```

## 5. Decision flow (exact order)

```python
def trading_strategy():
    # STEP 1: HIGHER TIMEFRAME BIAS
    bias = determine_higher_timeframe_bias(timeframes=["1d", "4h", "1h", "15m"])
    if bias.direction == "NEUTRAL":
        return "NO TRADE"
    draw_on_liquidity = identify_draw_on_liquidity(direction=bias.direction)
    if draw_on_liquidity is None:
        return "NO TRADE"

    # STEP 2: IDENTIFY KEY LEVELS
    key_levels = identify_key_levels(
        timeframes=["3m", "5m", "15m", "30m", "1h", "4h"],
        direction=bias.direction,
        types=["FVG", "INTERMEDIATE_HIGH_LOW", "CISD", "REJECTION_BLOCK"],
        max_levels=3,
    )
    if not key_levels:
        return "WAIT"

    # STEP 3: WAIT FOR PRICE TO HIT KEY LEVEL
    active_level = wait_for_key_level_reaction(key_levels=key_levels, direction=bias.direction)
    if active_level is None:
        return "WAIT"

    # STEP 4: IDENTIFY MANIPULATION LEG
    manipulation_leg = identify_manipulation_leg(key_level=active_level, direction=bias.direction)
    if manipulation_leg is None:
        return "WAIT"

    # STEP 5: FIND HIGHEST TIMEFRAME IFVG
    highest_ifvg = find_highest_timeframe_ifvg(manipulation_leg=manipulation_leg, direction=bias.direction)
    if highest_ifvg is None:
        return "WAIT"

    # STEP 6: CONFIRM INVERSION
    if not wait_for_ifvg_body_close(gap=highest_ifvg, direction=bias.direction):
        return "WAIT"

    # STEP 7: ENTRY AND STOP
    entry = get_ifvg_confirmation_close()
    stop_loss = calculate_swing_based_stop(manipulation_leg=manipulation_leg, direction=bias.direction)

    # STEP 8: TARGET
    target = identify_nearest_valid_liquidity(direction=bias.direction, draw_on_liquidity=draw_on_liquidity)
    risk_reward = calculate_risk_reward(entry=entry, stop=stop_loss, target=target)
    if risk_reward < 1.0:
        return "NO TRADE"

    # STEP 9: DAILY RISK RESTRICTIONS
    if daily_trade_limit_reached():
        return "NO TRADE"
    if daily_loss_limit_reached():
        return "NO TRADE"

    # STEP 10: EXECUTE (must go through the risk manager per AGENTS.md)
    if bias.direction == "BULLISH":
        return execute_long(entry=entry, stop_loss=stop_loss, take_profit=target)
    elif bias.direction == "BEARISH":
        return execute_short(entry=entry, stop_loss=stop_loss, take_profit=target)
    return "NO TRADE"
```

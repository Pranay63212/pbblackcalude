# fbt: futures backtester for ES/NQ 1-minute bars

claude pb black strat

An event-driven research backtester for the PB Blake ICT-style strategy described in
[`docs/STRATEGY.md`](docs/STRATEGY.md), which is the source of truth. Rules for
contributors and agents are in [`AGENTS.md`](AGENTS.md).

> **Research and paper simulation only.** There is no brokerage connection and no live
> order path. Nothing here is evidence that the strategy is profitable. Treat any result
> as hypothetical until it holds up **out-of-sample** with statistical significance and
> parameters that were never tuned on that sample.

## Quick start

```bash
pip install -r requirements.txt
python tests.py
python -m fbt.backtest --data data/ES_1m.csv.gz --symbol ES --oos-start 2020-01-01
```

Optional SMT confirmation (spec 1.4) with a second instrument:

```bash
python -m fbt.backtest --data data/ES_1m.csv.gz --symbol ES --smt-data data/NQ_1m.csv.gz --smt-symbol NQ
```

Other flags: `--start/--end` (CT dates), `--tz` (timezone of naive timestamps, default UTC),
`--data-symbol` (selects one series from a multi-symbol file), `--all-decisions`, `--out`.

## Data

Supply **one continuous 1-minute series per file** (`.csv`, `.csv.gz`, or `.dbn.zst` with
`pip install databento`). The required columns are a timestamp (`timestamp`, `ts_event`,
`datetime`, ...) giving the bar **open** time, plus `open, high, low, close, volume`.
`data/` and `results/` are git-ignored. Never commit data.

Invalid data stops the run with a `DataError` (exit code 2) and no results are written.
The loader never repairs data. It raises on:
- a missing or empty file, or an unsupported format
- missing columns, NaN or non-numeric values, or non-positive prices
- high < low, or open/close outside the bar's range
- negative volume
- duplicate or unaligned timestamps, or bars less than one minute apart
- several contracts mixed in one file without `--data-symbol`
- a `--start`/`--end` range the file doesn't cover

Gaps such as halts and holidays are reported, not filled.

**Continuous contracts:** roll gaps in unadjusted continuous data can create artificial
FVGs. Use back-adjusted data, or accept that roll days are contaminated.

**Scale:** 15 years is about 5.2M one-minute bars. Building every timeframe takes seconds.
The full pipeline costs about 2 ms per in-session evaluation, so a 15-year run takes roughly
5–15 minutes. Plan for about 2 GB of RAM.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | `EXECUTION_CONFIG` (spec 4.7, verbatim), strategy, risk and backtest parameters |
| `fbt/data.py` | Loading, validation, session-aligned resampling to 1m–1d, `MarketView` |
| `fbt/indicators.py` | FVG, gap status (respect/disrespect), swings, CISD, rejection blocks, `confirm_ifvg` |
| `fbt/structure.py` | Bias, SMT, draw on liquidity, key levels, manipulation leg, highest-TF IFVG, targets |
| `fbt/strategy.py` | The section 5 decision flow → `Decision` / `Signal` |
| `fbt/risk.py` | Session gate, daily limits, sizing, drawdown halt, single-use order approvals |
| `fbt/execution.py` | `PaperBroker` fill simulation (paper only) |
| `fbt/costs.py` | Instrument specs, commissions, slippage (defaults must never be lowered) |
| `fbt/backtest.py` | Event loop, statistics, reports, CLI |

### Lookahead protection

- Each bar on each timeframe carries a `close_time`. A bar exists for the strategy only
  once `close_time <= now`.
- The strategy receives only a `MarketView`: read-only numpy views truncated to completed
  bars. Higher-timeframe bars are aligned to the 17:00 CT Globex open, so a daily bar is
  visible only after its session ends.
- Swings are fractals and are confirmed only after `strength` later bars have closed.
- A signal on bar *t*'s close is filled at bar *t+1*'s open.
- `tests.py` checks this from several directions:
  - views never expose unfinished bars
  - decisions and full analysis snapshots are identical on full data, on data truncated at
    *t*, and on data whose future is replaced with garbage
  - a deliberately leaky `MarketView` makes these tests fail

### Costs and fills (conservative defaults)

- **Commission per contract per side:** ES and NQ $2.50, MES and MNQ $0.85.
- **Slippage:** 1 tick on every market order and stop fill. Limit orders get no slippage,
  but fill only on a trade-through (a touch is not enough).
- **Gaps:** a stop gapped through fills at the open, less slippage.
- **Ambiguous bars:** if stop and target fall in the same bar, the stop is assumed hit first.
- **Break-even:** a break-even move only tightens the stop and takes effect from the next bar.
- Net P&L = gross P&L (computed from slipped fill prices) − commissions.

### Risk controls (`RISK_CONFIG`, `EXECUTION_CONFIG`)

- Entries only when the signal bar closes within 08:30–10:00 America/Chicago.
  `allow_entries_after_end` cannot be enabled, and there is no 11:00 ET extension.
- One position at a time. Never adds to a position, and stops can never be widened.
- Daily limits (spec 4.5):
  - stop after one win
  - after one loss, continue only for a new A+ setup
  - stop after two losses, and allow at most two trades per day
- Daily realized-loss limit of 2%, and a 25% drawdown kill switch.
- Sizing risks 0.5% of equity per trade, including estimated round-trip costs, capped at 10
  contracts. With the default $50k account, an ES stop wider than about 4.5 points sizes to
  zero contracts and the trade is rejected. For small accounts use `--symbol MES`/`MNQ`
  with micro-contract data, or change `RISK_CONFIG`.
- The broker rejects any order that lacks a single-use approval from the risk manager.

## Outputs (`results/<symbol>_<timestamp>/`)

| File | Contents |
|---|---|
| `trades.csv` | One row per trade: entry and exit details, costs, R multiple, MFE/MAE, and the full setup (bias per TF, key level, IFVG, manipulation leg, draw on liquidity) |
| `decisions.csv` | Every non-WAIT decision and its reason (`--all-decisions` logs every evaluation) |
| `risk_rejections.csv` | Signals the risk manager refused, and why |
| `equity.csv` | Closed-trade equity curve and drawdown |
| `summary.json` | Statistics for all, in-sample and out-of-sample trades, a data-quality report, and the config snapshot |

**Statistics:**
- total trades, win rate, profit factor, net P&L, expectancy
- average and median R, maximum drawdown ($ and %), longest losing streak
- commission and slippage totals
- a t-test of mean R against zero, and a binomial test of the win rate against 50%

The summary states plainly when the sample is too small or the result isn't significant.

## How the spec maps to code, and the operational choices

The spec is prose. Turning it into code required choices that it doesn't state. All are
listed here, and the numeric ones live in `STRATEGY_CONFIG`. None adds a trading signal;
each defines how an existing concept is measured.

1. **FVG**: the three-candle gap (`low[c3] > high[c1]` for bullish), existing from c3's close. Minimum size is one tick.
2. **Respect / disrespect (1.2)**: *disrespect* is a body close through the relevant boundary; wicks don't count. *Respect* means price traded into the gap and a later candle closed back beyond it in the gap's direction, without disrespecting it first.
3. **Per-timeframe bias**: count respected and disrespected gaps over a lookback window (1d: 60, 4h: 90, 1h: 120, 15m: 96 bars). Bullish points = bullish respected + bearish disrespected. A tie means NEUTRAL.
4. **Overall bias**: Daily, 4H and 1H are primary. A direction needs at least 2 of the 3 to agree and none opposing. The 15M result is recorded as confirmation and isn't required (configurable). SMT is recorded as confluence and is never required.
5. **SMT (1.4)**: on 15m bars, compare the last 4 bars with the 12 before them. SMT means exactly one instrument traded beyond its prior extreme.
6. **Draw on liquidity (1.3)**: unswept 1h/4h fractal swings (strength 2) beyond the current price. Relative equal highs or lows (within 4 ticks) win. Otherwise take the nearest swing on the highest timeframe that has one.
7. **Key levels (2)**: built on 3m–4h from gaps that haven't been invalidated.
   - Touches are measured on completed 1m bars.
   - A touch starts a setup only if it happened within the last 60 minutes. An FVG touched longer ago counts as mitigated.
   - For a mitigated gap, the level becomes the latest swing (strength 1) inside it. It is swept by a strict 1m trade-through.
8. **CISD (2.2)**: take the opposing-close candle series that includes the candle that traded into the gap (or ends just before it). The level is that series' opening price. A later body close beyond it confirms the CISD.
   - Zone: from the series' extreme to its open.
   - It becomes invalid on a body close beyond the extreme.
   - The log records FVG overlap, intermediate overlap and multi-timeframe confirmation (within 4 ticks).
9. **Rejection block (2.3)**: the latest wick, at least one tick long, that reached into a live gap. The zone is the wick's range and CE is its midpoint. It becomes invalid on a body close beyond the wick.
10. **Selection (2.4)**: rank by priority (FVG/intermediate, then CISD, then rejection block), then distance from price, then higher timeframe. Keep at most 3. Only levels in the bias direction are built, and a level used for a trade isn't reused that day.
11. **Manipulation leg (3.2)**: it ends at the most extreme 1m bar since the touch. It starts at the most recent 1m fractal (strength 2), within the previous 240 bars, that is the extreme of everything from it to the end and lies beyond the key level.
12. **FVGs inside the leg (3.2–3.6)**:
    - Candles must fall within the timeframe buckets that contain the leg's start and end.
    - Gaps already inverted before the leg ended are dropped.
    - The spec's `max()` picks the highest timeframe. Among several gaps there it returns the first, which is the earliest-created gap.
13. **Confirmation (3.4–3.7)**: a candle on the IFVG's own timeframe must close exactly now, and it must be the first body close through the gap. An inversion that happened earlier is never chased.
14. **Entry (4.1)**:
    - `IFVG_BODY_CLOSE` (the default) is a market order at the next 1m open.
    - `IFVG_LIMIT` places a limit at the inverted gap's boundary. `CISD_LIMIT` places a limit at the leg's CISD, never worse than the confirming close.
    - Limits expire after 30 minutes or at the session end. They are cancelled if the target trades first.
    - Alternatives are used only if chosen, or if `alternative_entry_on_poor_rr` is True.
15. **Stops (4.2)**:
    - The default is 1 tick beyond the manipulation leg's extreme.
    - Alternatives are the confirming candle's body, the key level's far boundary, the inverted gap's far boundary, and an order block.
    - The order block is the last opposing-close candle on the IFVG timeframe before the inversion. The spec doesn't define it, so this is the conventional definition.
16. **Targets (4.3)**: candidates are:
    - unswept swings on 1m/5m/15m/1h (strength 2)
    - the manipulation leg's swing
    - unfilled FVG edges on 5m/15m/1h
    - the draw on liquidity

    The nearest candidate with R:R ≥ 1.0 wins. 3R is never targeted automatically, and if nothing reaches 1R there is no trade.
17. **Break-even (4.4)**: the configured `STRUCTURE_BASED` method moves the stop to entry once price reaches the manipulation leg's starting swing, if that swing lies between entry and target. **This is one configurable rule, not a formula attributed to PB Blake.** `FIXED_R` and disabling it are also supported.
18. **The A+ exception (4.5)**: after one loss, a second trade must use a different key level that was touched after the losing exit. A break-even scratch counts as a loss because fees make it negative.
19. **Session (4.6)**: a 1m bar whose close falls within 08:30–10:00 CT can trigger an entry. The 30-second chart isn't used.
20. **Safety flatten**: open positions are closed at 15:50 CT, before the CME halt. This only ever exits a position.

## Live scanner (alerts only, never trades)

`fbt/scanner.py` watches ES, NQ, YM and GC (alerts show full-size and micro sizing) and
runs the **same strategy code** as the backtester, a few seconds after each 1-minute
candle closes. Signals are reported only between 08:30 and 10:00 CT.

| Alert | Meaning |
|---|---|
| Pre-session (08:25 CT) | Bias per timeframe, draw on liquidity, current key levels |
| Setup forming | Price traded into a key level; waiting for the highest-timeframe IFVG to invert |
| **Signal** | Every rule met: entry, stop, target, R:R, the setup's details, and contracts for your risk settings |
| Stale / data error | Data stopped or is invalid; the symbol is not scanned until data is good again |

Checking more often than once a minute adds nothing: the rules act only on completed
candle body closes, and the smallest candle is 1 minute.

The scanner cannot see your fills, so the daily limits (spec 4.5) are printed as a
reminder rather than enforced. It contains no order code (a test checks this).

```bash
python -m fbt.scanner live --feed alpaca                         # free: SPY QQQ DIA GLD (stand-ins)
python -m fbt.scanner live --feed ibkr --symbols ES NQ YM GC      # TWS paper on port 7497
python -m fbt.scanner live --feed ibkr --port 4002               # IB Gateway paper
python -m fbt.scanner live --feed csv --folder data/live         # <SYMBOL>.csv files kept up to date
python -m fbt.scanner replay --data data/ES_1m.csv.gz --symbol ES --date 2024-03-12
```

- **Interactive Brokers:** run TWS or IB Gateway logged in to the paper account. Enable the
  API under *Global Configuration → API → Settings*. The account needs real-time market data
  for CME, CBOT and COMEX; without it IB sends delayed data or none. The connection is
  read-only. On first start it loads 20 days of 1-minute bars plus a year of daily bars,
  which takes a few minutes. The front-month contract is used, so expect oddities around rolls.
- **Alpaca (free):** set the environment variables `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
  to your paper-account keys (Windows: `setx APCA_API_KEY_ID "..."`, then open a new window).
  The free IEX feed covers one exchange, so quiet minutes can be missing; the scanner waits
  5 minutes before flagging stale data. SPY/QQQ/DIA/GLD track ES/NQ/YM/GC but are stand-ins,
  sized in shares (max 2,000, minimum stop 10 cents).
- **Phone alerts:** set the environment variable `FBT_ALERT_WEBHOOK` to an ntfy.sh topic URL
  or a Discord webhook. Never put it in a file in this repository.
- Alerts are also logged to `results/alerts.csv`.
- **Unattended use:** start it once and leave it running; it is idle at night and on weekends
  and scans 08:25-10:00 CT each weekday. Network errors, timeouts and failed history loads
  are reported and retried every minute instead of stopping the program. Add
  `--push-summary` to receive the 08:25 summary on your phone as a daily "still running"
  check. Keep the computer on, logged in, plugged in and set never to sleep, and set Windows
  Update "active hours" to cover the morning so it doesn't restart then.
- `replay` rehearses a past day through the same code, to check alerts against a chart.

## Automatic paper trading (Alpaca, paper only)

```bash
python -m fbt.scanner live --feed alpaca --trade --push-summary
```

With `--trade`, each signal on SPY/QQQ/DIA/GLD becomes one **bracket order** in your
Alpaca **paper** account: a market entry plus a stop-loss and a take-profit, both held at
Alpaca (good-till-cancelled), so the stop stays in place even if this computer turns off.

- Every order needs a single-use approval from the RiskManager: entries only 08:30-10:00 CT,
  sizing at 0.5% of the paper account's equity, minimum stop 10 cents, and the spec 4.5
  daily limits applied across all four symbols (stop after a win; after a loss only a new A+
  setup on a different, newly touched level; at most two trades).
- No entry while any position or open order exists in a traded symbol.
- Open trades are closed at 14:50 CT (`--flatten-time`), before the 15:00 CT ETF close.
- Order placed, fill and exit (with estimated P&L and R) are pushed to your phone; trades
  go to `results/paper_trades.csv`; the day's state is kept in `results/trader_state.json`
  so a restart the same day remembers it.
- **Paper only:** orders go to `paper-api.alpaca.markets`. `--live` needs
  `LIVE_TRADING_CONFIRMED=yes` and is still refused: live trading is not implemented.
- Order code lives only in `fbt/trader.py`; the scanner module itself never places orders.
- Tested against a stand-in for Alpaca's API, not yet against Alpaca's servers.

## Opt-in variations (not the strategy document)

Both `fbt.backtest` and `fbt.scanner` accept:

| Flag | Default (`spec`) | Alternative |
|---|---|---|
| `--daily-limits` | spec 4.5: stop after a win; after a loss only a new A+ setup; max 2 trades | `relaxed`: keep trading after wins and losses, up to 10 trades; stops after 4 losses or the 2% daily loss limit; one position at a time; a key level is never traded twice in a day |
| `--session` | spec 4.6: new entries 08:30-10:00 CT | `extended`: new entries until 14:30 CT (open trades are still flattened at 14:50 CT for ETFs, 15:50 CT in backtests) |

Without flags, everything follows docs/STRATEGY.md. Each run records its profiles in
`summary.json`.

## News filter, market-agreement filter, morning mark-out

```bash
python -m fbt.econ --out data/econ_calendar.csv          # this week's events + all Fed meetings
python -m fbt.scanner live --feed alpaca --trade --push-summary --news-filter --pair-filter
python -m fbt.backtest ... --news-filter --news-file data/econ_calendar.csv --pair-filter --smt-data ...
```

- **Morning mark-out (08:00 CT, `--premarket-time`):** for each symbol, price, bias per
  timeframe, paired-market bias, draw on liquidity, key levels, and the classic liquidity
  levels: yesterday's high/low, the overnight (since 17:00 CT) high/low and last week's
  high/low, each with its distance and whether it was already TAKEN. A separate message
  lists today's high-impact US news. With `--push-summary` both go to your phone.
- **`--news-filter` (not in the spec):** no new entries from 15 minutes before to 15 minutes
  after a high-impact US release (Forex Factory weekly feed + Federal Reserve FOMC dates).
  Most US data is released at 07:30 CT, before the 08:30 open, so the filter mainly affects
  09:00 CT releases and the 13:00 CT Fed decision (extended session). If the calendar cannot
  be downloaded for today, new entries are refused rather than assuming a quiet day. The
  live scanner saves each week's calendar to `data/econ_calendar.csv`, building a history
  for future backtests; until then, past news is known only for Fed meetings.
- **`--pair-filter` (not in the spec):** SPY trades only when QQQ has the same bias and vice
  versa (ES/NQ in backtests). If the paired market's data is missing, no trade. DIA, GLD,
  YM and GC have no pair and are unaffected.
- On ~29 days of Yahoo data neither filter could be judged: NQ's bias was neutral all
  month (daily bullish vs 4H/1H bearish), so the pair filter blocked every ES trade, and no
  trade fell inside a news window.

## Free real-data backtest (Yahoo Finance)

```bash
pip install yfinance
python -m fbt.yahoo --symbols ES NQ YM GC --out data/yahoo
python -m fbt.backtest --data data/yahoo/ES_1m.csv.gz --daily-data data/yahoo/ES_daily.csv.gz --symbol MES
```

Yahoo provides about 30 days of 1-minute bars plus daily bars, which supply the 60-day
daily-bias history. The series are continuous front-month futures, not back-adjusted.
Prices are rounded to 2 decimals to remove Yahoo's float32 storage noise (exact for every
supported tick). One year of daily bars is fetched because older Yahoo YM history contains
malformed candles, which the loader rejects. About two weeks of each month go to warm-up, so
expect only a handful of trades: useful for checking behaviour, not for judging the strategy.

## Known limitations

- The bar-based fill model can't see order-book queue position or intrabar order. The
  same-bar stop-first assumption is pessimistic.
- Discretionary parts of the method (what counts as "obvious" liquidity, A+ quality, and
  the discretionary after-11:00 trading) are approximated or excluded, as listed above.
- Optimizing the numbers in `STRATEGY_CONFIG` on the same data you evaluate on will
  overfit. Use `--oos-start` and keep the out-of-sample period untouched.

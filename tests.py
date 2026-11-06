"""Test suite. Run with:  python tests.py

AGENTS.md: all tests must pass before every commit; never delete or weaken a test.
Covers data validation, resampling, indicators, structure, strategy flow,
risk manager, execution/costs, statistics and lookahead protection.
"""
from __future__ import annotations

import copy
import gzip
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import default_config  # noqa: E402
from fbt import costs as costs_mod  # noqa: E402
from fbt import structure as structure_mod  # noqa: E402
from fbt.backtest import Backtester, compute_stats, main as cli_main, trades_frame  # noqa: E402
from fbt.costs import CostModel, get_instrument  # noqa: E402
from fbt.data import (NS_PER_MIN, Bars, DataError, MultiTimeframeData, load_ohlcv,  # noqa: E402
                      ns_to_chicago, validate_ohlcv)
from fbt.execution import PaperBroker, Trade  # noqa: E402
from fbt.indicators import (BEARISH, BULLISH, NEUTRAL, FVG, confirm_ifvg, find_cisd, find_fvgs,  # noqa: E402
                            find_rejection_block, gap_status, swing_highs, swing_lows)
from fbt.risk import Approval, Rejection, RiskManager, RiskViolation  # noqa: E402
from fbt.strategy import PBBlakeStrategy, Signal  # noqa: E402
from fbt.structure import (ManipulationLeg, calculate_risk_reward,  # noqa: E402
                           determine_higher_timeframe_bias, find_highest_timeframe_ifvg,
                           identify_draw_on_liquidity, identify_key_levels,
                           select_target, smt_divergence, timeframe_bias, TouchTracker)

CT = "America/Chicago"
ES = get_instrument("ES")


# ===========================================================================
# helpers
# ===========================================================================
def ct(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=CT)


def ns(s: str) -> int:
    return int(ct(s).tz_convert("UTC").value)


def make_bars(ohlc, tf="1m", start="2024-03-12 08:00"):
    """Bars from a list of (o, h, l, c) with consecutive timestamps."""
    arr = np.asarray(ohlc, float)
    m = {"1m": 1, "2m": 2, "3m": 3, "4m": 4, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}[tf]
    t0 = ns(start)
    ot = t0 + np.arange(len(arr), dtype=np.int64) * m * NS_PER_MIN
    return Bars(tf, ot, ot + m * NS_PER_MIN, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], np.ones(len(arr)))


class FakeView:
    """Minimal MarketView stand-in for unit tests of structure functions."""

    def __init__(self, bars: dict, now: int):
        self._b = bars
        self.now = now

    def bars(self, tf):
        return self._b[tf]

    def count(self, tf):
        return len(self._b[tf])

    @property
    def last_close(self):
        return float(self._b["1m"].close[-1])


def random_walk_df(days=30, seed=7, start="2023-01-02", step=0.5):
    """Synthetic 23h/day 1-minute random walk, ES-like tick grid. Test data only."""
    idx = pd.date_range(start, periods=days, freq="B")
    parts = []
    for d in idx:
        s = (d - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(CT)
        parts.append(pd.date_range(s, periods=23 * 60, freq="1min"))
    ts = parts[0].append(parts[1:]).tz_convert("UTC")
    rng = np.random.default_rng(seed)
    n = len(ts)
    c = 4000 + np.cumsum(rng.standard_t(3, n) * step)
    c = np.round(c * 4) / 4
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + np.round(np.abs(rng.normal(0, .4, n)) * 4) / 4
    l = np.minimum(o, c) - np.round(np.abs(rng.normal(0, .4, n)) * 4) / 4
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 100.0}, index=ts)


def bullish_scenario_df():
    """Hand-built bullish setup on 2024-03-12 (CT):
    overnight spike to 130 (buy-side liquidity), rally 07:15-07:30 leaving bullish
    FVGs, swing high ~110.25, sharp drop 08:30-08:37 into the gaps (leaving bearish
    FVGs inside the manipulation leg), base, then a rally that inverts them and runs to 135.
    """
    rows = []
    t = [ct("2024-03-11 17:00")]
    p = [100.0]

    def bar(o, c, wick=0.25):
        rows.append((t[0], o, max(o, c) + wick, min(o, c) - wick, c))
        t[0] += pd.Timedelta(minutes=1)

    def flat(until, price):
        while t[0] < ct(until):
            bar(price, price)

    def ramp(end, step):
        while (step > 0 and p[0] < end) or (step < 0 and p[0] > end):
            q = round(p[0] + step, 2)
            bar(p[0], q)
            p[0] = q

    flat("2024-03-11 23:00", 100.0)
    rows.append((t[0], 100.0, 130.0, 100.0, 100.0))
    t[0] += pd.Timedelta(minutes=1)
    flat("2024-03-12 07:15", 100.0)
    ramp(106, 0.5)
    flat("2024-03-12 07:45", 106.0)
    ramp(110, 0.25)
    flat("2024-03-12 08:30", 110.0)
    ramp(103.5, -1.0)
    flat("2024-03-12 08:50", 103.5)
    ramp(135, 1.0)
    flat("2024-03-12 11:00", 135.0)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).set_index("ts")
    df.index = df.index.tz_convert("UTC")
    df["volume"] = 10.0
    return df


class FixedBiasStrategy(PBBlakeStrategy):
    """Strategy with the bias step pinned (to test steps 2-10 in isolation)."""

    def __init__(self, cfg, inst, direction):
        super().__init__(cfg, inst)
        self.direction = direction

    def _bias(self, view, smt_view):
        d = self.direction
        return {"direction": d, "draw_on_liquidity": None, "liquidity_type": None, "daily_bias": d,
                "four_hour_bias": d, "one_hour_bias": d, "fifteen_minute_confirmation": None,
                "smt_confirmation": None}


def scenario_config():
    cfg = default_config()
    cfg["risk"]["risk_per_trade_pct"] = 2.0
    return cfg


def make_signal(direction=BULLISH, entry=100.0, stop=98.0, target=104.0, t=None, level_id="L1", touch=None):
    t = ns("2024-03-12 09:00") if t is None else t
    return Signal(direction, "IFVG_BODY_CLOSE", entry, stop, target, "TEST", abs(target - entry) / abs(entry - stop),
                  t, "MANIPULATION_SWING", None, {}, {}, {}, {}, None, level_id,
                  touch if touch is not None else t - 5 * NS_PER_MIN)


# ===========================================================================
# Data
# ===========================================================================
class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text, gz=False):
        path = os.path.join(self.tmp.name, name)
        if gz:
            with gzip.open(path, "wt") as fh:
                fh.write(text)
        else:
            with open(path, "w") as fh:
                fh.write(text)
        return path

    GOOD = ("timestamp,open,high,low,close,volume\n"
            "2024-03-12T14:30:00Z,100,101,99,100.5,10\n"
            "2024-03-12T14:31:00Z,100.5,101,100,100.75,12\n")

    def test_good_csv_and_gzip_load(self):
        df = load_ohlcv(self.write("a.csv", self.GOOD))
        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index.tz), "UTC")
        df2 = load_ohlcv(self.write("a.csv.gz", self.GOOD, gz=True))
        self.assertTrue(df.equals(df2))

    def test_naive_timestamps_use_given_timezone(self):
        text = "timestamp,open,high,low,close,volume\n2024-03-12 08:30:00,1,2,0.5,1.5,1\n"
        df = load_ohlcv(self.write("n.csv", text), tz=CT)
        self.assertEqual(df.index[0], pd.Timestamp("2024-03-12 13:30", tz="UTC"))

    def test_missing_file(self):
        with self.assertRaisesRegex(DataError, "not found"):
            load_ohlcv(os.path.join(self.tmp.name, "nope.csv"))

    def test_empty_file(self):
        with self.assertRaisesRegex(DataError, "empty"):
            load_ohlcv(self.write("e.csv", ""))

    def test_unsupported_format(self):
        with self.assertRaisesRegex(DataError, "Unsupported"):
            load_ohlcv(self.write("x.parquet", "abc"))

    def test_missing_column(self):
        with self.assertRaisesRegex(DataError, "missing required columns"):
            load_ohlcv(self.write("m.csv", "timestamp,open,high,low,close\n2024-03-12T14:30:00Z,1,2,0.5,1\n"))

    def test_no_timestamp_column(self):
        with self.assertRaisesRegex(DataError, "timestamp"):
            load_ohlcv(self.write("t.csv", "foo,open,high,low,close,volume\n1,1,2,0.5,1,1\n"))

    def test_nan_values(self):
        text = self.GOOD + "2024-03-12T14:32:00Z,100,,99,100,1\n"
        with self.assertRaisesRegex(DataError, "missing/non-numeric"):
            load_ohlcv(self.write("nan.csv", text))

    def test_high_below_low(self):
        text = "timestamp,open,high,low,close,volume\n2024-03-12T14:30:00Z,100,99,101,100,1\n"
        with self.assertRaisesRegex(DataError, "high < low"):
            load_ohlcv(self.write("hl.csv", text))

    def test_close_outside_range(self):
        text = "timestamp,open,high,low,close,volume\n2024-03-12T14:30:00Z,100,101,99,102,1\n"
        with self.assertRaisesRegex(DataError, "high below open/close"):
            load_ohlcv(self.write("c.csv", text))

    def test_non_positive_and_negative_volume(self):
        t1 = "timestamp,open,high,low,close,volume\n2024-03-12T14:30:00Z,0,1,0,1,1\n"
        with self.assertRaisesRegex(DataError, "non-positive"):
            load_ohlcv(self.write("z.csv", t1))
        t2 = "timestamp,open,high,low,close,volume\n2024-03-12T14:30:00Z,1,2,0.5,1,-5\n"
        with self.assertRaisesRegex(DataError, "negative volume"):
            load_ohlcv(self.write("v.csv", t2))

    def test_duplicate_timestamps(self):
        text = self.GOOD + "2024-03-12T14:31:00Z,100.5,101,100,100.75,12\n"
        with self.assertRaisesRegex(DataError, "duplicate"):
            load_ohlcv(self.write("d.csv", text))

    def test_not_minute_aligned(self):
        text = "timestamp,open,high,low,close,volume\n2024-03-12T14:30:30Z,1,2,0.5,1,1\n"
        with self.assertRaisesRegex(DataError, "whole minutes"):
            load_ohlcv(self.write("s.csv", text))

    def test_multiple_symbols_require_selection(self):
        text = ("timestamp,symbol,open,high,low,close,volume\n"
                "2024-03-12T14:30:00Z,ESH4,1,2,0.5,1,1\n2024-03-12T14:30:00Z,ESM4,1,2,0.5,1,1\n")
        path = self.write("sym.csv", text)
        with self.assertRaisesRegex(DataError, "symbols"):
            load_ohlcv(path)
        self.assertEqual(len(load_ohlcv(path, symbol="ESH4")), 1)
        with self.assertRaisesRegex(DataError, "not present"):
            load_ohlcv(path, symbol="NQH4")

    def test_requested_range_not_covered(self):
        path = self.write("r.csv", self.GOOD)
        with self.assertRaisesRegex(DataError, "does not cover"):
            load_ohlcv(path, start="2010-01-01")
        with self.assertRaisesRegex(DataError, "does not cover"):
            load_ohlcv(path, end="2030-01-01")

    def test_validate_rejects_naive_index_and_empty(self):
        df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
                          index=pd.DatetimeIndex(["2024-01-01"]))
        with self.assertRaisesRegex(DataError, "timezone-aware"):
            validate_ohlcv(df)
        with self.assertRaises(DataError):
            validate_ohlcv(df.iloc[:0])

    def test_cli_reports_data_error_instead_of_results(self):
        bad = self.write("bad.csv", "timestamp,open,high,low,close,volume\n2024-03-12T14:30:00Z,1,0.5,2,1,1\n")
        out = os.path.join(self.tmp.name, "out")
        code = cli_main(["--data", bad, "--symbol", "ES", "--out", out])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out), "no results must be written for invalid data")


class TestResampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_walk_df(days=4, seed=3)
        cls.data = MultiTimeframeData(cls.df, "ES")

    def test_5m_matches_manual_aggregation(self):
        b = self.data.bars["5m"]
        for k in (0, 7, 100, len(b) - 1):
            a = pd.Timestamp(int(b.open_time[k]), tz="UTC")
            win = self.df[(self.df.index >= a) & (self.df.index < a + pd.Timedelta(minutes=5))]
            self.assertAlmostEqual(b.open[k], win.open.iloc[0])
            self.assertAlmostEqual(b.high[k], win.high.max())
            self.assertAlmostEqual(b.low[k], win.low.min())
            self.assertAlmostEqual(b.close[k], win.close.iloc[-1])
            self.assertAlmostEqual(b.volume[k], win.volume.sum())
            self.assertEqual(b.close_time[k] - b.open_time[k], 5 * NS_PER_MIN)

    def test_sessions_align_to_17_ct(self):
        for tf in ("4h", "1d", "1h"):
            first = ns_to_chicago(self.data.bars[tf].open_time[0])
            self.assertEqual((first.hour, first.minute), (17, 0), tf)
        d = self.data.bars["1d"]
        self.assertEqual(ns_to_chicago(d.close_time[0]).hour, 17)
        h4 = [ns_to_chicago(x).hour for x in self.data.bars["4h"].open_time[:6]]
        self.assertEqual(h4, [17, 21, 1, 5, 9, 13])

    def test_dst_transition_keeps_local_alignment(self):
        df = random_walk_df(days=3, seed=4, start="2024-03-08")  # DST starts 2024-03-10
        data = MultiTimeframeData(df, "ES")
        for x in data.bars["1d"].open_time:
            self.assertEqual(ns_to_chicago(x).hour, 17)

    def test_all_timeframes_built_and_ordered(self):
        for tf, b in self.data.bars.items():
            self.assertTrue(np.all(np.diff(b.open_time) > 0), tf)
            self.assertTrue(np.all(b.close_time > b.open_time), tf)
            self.assertTrue(np.all(b.high >= b.low), tf)

    def test_arrays_are_read_only(self):
        with self.assertRaises(ValueError):
            self.data.bars["5m"].close[0] = 1.0


# ===========================================================================
# Lookahead protection
# ===========================================================================
class TestLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_walk_df(days=12, seed=11)
        cls.data = MultiTimeframeData(cls.df, "ES")

    def test_view_never_exposes_unfinished_bars(self):
        m1 = self.data.m1
        rng = np.random.default_rng(0)
        for i in rng.integers(0, len(m1), 200):
            now = int(m1.close_time[i])
            v = self.data.view(now)
            for tf in self.data.bars:
                b = v.bars(tf)
                if len(b):
                    self.assertLessEqual(int(b.close_time[-1]), now, tf)
                full = self.data.bars[tf]
                if len(b) < len(full):
                    self.assertGreater(int(full.close_time[len(b)]), now, tf)

    def test_htf_bar_appears_only_after_its_close(self):
        b5 = self.data.bars["5m"]
        k = 50
        close = int(b5.close_time[k])
        self.assertEqual(self.data.view(close - NS_PER_MIN).count("5m"), k)
        self.assertEqual(self.data.view(close).count("5m"), k + 1)
        d = self.data.bars["1d"]
        self.assertEqual(self.data.view(int(d.close_time[0]) - 1).count("1d"), 0)

    def test_swings_need_confirmation_bars(self):
        high = np.array([1, 2, 3, 10, 3, 2, 1], float)
        self.assertEqual(list(swing_highs(high, 2)), [3])
        self.assertEqual(list(swing_highs(high[:5], 2)), [])  # only one bar after the peak: not confirmed

    def _decisions(self, df, cut_ns=None):
        cfg = default_config()
        cfg["strategy"]["bias_min_agreement"] = 1      # exercise the full pipeline more often
        cfg["strategy"]["bias_allow_opposing"] = True
        cfg["risk"]["risk_per_trade_pct"] = 2.0
        data = MultiTimeframeData(df, "ES")
        res = Backtester(data, cfg, record_all_decisions=True).run()
        decs = [d.fingerprint() for d in res.decisions if cut_ns is None or d.time <= cut_ns]
        return decs, res

    def test_truncated_and_poisoned_future_give_identical_past_decisions(self):
        df = self.df
        days = sorted(set(ns_to_chicago(x).date() for x in df.index.as_unit("ns").asi8[::600]))
        cut = ct(f"{days[-3]} 09:15").tz_convert("UTC")
        cut_ns = int(cut.value)
        full, res_full = self._decisions(df, cut_ns)
        self.assertGreater(len(full), 200, "test must exercise many decisions")
        self.assertTrue(any(f[1] != "NO_TRADE" or f[2] != "BIAS_NEUTRAL" for f in full))
        # 1) future simply removed
        trunc, _ = self._decisions(df[df.index < cut + pd.Timedelta(minutes=0)], cut_ns)
        self.assertEqual(full, trunc)
        # 2) future replaced by completely different prices
        poisoned = df.copy()
        after = poisoned.index >= cut
        rng = np.random.default_rng(99)
        shift = np.round(rng.normal(0, 30, after.sum()) * 4) / 4
        for col in ("open", "high", "low", "close"):
            poisoned.loc[after, col] = poisoned.loc[after, col].to_numpy() + shift + 200
        pois, _ = self._decisions(poisoned, cut_ns)
        self.assertEqual(full, pois)

    @staticmethod
    def _snapshot(data, t, cfg):
        """Everything the strategy derives at time t (fresh strategy, no caches)."""
        strat = PBBlakeStrategy(cfg, ES)
        view = data.view(t)
        bias = determine_higher_timeframe_bias(view, cfg["strategy"], 0.25)
        snap = [tuple(sorted(bias.items())), view.last_close]
        for d in (BULLISH, BEARISH):
            draw = identify_draw_on_liquidity(view, d, cfg["strategy"], 0.25)
            snap.append(tuple(sorted(draw.items())) if draw else None)
            sel, all_lv = identify_key_levels(view, d, cfg["strategy"], 0.25, TouchTracker())
            snap.append(tuple((l.id, l.touch_index, l.mitigated, l.swept) for l in all_lv))
        snap.append(strat.evaluate(view).fingerprint())
        return tuple(snap)

    def test_point_in_time_analysis_ignores_future(self):
        cfg = default_config()
        cfg["strategy"]["bias_min_agreement"] = 1
        cfg["strategy"]["bias_allow_opposing"] = True
        m1 = self.data.m1
        sess = np.flatnonzero((self.data.close_minute_of_day >= 510) & (self.data.close_minute_of_day <= 600))
        sess = sess[sess > len(m1) // 3]  # leave some history for the higher timeframes
        rng = np.random.default_rng(5)
        picks = sorted(rng.choice(sess, 30, replace=False))
        differing_future = 0
        for i in picks:
            t = int(m1.close_time[i])
            cut = pd.Timestamp(t, tz="UTC")
            past = self.df[self.df.index < cut]          # bars that have closed by t
            poisoned = self.df.copy()
            fut = poisoned.index >= cut
            for col in ("open", "high", "low", "close"):
                poisoned.loc[fut, col] = poisoned.loc[fut, col].to_numpy() * 1.05 + 37.25
            base = self._snapshot(self.data, t, cfg)
            self.assertEqual(base, self._snapshot(MultiTimeframeData(past, "ES"), t, cfg), f"truncated @ {cut}")
            self.assertEqual(base, self._snapshot(MultiTimeframeData(poisoned, "ES"), t, cfg), f"poisoned @ {cut}")
            differing_future += int(fut.any())
        self.assertEqual(differing_future, 30)

    def test_entry_fills_after_signal_bar(self):
        cfg = scenario_config()
        data = MultiTimeframeData(bullish_scenario_df(), "ES")
        res = Backtester(data, cfg, strategy=FixedBiasStrategy(cfg, ES, BULLISH)).run()
        self.assertEqual(len(res.trades), 1)
        t = res.trades[0]
        i = int(np.searchsorted(data.m1.open_time, t.signal_time))
        self.assertGreaterEqual(t.entry_time, t.signal_time)
        self.assertEqual(t.entry_time, int(data.m1.open_time[i]))
        self.assertAlmostEqual(t.entry_price, data.m1.open[i] + ES.tick_size)


# ===========================================================================
# Indicators
# ===========================================================================
class TestIndicators(unittest.TestCase):
    def test_bullish_fvg(self):
        b = make_bars([(10, 11, 9, 10.5), (10.5, 14, 10.5, 13.5), (13.5, 15, 12, 14.5)])
        g = find_fvgs(b)
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0].direction, g[0].bottom, g[0].top), (BULLISH, 11.0, 12.0))
        self.assertEqual(g[0].created_time, int(b.close_time[2]))

    def test_bearish_fvg(self):
        b = make_bars([(15, 16, 14, 14.5), (14.5, 14.5, 10, 10.5), (10.5, 12, 9, 9.5)])
        g = find_fvgs(b)
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0].direction, g[0].bottom, g[0].top), (BEARISH, 12.0, 14.0))

    def test_no_fvg_when_wicks_touch_and_min_size(self):
        b = make_bars([(10, 11, 9, 10.5), (10.5, 14, 10.5, 13.5), (13.5, 15, 11, 14.5)])
        self.assertEqual(find_fvgs(b), [])
        b2 = make_bars([(10, 11, 9, 10.5), (10.5, 14, 10.5, 13.5), (13.5, 15, 11.25, 14.5)])
        self.assertEqual(len(find_fvgs(b2, min_size=0.25)), 1)
        self.assertEqual(len(find_fvgs(b2, min_size=0.5)), 0)

    def _gap_then(self, rest):
        base = [(10, 11, 9, 10.5), (10.5, 14, 10.5, 13.5), (13.5, 15, 12, 14.5)]
        b = make_bars(base + rest)
        return b, find_fvgs(b, 0, 3)[0]

    def test_gap_respected_by_reaction(self):
        b, g = self._gap_then([(14, 14.5, 11.5, 13), (13, 14, 12.5, 13.8)])
        st = gap_status(b, g)
        self.assertEqual(st.touched, 3)
        self.assertEqual(st.state, "RESPECTED")

    def test_wick_through_is_not_disrespect_body_close_is(self):
        b, g = self._gap_then([(14, 14.5, 10.0, 11.5)])  # wick below bottom (11), close inside
        self.assertEqual(gap_status(b, g).invalidated, -1)
        b, g = self._gap_then([(14, 14.5, 10.0, 10.75)])  # body close below bottom
        st = gap_status(b, g)
        self.assertEqual(st.invalidated, 3)
        self.assertEqual(st.state, "DISRESPECTED")

    def test_swings(self):
        low = np.array([5, 4, 3, 1, 3, 4, 5], float)
        self.assertEqual(list(swing_lows(low, 2)), [3])
        high = np.array([1, 3, 3, 1, 0], float)  # equal highs: first is the swing (left strict, right >=)
        self.assertEqual(list(swing_highs(high, 1)), [1])

    def test_bullish_cisd(self):
        b = make_bars([(10, 10.5, 9.5, 10), (10, 10.2, 9.0, 9.2), (9.2, 9.3, 8.0, 8.2),
                       (8.2, 9.5, 8.1, 9.4), (9.4, 10.5, 9.3, 10.2)])
        c = find_cisd(b, BULLISH, touch_index=2)
        self.assertIsNotNone(c)
        self.assertEqual((c.series_start, c.series_end, c.price), (1, 2, 10.0))
        self.assertEqual(c.confirm_index, 4)  # 9.4 close (bar 3) is not above 10.0, bar 4 is

    def test_bearish_cisd(self):
        b = make_bars([(10, 10.5, 9.5, 10), (10, 11, 9.9, 10.8), (10.8, 12, 10.7, 11.9),
                       (11.9, 12, 9.5, 9.8)])
        c = find_cisd(b, BEARISH, touch_index=2)
        self.assertEqual((c.price, c.confirm_index), (10.0, 3))

    def test_cisd_not_confirmed_or_invalidated(self):
        b = make_bars([(10, 10.2, 9.0, 9.2), (9.2, 9.3, 8.0, 8.2), (8.2, 9.5, 8.1, 9.4)])
        self.assertIsNone(find_cisd(b, BULLISH, touch_index=1))
        b2 = make_bars([(10, 10.2, 9.0, 9.2), (9.2, 9.3, 8.0, 8.2), (8.2, 10.5, 8.1, 10.4)])
        self.assertIsNotNone(find_cisd(b2, BULLISH, 1))
        self.assertIsNone(find_cisd(b2, BULLISH, 1, invalid_index=2))

    def test_rejection_block_and_ce(self):
        b, g = self._gap_then([(14, 14.5, 11.5, 13.5)])  # lower wick 11.5..13.5 into gap [11,12]
        rb = find_rejection_block(b, g)
        self.assertIsNotNone(rb)
        self.assertEqual((rb.wick_low, rb.wick_high, rb.ce), (11.5, 13.5, 12.5))
        b, g = self._gap_then([(14, 14.5, 12.5, 13.5)])  # wick does not reach the gap
        self.assertIsNone(find_rejection_block(b, g))

    def test_confirm_ifvg_spec_3_7(self):
        bear_gap = FVG("4m", BEARISH, 105.0, 103.0, 0, 2, 0, 0)
        self.assertTrue(confirm_ifvg(bear_gap, BULLISH, 105.25))
        self.assertFalse(confirm_ifvg(bear_gap, BULLISH, 105.0))  # must close ABOVE
        bull_gap = FVG("4m", BULLISH, 105.0, 103.0, 0, 2, 0, 0)
        self.assertTrue(confirm_ifvg(bull_gap, BEARISH, 102.75))
        self.assertFalse(confirm_ifvg(bull_gap, NEUTRAL, 1.0))


# ===========================================================================
# Structure
# ===========================================================================
class TestStructure(unittest.TestCase):
    def test_timeframe_bias(self):
        gap = [(10, 11, 9, 10.5), (10.5, 14, 10.5, 13.5), (13.5, 15, 12, 14.5)]
        resp = make_bars(gap + [(14, 14.5, 11.5, 13), (13, 14, 12.5, 13.8)], "1h")
        self.assertEqual(timeframe_bias(resp, 50)["direction"], BULLISH)
        disresp = make_bars(gap + [(14, 14.5, 10.0, 10.5)], "1h")
        self.assertEqual(timeframe_bias(disresp, 50)["direction"], BEARISH)
        bear = [(15, 16, 14, 14.5), (14.5, 14.5, 10, 10.5), (10.5, 12, 9, 9.5)]
        closed_through = make_bars(bear + [(9.5, 15, 9.4, 14.8)], "1h")
        self.assertEqual(timeframe_bias(closed_through, 50)["direction"], BULLISH)
        pending = make_bars(gap, "1h")
        self.assertEqual(timeframe_bias(pending, 50)["direction"], NEUTRAL)

    def _bias_with(self, daily, h4, h1, m15, cfg_over=None):
        cfg = default_config()["strategy"]
        cfg.update(cfg_over or {})
        res = iter([daily, h4, h1, m15])
        orig = structure_mod.timeframe_bias
        structure_mod.timeframe_bias = lambda *a, **k: {"direction": next(res)}
        try:
            dummy = make_bars([(1, 1, 1, 1)])
            view = FakeView({tf: dummy for tf in ("1d", "4h", "1h", "15m")}, 0)
            return determine_higher_timeframe_bias(view, cfg, 0.25)
        finally:
            structure_mod.timeframe_bias = orig

    def test_bias_combination(self):
        self.assertEqual(self._bias_with(BULLISH, BULLISH, BULLISH, BULLISH)["direction"], BULLISH)
        self.assertEqual(self._bias_with(BULLISH, BULLISH, NEUTRAL, BEARISH)["direction"], BULLISH)
        self.assertEqual(self._bias_with(BULLISH, BULLISH, BEARISH, BULLISH)["direction"], NEUTRAL)
        self.assertEqual(self._bias_with(BEARISH, NEUTRAL, NEUTRAL, BEARISH)["direction"], NEUTRAL)
        b = self._bias_with(BEARISH, BEARISH, BEARISH, BULLISH)
        self.assertEqual((b["direction"], b["fifteen_minute_confirmation"]), (BEARISH, False))
        b = self._bias_with(BEARISH, BEARISH, BEARISH, BULLISH, {"require_15m_confirmation": True})
        self.assertEqual(b["direction"], NEUTRAL)
        self.assertEqual(set(b), {"direction", "draw_on_liquidity", "liquidity_type", "daily_bias",
                                  "four_hour_bias", "one_hour_bias", "fifteen_minute_confirmation",
                                  "smt_confirmation"})  # spec 1.5 output shape

    def test_smt(self):
        def bars(lows):
            return make_bars([(l + 1, l + 2, l, l + 1) for l in lows], "15m")
        a = FakeView({"15m": bars([10, 9, 8, 9, 10, 9, 7.5, 8])}, 0)   # A sweeps its low (8 -> 7.5)
        b = FakeView({"15m": bars([10, 9, 8, 9, 10, 9, 8.5, 9])}, 0)   # B holds
        self.assertTrue(smt_divergence(a, b, "15m", 8, 2, BULLISH))
        self.assertFalse(smt_divergence(a, a, "15m", 8, 2, BULLISH))
        self.assertIsNone(smt_divergence(a, b, "15m", 50, 2, BULLISH))

    def _liq_view(self, highs_1h, last_close=100.0):
        h1 = make_bars([(h - 1, h, h - 2, h - 1) for h in highs_1h], "1h", "2024-03-11 17:00")
        h4 = make_bars([(99, 100, 98, 99)] * 3, "4h", "2024-03-11 17:00")
        now = int(h1.close_time[-1])
        m1 = make_bars([(last_close, last_close + .25, last_close - .25, last_close)], "1m",
                       str(ns_to_chicago(now - NS_PER_MIN).tz_localize(None)))
        return FakeView({"1h": h1, "4h": h4, "1m": m1}, now)

    def test_draw_on_liquidity_nearest_unswept(self):
        cfg = default_config()["strategy"]
        v = self._liq_view([100, 101, 110, 101, 100, 106, 100, 99, 98, 97])
        d = identify_draw_on_liquidity(v, BULLISH, cfg, 0.25)
        self.assertEqual((d["price"], d["liquidity_type"]), (106.0, "BUY_SIDE"))
        # a swing that was later exceeded is swept and ignored
        v = self._liq_view([100, 101, 104, 101, 100, 106, 100, 99, 98, 97])
        self.assertEqual(identify_draw_on_liquidity(v, BULLISH, cfg, 0.25)["price"], 106.0)
        self.assertIsNone(identify_draw_on_liquidity(v, NEUTRAL, cfg, 0.25))

    def test_draw_prefers_relative_equal_highs(self):
        cfg = default_config()["strategy"]
        # 112.5 and 112.25 are within 4 ticks and neither has been traded through
        v = self._liq_view([100, 101, 112.5, 101, 100, 112.25, 100, 101, 106, 101, 100, 99, 98])
        d = identify_draw_on_liquidity(v, BULLISH, cfg, 0.25)
        self.assertEqual((d["kind"], d["price"]), ("RELATIVE_EQUAL", 112.5))

    def test_target_nearest_valid_not_automatic_3r(self):
        targets = [(101.0, "a"), (102.0, "b"), (106.0, "c")]
        self.assertEqual(select_target(BULLISH, 100, 98, targets, 1.0)[:2], (102.0, "b"))
        self.assertEqual(select_target(BULLISH, 100, 99, targets, 1.0)[:2], (101.0, "a"))
        self.assertIsNone(select_target(BULLISH, 100, 90, targets, 1.0))
        self.assertEqual(select_target(BEARISH, 100, 102, [(97.0, "x"), (99.0, "y")], 1.0)[:2], (97.0, "x"))
        self.assertAlmostEqual(calculate_risk_reward(100, 95, 110), 2.0)
        self.assertEqual(calculate_risk_reward(100, 100, 110), 0.0)

    def test_highest_timeframe_ifvg_spec_3_3_example(self):
        """1M-4M have an IFVG inside the leg, 5M does not -> the 4M gap must be chosen."""
        gaps = {tf: [FVG(tf, BEARISH, 105.0, 104.0, 0, 2, 0, i)] for i, tf in enumerate(["1m", "2m", "3m", "4m"])}
        gaps["5m"] = []
        orig = structure_mod.fvgs_within_leg
        structure_mod.fvgs_within_leg = lambda view, tf, leg, d, m: gaps[tf]
        try:
            leg = ManipulationLeg(BULLISH, 0, 1, 110, 100, 0, 1)
            g = find_highest_timeframe_ifvg(None, leg, BULLISH, ["1m", "2m", "3m", "4m", "5m"])
            self.assertEqual(g.tf, "4m")
            self.assertIsNone(find_highest_timeframe_ifvg(None, leg, BEARISH, ["1m", "2m", "3m", "4m", "5m"]))
        finally:
            structure_mod.fvgs_within_leg = orig

    def test_touch_tracker_forward_only(self):
        m1 = make_bars([(10, 10.5, 9.5, 10)] * 5 + [(10, 10, 8, 9)] + [(9, 9.5, 8.5, 9)] * 4)
        tr = TouchTracker()
        self.assertEqual(tr.first(m1.head(5), 0, 8.5, BULLISH, False), -1)
        self.assertEqual(tr.first(m1.head(8), 0, 8.5, BULLISH, False), 5)
        self.assertEqual(tr.first(m1.head(8), 0, 8.0, BULLISH, True), -1)   # strict sweep needs < 8.0
        self.assertEqual(tr.first(m1.head(3), 0, 8.5, BULLISH, False), -1)  # time went back: cache reset


class TestScenario(unittest.TestCase):
    """Steps 2-10 on the hand-built bullish setup (bias step pinned)."""

    @classmethod
    def setUpClass(cls):
        cls.df = bullish_scenario_df()
        cls.data = MultiTimeframeData(cls.df, "ES")
        cls.cfg = scenario_config()
        cls.res = Backtester(cls.data, cls.cfg, strategy=FixedBiasStrategy(cls.cfg, ES, BULLISH),
                             record_all_decisions=True).run()

    def test_one_long_trade_with_spec_entry_stop_target(self):
        self.assertEqual(len(self.res.trades), 1)
        t = self.res.trades[0]
        self.assertEqual(t.direction, BULLISH)
        sig_local = ns_to_chicago(t.signal_time)
        self.assertTrue(ct("2024-03-12 08:30") <= sig_local <= ct("2024-03-12 10:00"))
        # default stop: below the manipulation swing low by one tick
        self.assertAlmostEqual(t.initial_stop, t.leg["end_price"] - 0.25)
        # entry reference = body close of the confirming IFVG candle, above the gap
        self.assertGreater(t.planned_entry, t.ifvg["upper"])
        self.assertGreaterEqual(t.planned_rr, 1.0)
        self.assertEqual(t.exit_reason, "TARGET")
        self.assertEqual(t.target, 130.0)

    def test_ifvg_is_highest_timeframe_inside_leg(self):
        t = self.res.trades[0]
        leg = ManipulationLeg(BULLISH, 0, 0, t.leg["start_price"], t.leg["end_price"], t.leg["start_time"],
                              t.leg["end_time"])
        view = self.data.view(t.signal_time)
        have = [tf for tf in ["1m", "2m", "3m", "4m", "5m"]
                if structure_mod.fvgs_within_leg(view, tf, leg, BEARISH, 0.25)]
        self.assertTrue(have)
        self.assertEqual(t.ifvg["timeframe"], have[-1])

    def test_no_entry_before_highest_tf_inversion(self):
        t = self.res.trades[0]
        tf = t.ifvg["timeframe"]
        b = self.data.bars[tf]
        # the confirming candle closes exactly at the signal time and is the first close above the gap
        j = int(np.searchsorted(b.close_time, t.signal_time))
        self.assertEqual(int(b.close_time[j]), t.signal_time)
        self.assertGreater(b.close[j], t.ifvg["upper"])
        c3 = int(np.searchsorted(b.close_time, t.ifvg["created_time"]))
        self.assertFalse((b.close[c3 + 1:j] > t.ifvg["upper"]).any())
        # earlier decisions were WAITs on that timeframe, not signals
        earlier = [d for d in self.res.decisions if d.time < t.signal_time and d.status == "SIGNAL"]
        self.assertEqual(earlier, [])

    def test_key_levels_aligned_and_capped(self):
        t = self.res.trades[0]
        view = self.data.view(t.signal_time)
        strat = PBBlakeStrategy(self.cfg, ES)
        sel, all_lv = identify_key_levels(view, BULLISH, self.cfg["strategy"], 0.25, strat.tracker)
        self.assertLessEqual(len(sel), 3)
        self.assertTrue(all(l.direction == BULLISH for l in all_lv))
        pr = [l.priority for l in sel]
        self.assertEqual(pr, sorted(pr))
        self.assertIn(t.key_level["type"], ("FVG", "INTERMEDIATE_HIGH_LOW", "CISD", "REJECTION_BLOCK"))
        for k in ("type", "direction", "timeframe", "upper_boundary", "lower_boundary", "midpoint", "mitigated",
                  "swept", "cisd_confirmed", "fvg_confluence", "smt_confluence", "valid"):
            self.assertIn(k, t.key_level)  # spec 2.5 fields

    def test_manipulation_leg(self):
        t = self.res.trades[0]
        m1 = self.data.m1
        s = int(np.searchsorted(m1.open_time, t.leg["start_time"]))
        e = int(np.searchsorted(m1.close_time, t.leg["end_time"]))
        self.assertAlmostEqual(m1.high[s], t.leg["start_price"])
        self.assertAlmostEqual(m1.low[e], t.leg["end_price"])
        self.assertAlmostEqual(m1.high[s:e + 1].max(), t.leg["start_price"])
        self.assertAlmostEqual(m1.low[s:e + 1].min(), t.leg["end_price"])
        self.assertLessEqual(t.leg["end_price"], t.key_level["upper_boundary"])

    def test_bearish_bias_never_takes_this_long(self):
        res = Backtester(self.data, self.cfg, strategy=FixedBiasStrategy(self.cfg, ES, BEARISH)).run()
        self.assertTrue(all(t.direction == BEARISH for t in res.trades))

    def test_session_window_enforced(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["execution"]["session"]["end"] = "08:45"   # setup confirms at 09:00 -> no trade
        res = Backtester(self.data, cfg, strategy=FixedBiasStrategy(cfg, ES, BULLISH)).run()
        self.assertEqual(res.trades, [])

    def test_rr_below_minimum_is_no_trade(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["execution"]["take_profit"]["minimum_rr"] = 5.0
        res = Backtester(self.data, cfg, strategy=FixedBiasStrategy(cfg, ES, BULLISH),
                         record_all_decisions=True).run()
        self.assertEqual(res.trades, [])
        self.assertTrue(any(d.reason == "RR_BELOW_MINIMUM" for d in res.decisions))

    def test_alternative_stop_methods(self):
        for m in ("CANDLE_BODY", "FVG_BOUNDARY", "LEG_FVG", "ORDER_BLOCK"):
            cfg = copy.deepcopy(self.cfg)
            cfg["execution"]["stop_loss"]["default"] = m
            res = Backtester(self.data, cfg, strategy=FixedBiasStrategy(cfg, ES, BULLISH),
                             record_all_decisions=True).run()
            for t in res.trades:
                self.assertLess(t.initial_stop, t.planned_entry, m)
                self.assertGreaterEqual(t.planned_rr, 1.0, m)

    def test_limit_entry_alternative_requires_retrace(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["execution"]["entry"]["method"] = "IFVG_LIMIT"
        res = Backtester(self.data, cfg, strategy=FixedBiasStrategy(cfg, ES, BULLISH)).run()
        # price never comes back to the inverted gap in this scenario -> no fill, no trade
        self.assertEqual(res.trades, [])

    def test_trade_log_columns(self):
        df = trades_frame(self.res.trades)
        for col in ("entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "net_pnl", "r_multiple",
                    "commission", "slippage_cost", "key_level_type", "ifvg_tf", "leg_start_price", "bias_daily"):
            self.assertIn(col, df.columns)


class TestSMTIntegration(unittest.TestCase):
    def test_backtest_with_second_instrument(self):
        cfg = default_config()
        cfg["strategy"]["bias_min_agreement"] = 1
        cfg["strategy"]["bias_allow_opposing"] = True
        cfg["risk"]["risk_per_trade_pct"] = 2.0
        es = MultiTimeframeData(random_walk_df(days=10, seed=1), "ES")
        nq = MultiTimeframeData(random_walk_df(days=10, seed=2, step=2.0), "NQ")
        res = Backtester(es, cfg, smt_data=nq).run()
        self.assertGreater(len(res.trades), 0)
        for t in res.trades:
            self.assertIn(t.bias["smt_confirmation"], (True, False, None))
            self.assertEqual(t.key_level["smt_confluence"], bool(t.bias["smt_confirmation"]))


# ===========================================================================
# Costs
# ===========================================================================
class TestCosts(unittest.TestCase):
    def test_defaults_never_lowered(self):
        """AGENTS.md: never lower the default fees or slippage. These floors are locked."""
        floors = {"ES": 2.50, "NQ": 2.50, "MES": 0.85, "MNQ": 0.85}
        for sym, floor in floors.items():
            self.assertGreaterEqual(costs_mod.DEFAULT_COMMISSION_PER_SIDE[sym], floor, sym)
            self.assertGreaterEqual(CostModel.default(sym).commission_per_side, floor, sym)
        self.assertGreaterEqual(costs_mod.DEFAULT_MARKET_SLIPPAGE_TICKS, 1)
        self.assertGreaterEqual(costs_mod.DEFAULT_STOP_SLIPPAGE_TICKS, 1)
        self.assertGreaterEqual(costs_mod.DEFAULT_LIMIT_SLIPPAGE_TICKS, 0)

    def test_instrument_specs(self):
        self.assertEqual(get_instrument("ES").point_value, 50.0)
        self.assertEqual(get_instrument("NQ").point_value, 20.0)
        self.assertEqual(get_instrument("MES").point_value, 5.0)
        self.assertEqual(get_instrument("ESH4").symbol, "ES")
        self.assertEqual(get_instrument("MNQZ4").symbol, "MNQ")
        with self.assertRaises(ValueError):
            get_instrument("CL")

    def test_slippage_and_commission(self):
        c = CostModel.default("ES")
        self.assertEqual(c.apply_slippage(100.0, "BUY", "MARKET"), 100.25)
        self.assertEqual(c.apply_slippage(100.0, "SELL", "STOP"), 99.75)
        self.assertEqual(c.apply_slippage(100.0, "BUY", "LIMIT"), 100.0)
        self.assertEqual(c.round_trip_commission(2), 10.0)
        self.assertEqual(c.estimated_round_trip_cost(1), 5.0 + 0.5 * 50)


# ===========================================================================
# Risk manager
# ===========================================================================
class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.costs = CostModel.default("ES")
        self.rm = RiskManager(self.cfg, self.costs)
        self.rm.new_day("2024-03-12")
        self.t = ns("2024-03-12 09:00")

    def approve(self, sig=None, minute=540, has_pos=False):
        return self.rm.approve(sig or make_signal(t=self.t), self.t, minute, has_pos)

    def test_sizing(self):
        # 0.5% of 50k = 250. Per contract: 2pt*50 + 5 commission + 0.5pt*50 slippage = 130 -> 1
        self.assertEqual(self.rm.size(make_signal(entry=100, stop=98)), 1)
        self.rm.equity = 500_000  # 2500 / 130 = 19 -> capped at max_contracts 10
        self.assertEqual(self.rm.size(make_signal(entry=100, stop=98)), 10)
        self.rm.equity = 50_000
        r = self.approve(make_signal(entry=100, stop=94, target=110, t=self.t))  # 6pt -> 330/contract
        self.assertIsInstance(r, Rejection)
        self.assertEqual(r.reason, "RISK_TOO_SMALL_FOR_ONE_CONTRACT")

    def test_session_window(self):
        self.assertIsInstance(self.approve(minute=8 * 60 + 30), Approval)
        self.rm._outstanding.clear()
        self.assertIsInstance(self.approve(minute=10 * 60), Approval)
        self.assertEqual(self.approve(minute=10 * 60 + 1).reason, "OUTSIDE_SESSION")
        self.assertEqual(self.approve(minute=8 * 60 + 29).reason, "OUTSIDE_SESSION")
        self.assertEqual(self.approve(minute=11 * 60 + 30).reason, "OUTSIDE_SESSION")  # no 11:00 extension

    def test_entries_after_end_cannot_be_enabled(self):
        cfg = default_config()
        cfg["execution"]["session"]["allow_entries_after_end"] = True
        with self.assertRaises(ValueError):
            RiskManager(cfg, self.costs)

    def test_no_adding_to_position(self):
        self.assertEqual(self.approve(has_pos=True).reason, "POSITION_ALREADY_OPEN")

    def test_wrong_side_and_tight_stops(self):
        self.assertEqual(self.approve(make_signal(entry=100, stop=101, t=self.t)).reason, "STOP_ON_WRONG_SIDE")
        self.assertEqual(self.approve(make_signal(entry=100, stop=98, target=99, t=self.t)).reason,
                         "TARGET_ON_WRONG_SIDE")
        self.assertEqual(self.approve(make_signal(entry=100, stop=99.5, t=self.t)).reason, "STOP_TOO_TIGHT")

    def _trade(self, pnl, exit_t=None):
        a = self.approve()
        self.assertIsInstance(a, Approval)
        self.rm.consume(a)
        self.rm.on_entry_filled(a)
        self.rm.on_trade_closed(pnl, exit_t or self.t + 10 * NS_PER_MIN)
        return a

    def test_stop_after_one_win(self):
        self._trade(+300)
        self.assertEqual(self.rm.day_blocked_reason(), "STOP_AFTER_WIN")
        self.assertEqual(self.approve().reason, "STOP_AFTER_WIN")

    def test_after_one_loss_only_new_a_plus_setup(self):
        self._trade(-150)
        self.assertIsNone(self.rm.day_blocked_reason())
        ctx = self.rm.trade_context()
        self.assertIn("L1", ctx.exclude_level_ids)
        self.assertEqual(ctx.require_touch_after_ns, self.t + 10 * NS_PER_MIN)
        same = make_signal(t=self.t + 20 * NS_PER_MIN, level_id="L1", touch=self.t + 15 * NS_PER_MIN)
        self.assertEqual(self.approve(same).reason, "KEY_LEVEL_ALREADY_TRADED")
        old_touch = make_signal(t=self.t + 20 * NS_PER_MIN, level_id="L2", touch=self.t)
        self.assertEqual(self.approve(old_touch).reason, "NO_NEW_A_PLUS_SETUP")
        new = make_signal(t=self.t + 20 * NS_PER_MIN, level_id="L2", touch=self.t + 15 * NS_PER_MIN)
        self.assertIsInstance(self.approve(new), Approval)

    def test_one_loss_stops_day_when_exception_disabled(self):
        self.cfg["execution"]["daily_limits"]["allow_new_A_PLUS_setup"] = False
        self.rm = RiskManager(self.cfg, self.costs)
        self.rm.new_day("2024-03-12")
        self._trade(-150)
        self.assertEqual(self.rm.day_blocked_reason(), "STOP_AFTER_LOSS")

    def test_two_losses_and_max_trades(self):
        self._trade(-150)
        a = self.rm.approve(make_signal(t=self.t, level_id="L2", touch=self.t + 11 * NS_PER_MIN), self.t, 540, False)
        self.rm.consume(a)
        self.rm.on_entry_filled(a)
        self.rm.on_trade_closed(-150, self.t + 30 * NS_PER_MIN)
        self.assertIn(self.rm.day_blocked_reason(), ("MAX_TRADES_REACHED", "STOP_AFTER_TWO_LOSSES"))
        self.rm.new_day("2024-03-13")
        self.assertIsNone(self.rm.day_blocked_reason())  # limits reset next day

    def test_scratch_trade_counts_as_loss(self):
        self._trade(-5.0)  # break-even exit minus fees
        self.assertEqual(self.rm.day.losses, 1)

    def test_daily_loss_limit_and_drawdown_halt(self):
        self.cfg["execution"]["daily_limits"]["max_trades"] = 10
        self.cfg["execution"]["daily_limits"]["max_daily_losses"] = 10
        self.cfg["execution"]["daily_limits"]["stop_after_two_losses"] = False
        self.cfg["execution"]["daily_limits"]["stop_after_one_loss"] = False
        self.rm = RiskManager(self.cfg, self.costs)
        self.rm.new_day("2024-03-12")
        self._trade(-1000.0)  # 2% of 50k
        self.assertEqual(self.rm.day_blocked_reason(), "DAILY_LOSS_LIMIT")
        self.rm.new_day("2024-03-13")
        self.rm.on_trade_closed(-12_000.0, self.t)
        self.assertTrue(self.rm.halted)
        self.assertEqual(self.rm.day_blocked_reason(), "MAX_DRAWDOWN_HALT")

    def test_stop_can_only_tighten(self):
        RiskManager.validate_stop_change(BULLISH, 98.0, 99.0)
        RiskManager.validate_stop_change(BEARISH, 102.0, 101.0)
        with self.assertRaises(RiskViolation):
            RiskManager.validate_stop_change(BULLISH, 98.0, 97.0)
        with self.assertRaises(RiskViolation):
            RiskManager.validate_stop_change(BEARISH, 102.0, 103.0)


# ===========================================================================
# Execution
# ===========================================================================
class TestExecution(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.costs = CostModel.default("ES")
        self.rm = RiskManager(self.cfg, self.costs)
        self.rm.new_day("2024-03-12")
        self.broker = PaperBroker(self.costs, self.rm, "ES")
        self.t = ns("2024-03-12 09:00")

    def submit(self, sig=None, expiry=None):
        a = self.rm.approve(sig or make_signal(t=self.t), self.t, 540, False)
        self.assertIsInstance(a, Approval)
        return self.broker.submit(a, expiry)

    def bar(self, o, h, l, c, k=1, flatten=False):
        ot = self.t + (k - 1) * NS_PER_MIN
        return self.broker.on_bar(o, h, l, c, ot, ot + NS_PER_MIN, flatten)

    def test_paper_only_and_approval_required(self):
        with self.assertRaises(RuntimeError):
            PaperBroker(self.costs, self.rm, "ES", mode="LIVE")
        fake = Approval(999, make_signal(t=self.t), 1, 100.0, self.t)
        with self.assertRaises(RiskViolation):
            self.broker.submit(fake)
        a = self.rm.approve(make_signal(t=self.t), self.t, 540, False)
        self.broker.submit(a)
        self.broker.pending = None
        with self.assertRaises(RiskViolation):  # approvals are single-use
            self.broker.submit(a)

    def test_market_entry_and_stop_with_slippage(self):
        self.submit()
        self.bar(100.5, 100.75, 100.0, 100.5)          # fill at open 100.5 + 0.25
        t = self.broker.position
        self.assertEqual(t.entry_price, 100.75)
        out = self.bar(100.0, 100.25, 97.5, 97.75, k=2)  # stop 98 -> 97.75 with slippage
        self.assertEqual(len(out), 1)
        t = out[0]
        self.assertEqual((t.exit_reason, t.exit_price), ("STOP", 97.75))
        self.assertAlmostEqual(t.gross_pnl, (97.75 - 100.75) * 50)
        self.assertAlmostEqual(t.commission, 5.0)
        self.assertAlmostEqual(t.net_pnl, t.gross_pnl - 5.0)
        self.assertAlmostEqual(t.r_multiple, t.net_pnl / ((100.75 - 98.0) * 50))
        self.assertAlmostEqual(self.rm.equity, 50_000 + t.net_pnl)

    def test_gap_through_stop_fills_at_open(self):
        self.submit()
        self.bar(100, 100.25, 99.75, 100)
        out = self.bar(96.0, 96.5, 95.0, 96.0, k=2)
        self.assertEqual(out[0].exit_price, 95.75)

    def test_target_requires_trade_through(self):
        self.submit()
        self.bar(100, 100.25, 99.75, 100)
        self.assertEqual(self.bar(103, 104.0, 102.5, 103.5, k=2), [])   # touch only
        out = self.bar(103.5, 104.25, 103.0, 104.0, k=3)
        self.assertEqual((out[0].exit_reason, out[0].exit_price), ("TARGET", 104.0))

    def test_same_bar_stop_and_target_assumes_stop(self):
        self.submit()
        self.bar(100, 100.25, 99.75, 100)
        out = self.bar(100, 105.0, 97.0, 101, k=2)
        self.assertEqual(out[0].exit_reason, "STOP")

    def test_break_even_moves_stop_to_entry(self):
        sig = make_signal(t=self.t, target=106.0)
        sig.break_even_reference = 102.0
        self.submit(sig)
        self.bar(100, 100.25, 99.75, 100)        # entry 100.25
        self.bar(101, 102.25, 100.75, 102, k=2)  # reference reached -> stop to entry
        self.assertEqual(self.broker.position.stop, 100.25)
        out = self.bar(101, 101, 100.0, 100.0, k=3)
        self.assertEqual(out[0].exit_reason, "BREAK_EVEN_STOP")
        self.assertLess(out[0].net_pnl, 0)       # costs make a scratch a small loss

    def test_limit_entry_needs_trade_through_and_expires(self):
        sig = make_signal(t=self.t)
        sig.entry_method = "IFVG_LIMIT"
        self.submit(sig, expiry=self.t + 3 * NS_PER_MIN)
        self.bar(100.5, 101, 100.0, 100.5)                  # touches 100, not through
        self.assertIsNone(self.broker.position)
        self.bar(100.5, 101, 99.75, 100.5, k=2)             # through -> fill at limit
        self.assertEqual(self.broker.position.entry_price, 100.0)
        # expiry path
        broker = PaperBroker(self.costs, RiskManager(self.cfg, self.costs), "ES")
        rm = broker.risk
        rm.new_day("2024-03-12")
        a = rm.approve(sig, self.t, 540, False)
        broker.submit(a, self.t + NS_PER_MIN)
        broker.on_bar(101, 101.5, 100.5, 101, self.t + NS_PER_MIN, self.t + 2 * NS_PER_MIN)
        self.assertTrue(broker.flat)
        self.assertEqual(rm.day.trades, 0)

    def test_flatten_and_force_close(self):
        self.submit()
        self.bar(100, 100.25, 99.75, 100)
        out = self.bar(101, 101.25, 100.75, 101, k=2, flatten=True)
        self.assertEqual((out[0].exit_reason, out[0].exit_price), ("FLATTEN_TIME", 100.75))
        self.assertIsNone(self.broker.force_close(100.0, self.t))

    def test_short_side(self):
        self.submit(make_signal(BEARISH, entry=100, stop=102, target=96, t=self.t))
        self.bar(100, 100.25, 99.75, 100)
        self.assertEqual(self.broker.position.entry_price, 99.75)
        out = self.bar(97, 97.25, 95.75, 96, k=2)
        self.assertEqual((out[0].exit_reason, out[0].exit_price), ("TARGET", 96.0))
        self.assertAlmostEqual(out[0].gross_pnl, (99.75 - 96.0) * 50)


# ===========================================================================
# Statistics
# ===========================================================================
class TestStats(unittest.TestCase):
    def mk(self, pnls, risk=100.0):
        out = []
        for i, p in enumerate(pnls):
            t = Trade(i, "ES", BULLISH, 1, 0, "IFVG_BODY_CLOSE", 100.0)
            t.net_pnl, t.risk_dollars, t.r_multiple = p, risk, p / risk
            t.exit_time = ns("2024-03-12 09:00") + i * NS_PER_MIN
            t.exit_reason = "TARGET" if p > 0 else "STOP"
            out.append(t)
        return out

    def test_metrics(self):
        s = compute_stats(self.mk([200, -100, -100, 300, -100]), 10_000)
        self.assertEqual(s["total_trades"], 5)
        self.assertAlmostEqual(s["win_rate"], 0.4)
        self.assertAlmostEqual(s["profit_factor"], 500 / 300)
        self.assertAlmostEqual(s["expectancy"], 40.0)
        self.assertAlmostEqual(s["average_r"], 0.4)
        self.assertAlmostEqual(s["max_drawdown"], -200.0)
        self.assertEqual(s["max_consecutive_losses"], 2)
        self.assertIn("too few", s["significance"])

    def test_no_trades_and_no_false_claims(self):
        s = compute_stats([], 10_000)
        self.assertEqual(s["total_trades"], 0)
        self.assertIsNone(s["win_rate"])
        s = compute_stats(self.mk([100, -100] * 30), 10_000)
        self.assertIn("not statistically different", s["significance"])
        self.assertGreater(s["p_value_win_rate_vs_50pct"], 0.5)


# ===========================================================================
# Scanner and live feeds (alerts only, no orders)
# ===========================================================================
from fbt import scanner as scanner_mod  # noqa: E402
from fbt.feeds import (ReplayFeed, completed_only, current_trading_date, ib_bars_to_frame,  # noqa: E402
                       ib_daily_to_frame, IBKRFeed, merge_bars)
from fbt.scanner import AlertSink, LiveRunner, ScanEvent, Scanner, replay_day  # noqa: E402


def _fixed_scanner(cfg, direction=BULLISH, symbols=("ES",)):
    return Scanner(cfg, list(symbols), strategy_factory=lambda s: FixedBiasStrategy(cfg, ES, direction))


class TestScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = bullish_scenario_df()
        cls.cfg = scenario_config()
        cls.data = {"ES": MultiTimeframeData(cls.df, "ES")}
        cls.events = replay_day(_fixed_scanner(cls.cfg), cls.data, "2024-03-12")
        cls.trade = Backtester(cls.data["ES"], cls.cfg, strategy=FixedBiasStrategy(cls.cfg, ES, BULLISH)).run().trades[0]

    def test_signal_matches_backtester(self):
        sig = [e for e in self.events if e.kind == "SIGNAL"]
        self.assertEqual(len(sig), 1)
        s = sig[0].data["signal"]
        t = self.trade
        self.assertEqual(sig[0].time, t.signal_time)
        self.assertEqual((s["entry_price"], s["stop"], s["target"]), (t.planned_entry, t.initial_stop, t.target))
        self.assertIn("ES LONG SIGNAL 09:00 CT", sig[0].text)
        self.assertIn("Stop  102.50", sig[0].text)
        self.assertIn("MES:", sig[0].text)  # micro sizing shown too

    def test_premarket_once_and_setup_forming_before_signal(self):
        kinds = [e.kind for e in self.events]
        self.assertEqual(kinds.count("PREMARKET"), 1)
        self.assertLess(kinds.index("PREMARKET"), kinds.index("SETUP_FORMING"))
        self.assertLess(kinds.index("SETUP_FORMING"), kinds.index("SIGNAL"))
        forming = [e.text for e in self.events if e.kind == "SETUP_FORMING"]
        self.assertTrue(any("5m candle body close above 109.75" in f for f in forming))

    def test_no_signals_outside_session(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["execution"]["session"]["end"] = "08:45"
        ev = replay_day(_fixed_scanner(cfg), self.data, "2024-03-12")
        self.assertFalse(any(e.kind == "SIGNAL" for e in ev))
        sc = _fixed_scanner(self.cfg)
        self.assertEqual(sc.scan(self.data, ns("2024-03-12 10:30")), [])

    def test_scanner_never_trades(self):
        src = open(scanner_mod.__file__).read() + open(sys.modules["fbt.feeds"].__file__).read()
        for forbidden in ("PaperBroker", ".submit(", "placeOrder", "MarketOrder", "LimitOrder"):
            self.assertNotIn(forbidden, src)

    def _runner(self, df, clock_box, cfg=None, sink_events=None, forming=True):
        feed = ReplayFeed({"ES": df}, include_forming_bar=forming)

        def clock():
            feed.now = clock_box[0]
            return clock_box[0]
        sink = AlertSink(webhook="", echo=lambda text: None)
        if sink_events is not None:
            sink.emit = lambda ev: sink_events.append(ev)
        return LiveRunner(_fixed_scanner(cfg or self.cfg), feed, sink, clock=clock), feed

    def test_live_runner_only_sees_completed_bars_and_matches_backtest(self):
        box = [ct("2024-03-12 08:00:03").tz_convert("UTC")]
        out = []
        runner, feed = self._runner(self.df, box, sink_events=out)
        runner.warm_up()
        while box[0] <= ct("2024-03-12 09:05:03"):
            runner.step()
            last = runner.store["ES"].index[-1]
            self.assertLessEqual(last + pd.Timedelta(minutes=1), box[0])  # forming bar never stored
            box[0] += pd.Timedelta(minutes=1)
        sig = [e for e in out if e.kind == "SIGNAL"]
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0].data["signal"]["entry_price"], self.trade.planned_entry)
        self.assertEqual(sig[0].data["signal"]["stop"], self.trade.initial_stop)

    def test_stale_and_invalid_data_are_reported_not_scanned(self):
        box = [ct("2024-03-12 08:45:03").tz_convert("UTC")]
        cut = self.df[self.df.index < ct("2024-03-12 08:35").tz_convert("UTC")]  # feed stopped at 08:35
        out = []
        runner, _ = self._runner(cut, box, sink_events=out)
        runner.warm_up()
        runner.step()
        self.assertEqual([e.kind for e in out], ["STALE"])
        bad = self.df.copy()
        i = bad.index.get_loc(ct("2024-03-12 08:44").tz_convert("UTC"))
        bad.iloc[i, bad.columns.get_loc("low")] = bad.iloc[i]["high"] + 5  # low above high
        out2 = []
        runner2, _ = self._runner(bad, box, sink_events=out2)
        runner2.warm_up()
        runner2.step()
        self.assertTrue(any(e.kind == "DATA_ERROR" for e in out2))
        self.assertFalse(any(e.kind == "SIGNAL" for e in out2))

    def test_alert_sink_webhook_and_log(self):
        sent = []
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "alerts.csv")
            sink = AlertSink(log_path=log, webhook="https://example.invalid/hook",
                             send=lambda url, text: sent.append(text), echo=lambda t: None)
            sink.emit(ScanEvent(ns("2024-03-12 09:00"), "ES", "SIGNAL", "ES LONG\nline2"))
            sink.emit(ScanEvent(ns("2024-03-12 08:25"), "ES", "PREMARKET", "summary"))
            self.assertEqual(sent, ["ES LONG\nline2"])           # pre-session summary is not pushed
            self.assertEqual(len(pd.read_csv(log)), 2)

            def boom(url, text):
                raise OSError("network down")
            msgs = []
            AlertSink(webhook="https://example.invalid/hook", send=boom, echo=msgs.append).emit(
                ScanEvent(0, "ES", "SIGNAL", "x"))
            self.assertIn("webhook failed", msgs[-1])             # failure reported, no crash
        quiet = AlertSink(webhook="", send=lambda u, t: sent.append("BAD"), echo=lambda t: None)
        quiet.emit(ScanEvent(0, "ES", "SIGNAL", "x"))
        self.assertNotIn("BAD", sent)


class _FakeBar:
    def __init__(self, date, o, h, l, c, v=1.0):
        self.date, self.open, self.high, self.low, self.close, self.volume = date, o, h, l, c, v


class TestFeeds(unittest.TestCase):
    def test_completed_only_and_merge(self):
        idx = pd.date_range("2024-03-12 14:00", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
        self.assertEqual(len(completed_only(df, pd.Timestamp("2024-03-12 14:02:30", tz="UTC"))), 2)
        self.assertEqual(len(completed_only(df, pd.Timestamp("2024-03-12 14:03", tz="UTC"))), 3)
        newer = df.iloc[-1:].copy()
        newer["close"] = 1.75
        m = merge_bars(df, newer)
        self.assertEqual(len(m), 3)
        self.assertEqual(m["close"].iloc[-1], 1.75)

    def test_ib_conversions(self):
        bars = [_FakeBar(pd.Timestamp("2024-03-12 14:00", tz="UTC").to_pydatetime(), 1, 2, 0.5, 1.5),
                _FakeBar(pd.Timestamp("2024-03-12 09:01", tz=CT).to_pydatetime(), 1, 2, 0.5, 1.5, -1)]
        df = ib_bars_to_frame(bars)
        self.assertEqual(list(df.index), [pd.Timestamp("2024-03-12 14:00", tz="UTC"),
                                          pd.Timestamp("2024-03-12 14:01", tz="UTC")])
        self.assertEqual(df["volume"].iloc[1], 0.0)  # IB's -1 'no volume' never becomes negative
        import datetime as dt
        daily = [_FakeBar(dt.date(2024, 3, 8), 1, 2, .5, 1.5), _FakeBar(dt.date(2024, 3, 11), 1, 2, .5, 1.5),
                 _FakeBar(dt.date(2024, 3, 12), 1, 2, .5, 1.5)]
        d = ib_daily_to_frame(daily, dt.date(2024, 3, 12))
        self.assertEqual(len(d), 2)  # today's unfinished day is dropped
        self.assertEqual(current_trading_date(pd.Timestamp("2024-03-12 23:30", tz="UTC")), dt.date(2024, 3, 13))

    def test_ibkr_feed_with_fake_client(self):
        class FakeIB:
            def __init__(self):
                self.calls = []

            def reqHistoricalData(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow,
                                  useRTH, formatDate):
                self.calls.append((durationStr, barSizeSetting, formatDate, useRTH))
                base = pd.Timestamp("2024-03-12 14:00", tz="UTC") - pd.Timedelta(days=len(self.calls) - 1)
                return [_FakeBar((base + pd.Timedelta(minutes=k)).to_pydatetime(), 10, 11, 9, 10.5)
                        for k in range(3)]

        feed = IBKRFeed(ib=FakeIB(), pacing_seconds=0)
        feed._contracts["ES"] = object()  # skip contract lookup (needs a live TWS)
        hist = feed.history("ES", 3)
        self.assertEqual(len(hist), 9)
        self.assertTrue(hist.index.is_monotonic_increasing)
        self.assertTrue(all(c == ("1 D", "1 min", 2, False) for c in feed.ib.calls))
        self.assertEqual(len(feed.recent("ES")), 3)

    def test_daily_history_merge_and_validation(self):
        df = random_walk_df(days=5, seed=8, start="2024-03-11")
        derived = MultiTimeframeData(df, "ES").bars["1d"]
        ext = pd.DataFrame({"open": [4000.0, 4010.0], "high": [4020.0, 4030.0], "low": [3990.0, 4000.0],
                            "close": [4010.0, 4020.0], "volume": [1.0, 1.0]},
                           index=pd.to_datetime(["2024-03-07", "2024-03-08"]))
        data = MultiTimeframeData(df, "ES", daily_history=ext)
        d = data.bars["1d"]
        self.assertEqual(len(d), len(derived) - 1 + 2)  # first (partial) 1m day replaced by history
        self.assertEqual(ns_to_chicago(d.open_time[0]).strftime("%Y-%m-%d %H:%M"), "2024-03-06 17:00")
        self.assertEqual(d.close[0], 4010.0)
        # a historical daily bar is only visible after its session closes
        self.assertEqual(data.view(int(d.close_time[1]) - 1).count("1d"), 1)
        bad = ext.copy()
        bad.iloc[0, bad.columns.get_loc("low")] = 5000.0
        with self.assertRaises(DataError):
            MultiTimeframeData(df, "ES", daily_history=bad)

    def test_new_instruments_and_cost_floors(self):
        floors = {"YM": 2.50, "MYM": 0.85, "GC": 2.50, "MGC": 0.85}
        for sym, floor in floors.items():
            self.assertGreaterEqual(costs_mod.DEFAULT_COMMISSION_PER_SIDE[sym], floor, sym)
        self.assertEqual(get_instrument("GCJ4").point_value, 100.0)
        self.assertEqual(get_instrument("MGCJ4").symbol, "MGC")
        self.assertEqual(get_instrument("YMH4").point_value, 5.0)
        self.assertEqual(get_instrument("MYM").point_value, 0.5)


class TestUnattendedRunning(unittest.TestCase):
    """The scanner runs for days with nobody watching: it must report problems and keep going."""

    def setUp(self):
        self.df = bullish_scenario_df()
        self.cfg = scenario_config()
        self.out = []

    def _runner(self, feed, box):
        sink = AlertSink(webhook="", echo=lambda t: None)
        sink.emit = lambda ev: self.out.append(ev)
        return LiveRunner(_fixed_scanner(self.cfg), feed, sink, clock=lambda: box[0])

    def test_unexpected_feed_errors_are_reported_and_recovered(self):
        feed = ReplayFeed({"ES": self.df})
        box = [ct("2024-03-12 08:40:03").tz_convert("UTC")]
        feed.now = box[0]
        runner = self._runner(feed, box)
        runner.warm_up()
        real_recent = feed.recent
        feed.recent = lambda sym: (_ for _ in ()).throw(TimeoutError("read timed out"))
        runner.step()  # must not raise
        self.assertTrue(any(e.kind == "DATA_ERROR" and "TimeoutError" in e.text for e in self.out))
        feed.recent = real_recent
        box[0] += pd.Timedelta(minutes=1)
        feed.now = box[0]
        self.out.clear()
        runner.step()
        self.assertFalse(any(e.kind == "DATA_ERROR" for e in self.out))

    def test_failed_warm_up_is_retried(self):
        feed = ReplayFeed({"ES": self.df})
        box = [ct("2024-03-12 08:40:03").tz_convert("UTC")]
        feed.now = box[0]
        real_history = feed.history
        feed.history = lambda sym, days: (_ for _ in ()).throw(ConnectionResetError("network down"))
        runner = self._runner(feed, box)
        runner.warm_up()
        self.assertIn("will retry", self.out[-1].text)
        feed.history = real_history
        runner.step()
        self.assertIn("ES", runner._warmed)
        self.assertGreater(len(runner.store["ES"]), 500)

    def test_quiet_outside_active_hours(self):
        feed = ReplayFeed({"ES": self.df})
        calls = []
        real = feed.recent
        feed.recent = lambda sym: calls.append(sym) or real(sym)
        box = [ct("2024-03-12 03:00:03").tz_convert("UTC")]
        feed.now = box[0]
        runner = self._runner(feed, box)
        runner.warm_up()
        self.assertEqual(runner.step(), [])
        self.assertEqual(calls, [])  # nights: no polling, no scanning

    def test_push_summary_is_opt_in(self):
        sent = []
        ev = ScanEvent(ns("2024-03-12 08:25"), "ES", "PREMARKET", "summary")
        AlertSink(webhook="https://example.invalid/h", send=lambda u, t: sent.append(t), echo=lambda t: None).emit(ev)
        self.assertEqual(sent, [])
        AlertSink(webhook="https://example.invalid/h", send=lambda u, t: sent.append(t), echo=lambda t: None,
                  push_summary=True).emit(ev)
        self.assertEqual(sent, ["summary"])

    def test_alpaca_timeout_becomes_data_error(self):
        import urllib.request
        from fbt.feeds import AlpacaFeed
        f = AlpacaFeed(env=TestAlpacaYahooETF.ENV)
        orig = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out"))
        try:
            with self.assertRaisesRegex(DataError, "TimeoutError"):
                f.recent("SPY")
        finally:
            urllib.request.urlopen = orig


class FakeAlpacaTrading:
    """Stand-in for Alpaca's paper Trading API. Records every call."""

    def __init__(self, equity="100000"):
        from fbt.trader import PAPER_URL
        self.base = PAPER_URL
        self.calls, self.positions, self.open, self.orders = [], [], [], {}
        self.equity = equity
        self.n = 0

    def __call__(self, method, url, headers, body):
        assert url.startswith(self.base), url                 # paper endpoint only
        assert headers["APCA-API-KEY-ID"] and "secret" not in url
        self.calls.append((method, url[len(self.base):], body))
        path = url[len(self.base):]
        if method == "GET" and path == "/account":
            return {"equity": self.equity}
        if method == "GET" and path == "/positions":
            return self.positions
        if method == "GET" and path.startswith("/orders?status=open"):
            return self.open
        if method == "GET" and path.startswith("/orders/"):
            return self.orders[path.split("/")[2].split("?")[0]]
        if method == "POST" and path == "/orders":
            self.n += 1
            oid = f"o{self.n}"
            self.orders[oid] = {"id": oid, "status": "new", "symbol": body["symbol"],
                                "legs": [{"id": oid + "-tp", "type": "limit", "status": "new"},
                                         {"id": oid + "-sl", "type": "stop", "status": "held"}]}
            return self.orders[oid]
        if method == "DELETE" and path.startswith("/orders/"):
            leg_id = path.split("/")[2]
            for o in self.orders.values():
                for leg in o.get("legs", []):
                    if leg["id"] == leg_id:
                        leg["status"] = "canceled"
            return {}
        if method == "DELETE" and path.startswith("/positions/"):
            self.orders["close1"] = {"id": "close1", "status": "new", "symbol": path.split("/")[2], "legs": []}
            return self.orders["close1"]
        raise AssertionError(f"unexpected call {method} {path}")

    def fill_entry(self, oid, price, qty=None):
        o = self.orders[oid]
        o.update(status="filled", filled_avg_price=str(price), filled_qty=str(qty or 0) if qty else None)
        o["legs"][1]["status"] = "new"

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


class TestPaperTrader(unittest.TestCase):
    KEYS = {"APCA_API_KEY_ID": "test-id-not-real", "APCA_API_SECRET_KEY": "test-secret-not-real"}

    def setUp(self):
        from fbt.trader import AlpacaPaperBroker, PaperTrader
        self.tmp = tempfile.TemporaryDirectory()
        self.api = FakeAlpacaTrading()
        self.cfg = default_config()
        self.day = pd.Timestamp.now(tz=CT).normalize()
        self.trader = PaperTrader(self.cfg, lambda rm: AlpacaPaperBroker(rm, http=self.api, env=self.KEYS), ["SPY"],
                                  state_path=os.path.join(self.tmp.name, "state.json"),
                                  trades_log=os.path.join(self.tmp.name, "trades.csv"))

    def tearDown(self):
        self.tmp.cleanup()

    def at(self, hhmm):
        h, m = map(int, hhmm.split(":"))
        return (self.day + pd.Timedelta(hours=h, minutes=m, seconds=3)).tz_convert("UTC")

    def signal_event(self, hhmm, direction=BULLISH, entry=100.0, stop=98.0, target=104.0, level="L1", touch=None):
        t = int(self.at(hhmm).floor("min").value)
        sig = make_signal(direction, entry, stop, target, t=t, level_id=level,
                          touch=touch if touch is not None else t - 5 * NS_PER_MIN)
        return ScanEvent(t, "SPY", "SIGNAL", "signal", {"signal": sig.to_dict()})

    def kinds(self, evs):
        return [e.kind for e in evs]

    def test_mode_is_paper_only(self):
        from fbt.trader import TradingError, build_paper_trader, resolve_trading_mode
        self.assertEqual(resolve_trading_mode(False, env={}), "paper")
        with self.assertRaisesRegex(TradingError, "LIVE_TRADING_CONFIRMED"):
            resolve_trading_mode(True, env={})
        with self.assertRaisesRegex(TradingError, "not implemented"):
            resolve_trading_mode(True, env={"LIVE_TRADING_CONFIRMED": "yes"})
        with self.assertRaisesRegex(TradingError, "feed alpaca"):
            build_paper_trader(self.cfg, ["SPY"], feed_name="ibkr", env=self.KEYS)
        with self.assertRaisesRegex(TradingError, "futures"):
            build_paper_trader(self.cfg, ["ES"], feed_name="alpaca", env=self.KEYS)
        with self.assertRaisesRegex(TradingError, "APCA_API_KEY_ID"):
            build_paper_trader(self.cfg, ["SPY"], feed_name="alpaca", env={})

    def test_orders_need_single_use_risk_approval(self):
        from fbt.risk import RiskViolation
        b = self.trader.broker
        self.assertNotIn("test-secret", repr(b))
        fake = Approval(999, make_signal(t=int(self.at("09:00").value)), 10, 20.0, 0)
        with self.assertRaises(RiskViolation):
            b.submit_bracket(fake, "SPY")
        self.assertEqual(self.api.posts(), [])  # nothing reached Alpaca

    def test_signal_becomes_bracket_order_then_win_stops_the_day(self):
        evs = self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.assertEqual(self.kinds(evs), ["ORDER"])
        (_, path, body), = self.api.posts()
        self.assertEqual((body["side"], body["type"], body["order_class"], body["time_in_force"]),
                         ("buy", "market", "bracket", "gtc"))
        self.assertEqual((body["stop_loss"]["stop_price"], body["take_profit"]["limit_price"]), ("98.00", "104.00"))
        # 0.5% of $100k = $500 / (2.00 stop + 0.01 commission + 0.02 slippage) per share = 246 shares
        self.assertEqual(body["qty"], "246")
        self.assertTrue(body["client_order_id"].startswith("pbb-SPY-"))
        self.api.fill_entry("o1", 100.05)
        self.assertEqual(self.kinds(self.trader.on_minute(self.at("09:01"), [])), ["FILL"])
        self.assertEqual(self.trader.risk.day.trades, 1)
        self.api.orders["o1"]["legs"][0].update(status="filled", filled_avg_price="104.00")
        ev = self.trader.on_minute(self.at("09:20"), [])
        self.assertEqual(self.kinds(ev), ["EXIT"])
        self.assertIn("WIN", ev[0].text)
        self.assertEqual(self.trader.risk.day.wins, 1)
        # spec 4.5: one win -> done for the day
        ev = self.trader.on_minute(self.at("09:40"), [self.signal_event("09:40", level="L2")])
        self.assertEqual(self.kinds(ev), ["SKIPPED"])
        self.assertIn("STOP_AFTER_WIN", ev[0].text)
        self.assertEqual(len(self.api.posts()), 1)
        log = pd.read_csv(os.path.join(self.tmp.name, "trades.csv"))
        self.assertEqual(log["reason"].iloc[0], "TARGET")

    def test_stop_loss_then_only_new_a_plus_setup(self):
        self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.api.fill_entry("o1", 100.00)
        self.trader.on_minute(self.at("09:01"), [])
        self.api.orders["o1"]["legs"][1].update(status="filled", filled_avg_price="97.99")
        ev = self.trader.on_minute(self.at("09:05"), [])
        self.assertIn("LOSS", ev[0].text)
        self.assertEqual(self.trader.risk.day.losses, 1)
        same = self.trader.on_minute(self.at("09:10"), [self.signal_event("09:10", level="L1")])
        self.assertIn("KEY_LEVEL_ALREADY_TRADED", same[0].text)
        old = self.trader.on_minute(self.at("09:12"), [self.signal_event(
            "09:12", level="L2", touch=int(self.at("09:02").floor("min").value))])
        self.assertIn("NO_NEW_A_PLUS_SETUP", old[0].text)
        new = self.trader.on_minute(self.at("09:20"), [self.signal_event("09:20", level="L3")])
        self.assertEqual(self.kinds(new), ["ORDER"])

    def test_existing_position_blocks_entry(self):
        self.api.positions = [{"symbol": "SPY", "qty": "10"}]
        ev = self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.assertIn("POSITION_ALREADY_OPEN", ev[0].text)
        self.assertEqual(self.api.posts(), [])

    def test_outside_session_signal_not_traded(self):
        ev = self.trader.on_minute(self.at("10:05"), [self.signal_event("10:05")])
        self.assertIn("OUTSIDE_SESSION", ev[0].text)
        self.assertEqual(self.api.posts(), [])

    def test_rejected_entry_is_not_a_trade(self):
        self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.api.orders["o1"]["status"] = "rejected"
        ev = self.trader.on_minute(self.at("09:01"), [])
        self.assertEqual(self.kinds(ev), ["ORDER_ERROR"])
        self.assertEqual(self.trader.risk.day.trades, 0)
        self.assertIsNone(self.trader.open)

    def test_flatten_before_etf_close(self):
        self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.api.fill_entry("o1", 100.00)
        self.trader.on_minute(self.at("09:01"), [])
        self.assertEqual(self.trader.on_minute(self.at("14:49"), []), [])
        ev = self.trader.on_minute(self.at("14:50"), [])
        self.assertIn("end-of-day flatten", ev[0].text)
        deletes = [c[1] for c in self.api.calls if c[0] == "DELETE"]
        self.assertEqual(deletes, ["/orders/o1-tp", "/orders/o1-sl", "/positions/SPY"])
        self.api.orders["close1"].update(status="filled", filled_avg_price="101.00")
        ev = self.trader.on_minute(self.at("14:51"), [])
        self.assertIn("FLATTEN_TIME", ev[0].text)
        self.assertIsNone(self.trader.open)

    def test_restart_same_day_remembers_open_trade_and_counts(self):
        from fbt.trader import AlpacaPaperBroker, PaperTrader
        self.trader.on_minute(self.at("09:00"), [self.signal_event("09:00")])
        self.api.fill_entry("o1", 100.00)
        self.trader.on_minute(self.at("09:01"), [])
        again = PaperTrader(self.cfg, lambda rm: AlpacaPaperBroker(rm, http=self.api, env=self.KEYS), ["SPY"],
                            state_path=self.trader.state_path, trades_log=self.trader.trades_log)
        self.assertEqual(again.open.order_id, "o1")
        self.assertEqual(again.risk.day.trades, 1)
        self.assertTrue(again.risk.position_open)
        state = open(self.trader.state_path).read()
        self.assertNotIn("test-secret", state)
        self.assertNotIn("test-id", state)

    def test_live_runner_hands_signal_to_trader(self):
        from fbt.trader import AlpacaPaperBroker, PaperTrader
        df = bullish_scenario_df()
        spy = get_instrument("SPY")
        cfg = scenario_config()
        api = FakeAlpacaTrading()
        trader = PaperTrader(cfg, lambda rm: AlpacaPaperBroker(rm, http=api, env=self.KEYS), ["SPY"],
                             state_path=None, trades_log=None)
        feed = ReplayFeed({"SPY": df})
        box = [ct("2024-03-12 08:58:03").tz_convert("UTC")]

        def clock():
            feed.now = box[0]
            return box[0]
        out = []
        sink = AlertSink(webhook="", echo=lambda t: None)
        sink.emit = lambda ev: out.append(ev)
        scanner = Scanner(cfg, ["SPY"], strategy_factory=lambda s: FixedBiasStrategy(cfg, spy, BULLISH))
        runner = LiveRunner(scanner, feed, sink, clock=clock, trader=trader)
        runner.warm_up()
        for _ in range(4):
            runner.step()
            box[0] += pd.Timedelta(minutes=1)
        kinds = [e.kind for e in out]
        self.assertIn("SIGNAL", kinds)
        self.assertIn("ORDER", kinds)
        body = api.posts()[0][2]
        self.assertEqual(body["stop_loss"]["stop_price"], "102.74")   # leg low 102.75 - 1 cent
        self.assertEqual(body["take_profit"]["limit_price"], "130.00")


class TestAlpacaYahooETF(unittest.TestCase):
    ENV = {"APCA_API_KEY_ID": "test-id-not-real", "APCA_API_SECRET_KEY": "test-secret-not-real"}

    def test_alpaca_requires_env_keys_and_hides_them(self):
        from fbt.feeds import AlpacaFeed
        with self.assertRaisesRegex(DataError, "APCA_API_KEY_ID"):
            AlpacaFeed(env={})
        f = AlpacaFeed(env=self.ENV, http=lambda u, h: {"bars": []})
        self.assertNotIn("test-secret", repr(f))
        self.assertNotIn("test-id", repr(f))

    def test_alpaca_pagination_headers_and_conversion(self):
        from fbt.feeds import AlpacaFeed
        calls = []
        pages = [{"bars": [{"t": "2026-11-06T14:30:00Z", "o": 1, "h": 2, "l": .5, "c": 1.5, "v": 10}],
                  "next_page_token": "abc"},
                 {"bars": [{"t": "2026-11-06T14:31:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.75, "v": 5}],
                  "next_page_token": None}]

        def http(url, headers):
            calls.append((url, headers))
            return pages[len(calls) - 1]
        f = AlpacaFeed(env=self.ENV, http=http)
        df = f.history("SPY", 1)
        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertIn("feed=iex", calls[0][0])
        self.assertIn("page_token=abc", calls[1][0])
        self.assertEqual(calls[0][1]["APCA-API-KEY-ID"], "test-id-not-real")
        self.assertNotIn("test-secret-not-real", calls[0][0])  # secret only in headers, never in the URL

    def test_alpaca_daily_excludes_today(self):
        from fbt.feeds import AlpacaFeed, current_trading_date
        today = current_trading_date(pd.Timestamp.now(tz="UTC"))
        y = pd.Timestamp(today) - pd.Timedelta(days=1)
        bars = [{"t": (pd.Timestamp(d).tz_localize("America/New_York")).tz_convert("UTC").isoformat(),
                 "o": 1, "h": 2, "l": .5, "c": 1.5, "v": 1} for d in (y, pd.Timestamp(today))]
        f = AlpacaFeed(env=self.ENV, http=lambda u, h: {"bars": bars})
        d = f.daily_history("SPY")
        self.assertEqual(list(d.index.date), [y.date()])

    def test_yahoo_rounding_forming_bar_and_daily(self):
        from fbt import yahoo
        now = pd.Timestamp("2026-11-06 18:40:30", tz="UTC")
        idx = pd.date_range("2026-11-06 18:37", periods=4, freq="1min", tz="UTC")
        m1 = pd.DataFrame({"Open": np.float32([3969.9, 3970.0, 3970.1, 3970.2]).astype(float),
                           "High": np.float32([3970.0, 3970.1, 3970.2, 3970.3]).astype(float),
                           "Low": np.float32([3969.8, 3969.9, 3970.0, 3970.1]).astype(float),
                           "Close": np.float32([3970.0, 3970.1, 3970.2, 3970.3]).astype(float),
                           "Adj Close": 0.0, "Volume": 1.0}, index=idx)
        daily = pd.DataFrame({"Open": [1.0, 1.0], "High": [2.0, 2.0], "Low": [.5, .5], "Close": [1.5, 1.5],
                              "Volume": [1.0, 1.0]}, index=pd.to_datetime(["2026-11-05", "2026-11-06"]))

        class FakeYF:
            @staticmethod
            def download(ticker, **kw):
                return m1 if kw.get("interval") == "1m" else daily
        orig = yahoo._yf
        yahoo._yf = lambda: FakeYF
        try:
            df = yahoo.download_1m("GC", days=1, now=now, pause=0)
            self.assertEqual(len(df), 3)                       # 18:40 bar still forming -> dropped
            self.assertEqual(df["open"].iloc[0], 3969.9)       # float32 noise removed, exact tick
            d = yahoo.download_daily("GC", now=now)
            # 12:40 CT on Nov 6: the Nov 6 session is still trading -> only Nov 5 is a finished daily bar
            self.assertEqual(list(d.index.date), [pd.Timestamp("2026-11-05").date()])
            big = m1.copy()
            big[["Open", "High", "Low", "Close"]] = 70_000.0
            with self.assertRaises(DataError):
                yahoo._normalise(big, "X")
        finally:
            yahoo._yf = orig

    def test_etf_specs_sizing_and_costs(self):
        spy = get_instrument("SPY")
        self.assertEqual((spy.unit, spy.point_value, spy.max_units, spy.min_stop_ticks), ("share", 1.0, 2000, 10))
        self.assertGreaterEqual(costs_mod.DEFAULT_COMMISSION_PER_SIDE["SPY"], 0.005)
        cfg = default_config()
        rm = RiskManager(cfg, CostModel.default("SPY"))
        rm.new_day("2026-11-06")
        t = ns("2026-11-06 09:00")
        # 250$ risk / (0.50 stop + 0.01 commission + 0.02 slippage) = 471 shares
        self.assertEqual(rm.size(make_signal(entry=500.0, stop=499.5, target=501.0, t=t)), 471)
        rm.equity = 10_000_000
        self.assertEqual(rm.size(make_signal(entry=500.0, stop=499.5, target=501.0, t=t)), 2000)
        r = rm.approve(make_signal(entry=500.0, stop=499.95, target=501.0, t=t), t, 540, False)
        self.assertEqual(r.reason, "STOP_TOO_TIGHT")              # 5 cents < 10-tick minimum
        self.assertEqual(costs_mod.SMT_PAIRS["SPY"], "QQQ")

    def test_daily_data_loading_errors(self):
        from fbt.backtest import load_daily
        self.assertIsNone(load_daily(None))
        with self.assertRaisesRegex(DataError, "not found"):
            load_daily("/nonexistent/daily.csv")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            random_walk_df(days=3, seed=2).to_csv(p, index_label="timestamp")
            code = cli_main(["--data", p, "--daily-data", os.path.join(d, "nope.csv"), "--symbol", "ES",
                             "--out", os.path.join(d, "out")])
            self.assertEqual(code, 2)

    def test_scanner_alpaca_without_keys_exits_cleanly(self):
        saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("APCA_")}
        try:
            self.assertEqual(scanner_mod.main(["live", "--feed", "alpaca"]), 2)
        finally:
            os.environ.update(saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)

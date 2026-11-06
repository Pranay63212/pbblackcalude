"""Event-driven backtester over 1-minute bars.

Loop for every completed 1m bar i (time = its close):
  1. the broker processes bar i (fills orders placed at bar i-1's close, exits);
  2. if flat and inside the session, the strategy evaluates a MarketView at
     bar i's close; an approved signal becomes an order acting from bar i+1.

Usage:
  python -m fbt.backtest --data data/ES_1m.csv.gz --symbol ES \
      [--smt-data data/NQ_1m.csv.gz --smt-symbol NQ] [--start 2010-01-01] [--end 2025-01-01] \
      [--oos-start 2020-01-01] [--out results]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .costs import CostModel, get_instrument
from .data import (DataError, MultiTimeframeData, data_quality_report, load_ohlcv, ns_to_chicago,
                   parse_hhmm)
from .execution import PaperBroker, Trade
from .risk import Approval, RiskManager
from .strategy import Decision, PBBlakeStrategy

DISCLAIMER = ("Research / paper simulation only. Results are hypothetical, include modelled costs, and are "
              "not evidence of profitability unless confirmed out-of-sample with statistical significance.")


@dataclass
class BacktestResult:
    trades: List[Trade]
    decisions: List[Decision]
    rejections: List[dict]
    equity_curve: pd.DataFrame
    stats: dict
    oos_stats: Optional[dict] = None
    in_sample_stats: Optional[dict] = None
    data_report: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, data: MultiTimeframeData, config: dict, smt_data: Optional[MultiTimeframeData] = None,
                 costs: Optional[CostModel] = None, record_all_decisions: bool = False,
                 strategy: Optional[PBBlakeStrategy] = None):
        self.data = data
        self.smt = smt_data
        self.cfg = config
        self.inst = get_instrument(data.symbol)
        self.costs = costs or CostModel.default(data.symbol)
        self.risk = RiskManager(config, self.costs)
        self.broker = PaperBroker(self.costs, self.risk, data.symbol)
        self.strategy = strategy or PBBlakeStrategy(config, self.inst)
        self.record_all = record_all_decisions
        sess = config["execution"]["session"]
        self.s_start = parse_hhmm(sess["start"])
        self.s_end = parse_hhmm(sess["end"])
        self.flatten = parse_hhmm(config["backtest"]["flatten_time"])
        if self.flatten <= self.s_end:
            raise ValueError("flatten_time must be after the session end.")
        self.expiry_ns = int(config["strategy"]["limit_order_expiry_minutes"] * 60 * 1e9)

    def run(self, progress: bool = False) -> BacktestResult:
        d = self.data
        m1 = d.m1
        n = len(m1)
        minute = d.close_minute_of_day
        dates = d.close_local_date
        in_sess = (minute >= self.s_start) & (minute <= self.s_end)
        decisions: List[Decision] = []
        rejections: List[dict] = []
        t0 = time.time()
        last_day = None
        session_end_ns = None
        days_seen = 0
        for i in range(n):
            if not in_sess[i] and self.broker.flat:
                continue
            day = dates[i]
            if day != last_day:
                self.risk.new_day(day)
                last_day = day
                end_local = pd.Timestamp(day) + pd.Timedelta(minutes=self.s_end)
                session_end_ns = end_local.tz_localize("America/Chicago").tz_convert("UTC").value
                days_seen += 1
                if progress and days_seen % 250 == 0:
                    print(f"  {day}  trades={len(self.broker.closed)}  {time.time() - t0:.0f}s", file=sys.stderr)
            now = int(m1.close_time[i])
            if not self.broker.flat:
                self.broker.on_bar(float(m1.open[i]), float(m1.high[i]), float(m1.low[i]), float(m1.close[i]),
                                   int(m1.open_time[i]), now, flatten_now=bool(minute[i] >= self.flatten))
            if not (in_sess[i] and self.broker.flat):
                continue
            if self.risk.day_blocked_reason() is not None:
                continue
            view = d.view(now)
            smt_view = self.smt.view(now) if self.smt is not None else None
            dec = self.strategy.evaluate(view, smt_view, self.risk.trade_context())
            if self.record_all or dec.status != "WAIT":
                decisions.append(dec)
            if dec.signal is None:
                continue
            res = self.risk.approve(dec.signal, now, int(minute[i]), has_position=not self.broker.flat)
            if not isinstance(res, Approval):
                rejections.append({"time": now, "reason": res.reason, "direction": dec.signal.direction,
                                   "entry": dec.signal.entry_price, "stop": dec.signal.stop})
                continue
            expiry = None
            if dec.signal.entry_method != "IFVG_BODY_CLOSE":
                expiry = min(now + self.expiry_ns, session_end_ns)
            self.broker.submit(res, expiry)
        if n:
            self.broker.force_close(float(m1.close[-1]), int(m1.close_time[-1]))
        trades = list(self.broker.closed)
        eq = equity_curve(trades, self.cfg["risk"]["initial_equity"])
        stats = compute_stats(trades, self.cfg["risk"]["initial_equity"])
        return BacktestResult(trades, decisions, rejections, eq, stats)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def equity_curve(trades: List[Trade], initial: float) -> pd.DataFrame:
    rows = [{"time": None, "equity": initial}]
    eq = initial
    for t in trades:
        eq += t.net_pnl
        rows.append({"time": str(ns_to_chicago(t.exit_time)), "equity": eq})
    df = pd.DataFrame(rows)
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] - df["peak"]
    return df


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    logp = lambda i: (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)  # noqa: E731
                      + i * math.log(p) + (n - i) * math.log(1 - p))
    pk = logp(k)
    total = 0.0
    for i in range(n + 1):
        li = logp(i)
        if li <= pk + 1e-9:
            total += math.exp(li)
    return min(1.0, total)


def compute_stats(trades: List[Trade], initial: float) -> dict:
    n = len(trades)
    base = {"total_trades": n, "disclaimer": DISCLAIMER}
    if n == 0:
        base.update({"win_rate": None, "profit_factor": None, "net_pnl": 0.0, "max_drawdown": 0.0,
                     "max_drawdown_pct": 0.0, "expectancy": None, "average_r": None,
                     "significance": "No trades: nothing can be concluded."})
        return base
    pnl = np.array([t.net_pnl for t in trades])
    r = np.array([t.r_multiple for t in trades])
    wins = pnl > 0
    gp = float(pnl[wins].sum())
    gl = float(-pnl[~wins].sum())
    eq = initial + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[initial], eq]))[1:]
    dd = eq - peak
    dd_pct = dd / peak * 100
    streak = best = 0
    for w in wins:
        streak = 0 if w else streak + 1
        best = max(best, streak)
    mean_r = float(r.mean())
    sd_r = float(r.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean_r / (sd_r / math.sqrt(n)) if n > 1 and sd_r > 0 else float("nan")
    p_mean = math.erfc(abs(t_stat) / math.sqrt(2)) if not math.isnan(t_stat) else float("nan")
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    base.update({
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else None),
        "net_pnl": float(pnl.sum()),
        "total_commission": float(sum(t.commission for t in trades)),
        "total_slippage_cost": float(sum(t.slippage_cost for t in trades)),
        "expectancy": float(pnl.mean()),
        "average_r": mean_r,
        "median_r": float(np.median(r)),
        "std_r": sd_r,
        "average_win": float(pnl[wins].mean()) if wins.any() else 0.0,
        "average_loss": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "max_drawdown": float(dd.min()),
        "max_drawdown_pct": float(dd_pct.min()),
        "max_consecutive_losses": int(best),
        "final_equity": float(eq[-1]),
        "long_trades": int(sum(1 for t in trades if t.direction == "BULLISH")),
        "short_trades": int(sum(1 for t in trades if t.direction != "BULLISH")),
        "exit_reasons": reasons,
        "t_stat_mean_r": t_stat,
        "p_value_mean_r": p_mean,
        "p_value_win_rate_vs_50pct": _binom_two_sided_p(int(wins.sum()), n),
    })
    if n < 30:
        base["significance"] = f"Only {n} trades: far too few for any statistical conclusion."
    elif not (p_mean < 0.05):
        base["significance"] = "Mean R is not statistically different from zero (p >= 0.05)."
    else:
        base["significance"] = ("Mean R differs from zero at p < 0.05 on THIS sample. Only meaningful if this "
                                "sample is out-of-sample and parameters were not tuned on it.")
    return base


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _ts(ns):
    return str(ns_to_chicago(ns)) if ns else ""


def trades_frame(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        d = t.to_dict()
        kl, fv, leg, bias = d.pop("key_level"), d.pop("ifvg"), d.pop("leg"), d.pop("bias")
        for k in ("signal_time", "entry_time", "exit_time"):
            d[k] = _ts(d[k])
        d.update({
            "key_level_type": kl.get("type"), "key_level_tf": kl.get("timeframe"),
            "key_level_upper": kl.get("upper_boundary"), "key_level_lower": kl.get("lower_boundary"),
            "key_level_touch_time": _ts(kl.get("touch_time")), "key_level_cisd_confirmed": kl.get("cisd_confirmed"),
            "key_level_fvg_confluence": kl.get("fvg_confluence"), "key_level_smt": kl.get("smt_confluence"),
            "ifvg_tf": fv.get("timeframe"), "ifvg_upper": fv.get("upper"), "ifvg_lower": fv.get("lower"),
            "leg_start_price": leg.get("start_price"), "leg_end_price": leg.get("end_price"),
            "leg_start_time": _ts(leg.get("start_time")), "leg_end_time": _ts(leg.get("end_time")),
            "bias_daily": bias.get("daily_bias"), "bias_4h": bias.get("four_hour_bias"),
            "bias_1h": bias.get("one_hour_bias"), "bias_15m_confirmation": bias.get("fifteen_minute_confirmation"),
            "smt_confirmation": bias.get("smt_confirmation"), "draw_on_liquidity": bias.get("draw_on_liquidity"),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def write_reports(result: BacktestResult, out_dir: str, config: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    trades_frame(result.trades).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    result.equity_curve.to_csv(os.path.join(out_dir, "equity.csv"), index=False)
    dec = pd.DataFrame([{"time": _ts(x.time), "status": x.status, "reason": x.reason, "bias": x.bias_direction,
                         "key_levels": "|".join(x.key_level_ids)} for x in result.decisions])
    dec.to_csv(os.path.join(out_dir, "decisions.csv"), index=False)
    pd.DataFrame([{**r, "time": _ts(r["time"])} for r in result.rejections]).to_csv(
        os.path.join(out_dir, "risk_rejections.csv"), index=False)
    summary = {"all": result.stats, "in_sample": result.in_sample_stats, "out_of_sample": result.oos_stats,
               "data": result.data_report, "config": config}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return out_dir


def format_stats(stats: dict, title: str = "ALL TRADES") -> str:
    keys = ["total_trades", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "expectancy", "average_r",
            "max_drawdown", "max_drawdown_pct", "max_consecutive_losses", "total_commission",
            "total_slippage_cost", "t_stat_mean_r", "p_value_mean_r", "p_value_win_rate_vs_50pct"]
    lines = [f"== {title} =="]
    for k in keys:
        if k in stats:
            v = stats[k]
            lines.append(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    lines.append(f"  significance: {stats.get('significance')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_daily(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Daily bars CSV indexed by trading date (as written by fbt.yahoo)."""
    if not path:
        return None
    if not os.path.exists(path):
        raise DataError(f"Daily data file not found: {path}")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:
        raise DataError(f"Could not parse daily data {path}: {exc}") from exc
    df.columns = [str(c).lower() for c in df.columns]
    return df


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import default_config

    ap = argparse.ArgumentParser(description="PB Blake strategy backtest (paper / research only)")
    ap.add_argument("--data", required=True, help="1-minute OHLCV file (.csv, .csv.gz, .dbn.zst)")
    ap.add_argument("--symbol", required=True, help="ES, NQ, MES or MNQ")
    ap.add_argument("--data-symbol", default=None, help="value of the file's 'symbol' column to select")
    ap.add_argument("--smt-data", default=None, help="optional second instrument for SMT (e.g. NQ when trading ES)")
    ap.add_argument("--smt-symbol", default=None)
    ap.add_argument("--daily-data", default=None, help="optional daily bars (date,open,high,low,close,volume) "
                    "for warm-up when the 1-minute file is short, e.g. from python -m fbt.yahoo")
    ap.add_argument("--smt-daily-data", default=None)
    ap.add_argument("--tz", default="UTC", help="timezone of naive timestamps in the file")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--oos-start", default=None, help="date splitting in-sample / out-of-sample stats")
    ap.add_argument("--out", default="results")
    ap.add_argument("--all-decisions", action="store_true", help="log every evaluation, not only non-WAIT")
    args = ap.parse_args(argv)

    cfg = default_config()
    try:
        df = load_ohlcv(args.data, symbol=args.data_symbol, tz=args.tz, start=args.start, end=args.end)
        data = MultiTimeframeData(df, args.symbol, daily_history=load_daily(args.daily_data))
        smt = None
        if args.smt_data:
            sdf = load_ohlcv(args.smt_data, tz=args.tz, start=args.start, end=args.end)
            smt = MultiTimeframeData(sdf, args.smt_symbol or "NQ", daily_history=load_daily(args.smt_daily_data))
    except DataError as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return 2
    report = data_quality_report(df)
    print(f"Loaded {report['bars']} bars {report['first']} -> {report['last']}", file=sys.stderr)
    bt = Backtester(data, cfg, smt, record_all_decisions=args.all_decisions)
    res = bt.run(progress=True)
    res.data_report = report
    if args.oos_start:
        cut = pd.Timestamp(args.oos_start, tz="America/Chicago").tz_convert("UTC").value
        ins = [t for t in res.trades if t.entry_time < cut]
        oos = [t for t in res.trades if t.entry_time >= cut]
        res.in_sample_stats = compute_stats(ins, cfg["risk"]["initial_equity"])
        res.oos_stats = compute_stats(oos, cfg["risk"]["initial_equity"])
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out = write_reports(res, os.path.join(args.out, f"{args.symbol}_{stamp}"), cfg)
    print(format_stats(res.stats))
    if res.oos_stats:
        print(format_stats(res.in_sample_stats, "IN-SAMPLE"))
        print(format_stats(res.oos_stats, "OUT-OF-SAMPLE"))
    print(DISCLAIMER)
    print(f"Reports written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

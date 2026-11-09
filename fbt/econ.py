"""Economic calendar for the optional news filter.

Sources:
* Forex Factory weekly feed (nfs.faireconomy.media/ff_calendar_thisweek.json):
  this week's events with impact ratings. Free, no key, current week only.
* Federal Reserve FOMC calendar page: meeting dates for past and future
  years. The statement is at 14:00 ET on the last day of each meeting
  (press conference 14:30 ET).
* A local CSV (data/econ_calendar.csv) that the live scanner extends every
  week, so a real history builds up for future backtests.

Nothing is invented: a backtest only knows the events that are in the CSV.
The calendar records which days it actually covers; when the news filter is
on and today is not covered, the RiskManager refuses new trades instead of
assuming a quiet day.

Usage:
  python -m fbt.econ --out data/econ_calendar.csv      # this week + all FOMC dates
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from html import unescape
from typing import Iterable, List, Optional, Set

import pandas as pd

from .net import ssl_context
from .data import CHICAGO, NS_PER_MIN

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
UA = {"User-Agent": "Mozilla/5.0 (pbblackcalude research scanner)"}
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                      "Nov", "Dec"], 1)}  # matched on the first three letters


class CalendarError(RuntimeError):
    pass


@dataclass(frozen=True)
class EconEvent:
    time_ns: int        # UTC
    title: str
    country: str
    impact: str         # High / Medium / Low / Holiday

    @property
    def local(self) -> pd.Timestamp:
        return pd.Timestamp(self.time_ns, tz="UTC").tz_convert(CHICAGO)


class EconCalendar:
    def __init__(self, events: Iterable[EconEvent] = (), covered_days: Optional[Set] = None,
                 countries=("USD",), impacts=("High",)):
        self.countries = tuple(countries)
        self.impacts = tuple(impacts)
        self.events: List[EconEvent] = []
        # None = coverage unknown (e.g. a backtest CSV); a set = exactly these CT dates are covered
        self.covered_days: Optional[Set] = None if covered_days is None else set(covered_days)
        self.update(events)

    def update(self, events: Iterable[EconEvent], covered_days: Optional[Set] = None) -> None:
        seen = {(e.time_ns, e.title, e.country) for e in self.events}
        for e in events:
            if (e.time_ns, e.title, e.country) not in seen:
                self.events.append(e)
                seen.add((e.time_ns, e.title, e.country))
        self.events.sort(key=lambda e: e.time_ns)
        if covered_days is not None:
            self.covered_days = (self.covered_days or set()) | set(covered_days)

    def relevant(self) -> List[EconEvent]:
        return [e for e in self.events if e.country in self.countries and e.impact in self.impacts]

    def covers(self, day) -> bool:
        return self.covered_days is None or day in self.covered_days

    def in_blackout(self, now_ns: int, minutes_before: float, minutes_after: float) -> Optional[EconEvent]:
        """The relevant event whose [time - before, time + after] window contains now_ns."""
        for e in self.relevant():
            if e.time_ns - minutes_before * NS_PER_MIN <= now_ns <= e.time_ns + minutes_after * NS_PER_MIN:
                return e
        return None

    def on_day(self, day) -> List[EconEvent]:
        return [e for e in self.relevant() if e.local.date() == day]

    # --- persistence
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"time_utc": pd.Timestamp(e.time_ns, tz="UTC").isoformat(), "title": e.title,
                              "country": e.country, "impact": e.impact} for e in self.events])

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.to_frame().to_csv(path, index=False)


def load_csv(path: str) -> List[EconEvent]:
    if not os.path.exists(path):
        raise CalendarError(f"calendar file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in ("time_utc", "title", "country", "impact") if c not in df.columns]
    if missing:
        raise CalendarError(f"{path}: missing columns {missing}")
    out = []
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["time_utc"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        out.append(EconEvent(int(ts.value), str(r["title"]), str(r["country"]), str(r["impact"])))
    return out


# ---------------------------------------------------------------------------
# Forex Factory weekly feed
# ---------------------------------------------------------------------------
def parse_faireconomy(items: list):
    """-> (events, covered CT dates). Events without a clock time (all-day) are kept at their date."""
    events, days = [], set()
    for it in items or []:
        try:
            ts = pd.Timestamp(it["date"])
        except Exception:
            continue
        ts = ts.tz_localize("America/New_York") if ts.tzinfo is None else ts
        events.append(EconEvent(int(ts.tz_convert("UTC").value), str(it.get("title", "")).strip(),
                                str(it.get("country", "")).strip(), str(it.get("impact", "")).strip()))
        days.add(ts.tz_convert(CHICAGO).date())
    if days:
        # the feed is one full Sunday-Saturday week: it covers every day of that week
        lo = pd.Timestamp(min(days))
        sunday = lo - pd.Timedelta(days=(lo.weekday() + 1) % 7)
        days |= {(sunday + pd.Timedelta(days=k)).date() for k in range(7)}
    return events, days


def _get(url: str, http=None) -> str:
    if http is not None:
        return http(url)
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20, context=ssl_context()) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise CalendarError(f"could not download {url}: {type(exc).__name__}") from None


def fetch_faireconomy(http=None):
    try:
        return parse_faireconomy(json.loads(_get(FF_URL, http)))
    except ValueError as exc:
        raise CalendarError(f"economic calendar feed was not valid JSON: {exc}") from None


# ---------------------------------------------------------------------------
# Federal Reserve FOMC calendar
# ---------------------------------------------------------------------------
def parse_fomc_html(text: str) -> List[EconEvent]:
    """Statement 14:00 ET and press conference 14:30 ET on each meeting's last day."""
    out = []
    parts = re.split(r"(\d{4}) FOMC Meetings", text)
    for k in range(1, len(parts) - 1, 2):
        year, block = int(parts[k]), parts[k + 1]
        rows = re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<',
                          block, re.S)
        for month_txt, date_txt in rows:
            month_txt, date_txt = unescape(month_txt).strip(), unescape(date_txt).strip()
            last_month = month_txt.split("/")[-1].strip()[:3].title()
            days = re.findall(r"\d+", date_txt)
            if last_month not in MONTHS or not days or "notation" in date_txt.lower():
                continue
            day = int(days[-1])
            for hhmm, title in (("14:00", "FOMC Statement"), ("14:30", "FOMC Press Conference")):
                ts = pd.Timestamp(f"{year}-{MONTHS[last_month]:02d}-{day:02d} {hhmm}", tz="America/New_York")
                out.append(EconEvent(int(ts.tz_convert("UTC").value), title, "USD", "High"))
    return out


def fetch_fomc(http=None) -> List[EconEvent]:
    events = parse_fomc_html(_get(FOMC_URL, http))
    if not events:
        raise CalendarError("could not find FOMC meeting dates on the Federal Reserve page")
    return events


def refresh_calendar(cal: EconCalendar, cache_path: Optional[str] = None, http=None) -> int:
    """Fetch this week's events (+ FOMC dates), merge into `cal`, extend the CSV cache."""
    events, days = fetch_faireconomy(http)
    try:
        events = events + fetch_fomc(http)
    except CalendarError:
        pass  # this week's feed already lists FOMC days; the Fed page is a bonus
    before = len(cal.events)
    cal.update(events, covered_days=days)
    if cache_path:
        if os.path.exists(cache_path):
            try:
                cal.update(load_csv(cache_path))
            except CalendarError:
                pass
        cal.save_csv(cache_path)
    return len(cal.events) - before


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the economic calendar (this week + FOMC dates)")
    ap.add_argument("--out", default="data/econ_calendar.csv")
    args = ap.parse_args(argv)
    cal = EconCalendar()
    try:
        refresh_calendar(cal, args.out)
    except CalendarError as exc:
        print(f"CALENDAR ERROR: {exc}", file=sys.stderr)
        return 2
    for e in cal.relevant():
        if e.local >= pd.Timestamp.now(tz=CHICAGO) - pd.Timedelta(days=7):
            print(f"{e.local:%a %b %d %H:%M} CT  {e.title}")
    print(f"Saved {len(cal.events)} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

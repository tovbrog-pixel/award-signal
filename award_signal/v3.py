"""Version 3 trading rule: buy after the budget release, sell on 1 July.

Rule frozen 2026-11-27 before any holdout data was collected. It turns the
timing check's one strong window into a trade:

  universe   companies exposed to a rising area (rule v1), exposure measured from
             awards public before the release date (version 2 matching)
  entry      open of the 6th session on or after the line-item release date (R+5)
  exit       first open after 1 July of the year before the fiscal year
  skip       a year whose window is shorter than 10 sessions (FY2026: released 25 June)
  band       market cap $100M to $5B at the close before entry
  costs      the same round-trip cost model as every other test
  pass       rising-area companies beat at least 95% of random-area draws, and at
             least 60% of years beat IWM after costs

Holdout. The rule was found on FY2017-FY2027. It is judged on request years
FY2009-FY2016, which no return test has used. Those years differ in two ways,
fixed in advance:
  - award history comes from USAspending (DoD prime awards of $7.5M and up,
    public 90 days after the action date) plus DoD SBIR awards, because the DoD
    announcement archive the main tests used starts in 2014
  - many small companies filed no share counts with SEC before mid-2011; when
    none was filed before entry, the first count filed within 18 months after
    is used for the market-cap band only (it never touches returns)
Known limit, found when collecting the data: USAspending's award search starts
1 October 2007, so FY2009 (released February 2008) sees only four months of
award history and FY2010 about nineteen months, against three years elsewhere.
Known limit: price data comes from Yahoo, which lacks many companies that
delisted since; the holdout therefore leans toward survivors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sec import _entries
from .stages import exposure, random_area_placebo
from .study import round_trip_cost

RELEASE_DATES = {
    2009: "2008-02-04", 2010: "2009-05-07", 2011: "2010-02-01", 2012: "2011-02-14",
    2013: "2012-02-13", 2014: "2013-04-10", 2015: "2014-03-04", 2016: "2015-02-02",
    2017: "2016-02-09", 2018: "2017-05-23", 2019: "2018-02-12", 2020: "2019-03-12",
    2021: "2020-02-10", 2022: "2021-05-28", 2023: "2022-03-28", 2024: "2023-03-13",
    2025: "2024-03-11", 2026: "2025-06-25", 2027: "2026-04-21",
}
HOLDOUT_FYS = range(2009, 2017)
MIN_SESSIONS = 10


def shares_for_band(facts: dict, date: pd.Timestamp) -> float | None:
    """Latest share count filed on or before `date`; failing that, the first one filed
    within 18 months after (holdout fallback, market-cap band only)."""
    before, after = None, None
    for e in _entries(facts, "dei", "EntityCommonStockSharesOutstanding", "shares"):
        filed = pd.Timestamp(e["filed"])
        key = (filed, pd.Timestamp(e["end"]), float(e["val"]))
        if filed <= date:
            if before is None or key[:2] > before[:2]:
                before = key
        elif filed <= date + pd.DateOffset(months=18):
            if after is None or key[:2] < after[:2]:
                after = key
    return (before or after or (None, None, None))[2]


def trades(signals: pd.DataFrame, awards: pd.DataFrame, fundamentals, provider, cfg: dict,
           years=None, log=print) -> pd.DataFrame:
    sig_cfg = cfg["signal"]
    bench = provider.get(cfg["prices"]["benchmark"])
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for rfy, g in signals.groupby("request_fy"):
        if int(rfy) not in RELEASE_DATES or (years is not None and int(rfy) not in years):
            continue
        rel = pd.Timestamp(RELEASE_DATES[int(rfy)])
        exit_date = pd.Timestamp(f"{int(rfy) - 1}-07-01")
        rising = set(g.loc[g["rising"], "area"])
        exp = exposure(awards, rel, cfg)
        exp = exp[exp["exposed"]]
        log(f"  FY{rfy} (released {rel.date()}): {len(rising)} rising areas, {exp['key'].nunique()} exposed companies")
        for r in exp.itertuples(index=False):
            rec = {"request_fy": rfy, "release_date": rel, "area": r.area, "rising": r.area in rising,
                   "ticker": r.ticker, "cik": r.cik, "status": "", "entry_date": pd.NaT, "exit_date": pd.NaT,
                   "market_cap": np.nan, "ret": np.nan, "bench_ret": np.nan, "cost": np.nan, "net_excess": np.nan}
            if pd.notna(r.delisting_date) and r.delisting_date <= rel:
                rec["status"] = "delisted before release"
                rows.append(rec)
                continue
            if r.ticker not in cache:
                px = provider.get(r.ticker)
                cache[r.ticker] = px.dropna(subset=["open", "close"]) if px is not None and len(px) > 30 else None
            px = cache[r.ticker]
            if px is None:
                rec["status"] = "no price data"
                rows.append(rec)
                continue
            px = px[px["open"] > 0]
            d = px.index
            R = int(d.searchsorted(rel, side="left"))
            i, j = R + 5, int(d.searchsorted(exit_date, side="right"))
            if R < 21 or j >= len(d):
                rec["status"] = "outside price history"
                rows.append(rec)
                continue
            if j - i < MIN_SESSIONS:
                rec["status"] = "window too short"
                rows.append(rec)
                continue
            facts = fundamentals.facts(r.cik) if pd.notna(r.cik) else {}
            shares = shares_for_band(facts, d[i])
            mcap = (shares or np.nan) * px["close"].iloc[i - 1]
            rec.update(entry_date=d[i], exit_date=d[j], market_cap=mcap)
            if np.isnan(mcap):
                rec["status"] = "market cap unknown"
            elif not (sig_cfg["min_market_cap"] <= mcap <= sig_cfg["max_market_cap"]):
                rec["status"] = "outside market-cap band"
            else:
                o = px["open"].to_numpy(float)
                b_open = bench["open"].reindex(d).ffill().to_numpy(float)
                adv = float((px["close"] * px["volume"]).iloc[max(0, i - 20):i].mean())
                cost = round_trip_cost(adv, cfg)
                ret, bret = o[j] / o[i] - 1, b_open[j] / b_open[i] - 1
                rec.update(status="traded", ret=ret, bench_ret=bret, cost=cost, net_excess=ret - bret - cost)
            rows.append(rec)
    return pd.DataFrame(rows)


def judge(rows: pd.DataFrame, signals: pd.DataFrame, cfg: dict) -> dict:
    real = rows[(rows["status"] == "traded") & rows["rising"]].drop_duplicates(["request_fy", "ticker"])
    by_year = real.groupby("request_fy")["net_excess"].mean()
    placebo = random_area_placebo(rows, signals, cfg) if len(real) else {"companies": 0}
    need = int(np.ceil(cfg["stages"]["min_positive_share"] * len(by_year))) if len(by_year) else 0
    yearly = {"years": int(len(by_year)), "years_beating_iwm": int((by_year > 0).sum()), "years_required": need,
              "passed": bool(len(by_year) and (by_year > 0).sum() >= need)}
    return {"positions": int(len(real)),
            "mean_net_excess": float(real["net_excess"].mean()) if len(real) else None,
            "by_year": by_year.round(4).to_dict(), "yearly_test": yearly, "placebo": placebo,
            "status_counts": rows[rows["rising"]]["status"].value_counts().to_dict(),
            "passed": bool(yearly["passed"] and placebo.get("passes_95pct", False))}

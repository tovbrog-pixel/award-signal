"""Stages 2 and 3: exposed companies for each rising area, and the 12-month backtest.

Stage 2. Each DoD award announcement names a company and describes the work.
Recipients are matched to public companies by exact normalized name (SEC
ticker list, plus delisted companies with a known CIK). Award descriptions are
tagged with the frozen area keywords. At a signal date D, a company is exposed
to an area when, over the 3 years before D, at least `exposure_share` of its
announced award dollars carried that area's tag and the area dollars reach
`exposure_min_dollars`. Only information dated before D is used.

Stage 3. On each signal date, buy every exposed company in the $100M-$5B
market-cap band for each rising area, at the next session's open, and hold
`hold_days` sessions. Results are compared with IWM, and with a placebo that
draws the same number of areas at random from all areas in the same year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .budget import tag_areas
from .matching import normalize
from .study import TickerFrame, round_trip_cost


def company_awards(raw: pd.DataFrame, index: dict[str, dict], areas: dict[str, list[str]]) -> pd.DataFrame:
    """Single-company award paragraphs matched to public companies, with area tags."""
    df = raw[~raw["multi_award"].astype(bool)].copy()
    df["key"] = df["recipient_name"].map(normalize)
    df = df[df["key"].isin(index.keys())]
    df["action_date"] = pd.to_datetime(df["action_date"])
    df["amount"] = pd.to_numeric(df["face_value"], errors="coerce").fillna(0)
    df["areas"] = tag_areas(df["description"].fillna(""), areas)
    df["ticker"] = df["key"].map(lambda k: index[k]["ticker"])
    df["cik"] = df["key"].map(lambda k: index[k]["cik"])
    df["delisting_date"] = pd.to_datetime(df["key"].map(lambda k: index[k]["delisting_date"]), errors="coerce")
    return df[["key", "ticker", "cik", "delisting_date", "action_date", "amount", "areas", "recipient_name"]]


def exposure(awards: pd.DataFrame, date: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Company x area exposure from awards dated in the 3 years before `date`."""
    s = cfg["stages"]
    seen = awards["available_date"] if "available_date" in awards.columns else awards["action_date"]
    w = awards[(awards["action_date"] < date) & (seen < date) &
               (awards["action_date"] >= date - pd.DateOffset(years=int(s["lookback_years"])))]
    if w.empty:
        return pd.DataFrame(columns=["key", "ticker", "cik", "delisting_date", "area", "area_dollars",
                                     "total_dollars", "share", "exposed"])
    totals = w.groupby("key")["amount"].sum()
    ex = w.explode("areas").dropna(subset=["areas"])
    by = ex.groupby(["key", "areas"])["amount"].sum().reset_index().rename(
        columns={"areas": "area", "amount": "area_dollars"})
    by["total_dollars"] = by["key"].map(totals)
    by["share"] = by["area_dollars"] / by["total_dollars"].where(by["total_dollars"] > 0)
    first = w.drop_duplicates("key").set_index("key")
    for col in ("ticker", "cik", "delisting_date"):
        by[col] = by["key"].map(first[col])
    by["exposed"] = (by["share"] >= s["exposure_share"]) & (by["area_dollars"] >= s["exposure_min_dollars"])
    return by


def build_positions(signals: pd.DataFrame, awards: pd.DataFrame, fundamentals, provider, cfg: dict,
                    log=print) -> pd.DataFrame:
    """One row per (signal year, area, company): every area, rising or not (the placebo needs all)."""
    s, sig_cfg = cfg["stages"], cfg["signal"]
    h = int(s["hold_days"])
    bench = provider.get(cfg["prices"]["benchmark"])
    bench_last = bench.index[-1]
    frames: dict[str, TickerFrame | None] = {}
    rows = []
    for rfy, g in signals.groupby("request_fy"):
        date = pd.Timestamp(g["signal_date"].iloc[0])
        rising = set(g.loc[g["rising"], "area"])
        exp = exposure(awards, date, cfg)
        exp = exp[exp["exposed"]]
        log(f"  FY{rfy} (signal {date.date()}): {len(rising)} rising areas, "
            f"{exp['key'].nunique()} exposed companies across all areas")
        for r in exp.itertuples(index=False):
            rec = {"request_fy": rfy, "signal_date": date, "area": r.area, "rising": r.area in rising,
                   "ticker": r.ticker, "cik": r.cik, "share": r.share, "area_dollars": r.area_dollars,
                   "status": "", "entry_date": pd.NaT, "market_cap": np.nan, "ret": np.nan,
                   "bench_ret": np.nan, "cost": np.nan, "net_excess": np.nan}
            if pd.notna(r.delisting_date) and r.delisting_date <= date:
                rec["status"] = "delisted before signal"
                rows.append(rec)
                continue
            if r.ticker not in frames:
                px = provider.get(r.ticker)
                frames[r.ticker] = TickerFrame(px, bench, h) if px is not None and len(px) > 30 else None
            tf = frames[r.ticker]
            if tf is None:
                rec["status"] = "no price data"
                rows.append(rec)
                continue
            i = tf.entry_index(date, 0)
            if i is None or i < 21:
                rec["status"] = "outside price history"
                rows.append(rec)
                continue
            shares = fundamentals.shares_asof(r.cik, date) if pd.notna(r.cik) else None
            mcap = (shares or np.nan) * tf.close[i - 1]
            rec.update(entry_date=tf.dates[i], market_cap=mcap)
            if np.isnan(mcap):
                rec["status"] = "market cap unknown"
            elif not (sig_cfg["min_market_cap"] <= mcap <= sig_cfg["max_market_cap"]):
                rec["status"] = "outside market-cap band"
            elif np.isnan(tf.ret[i]):
                stopped = tf.dates[-1] < bench_last - pd.Timedelta(days=10)
                rec["status"] = "stopped trading during hold" if stopped else "hold not finished"
                # a hold still running gets its return to date, labeled as such
                if not stopped:
                    last = len(tf.dates) - 1
                    b_open = bench["open"].reindex(tf.dates).ffill().to_numpy(float)
                    rec["ret"] = tf.close[last] / tf.open[i] - 1
                    rec["bench_ret"] = bench["close"].reindex(tf.dates).ffill().to_numpy(float)[last] / b_open[i] - 1
            else:
                cost = round_trip_cost(tf.adv[i], cfg)
                rec.update(status="traded", ret=tf.ret[i], bench_ret=tf.bench_ret[i], cost=cost,
                           net_excess=tf.excess[i] - cost)
            rows.append(rec)
    return pd.DataFrame(rows)


def portfolio_by_year(pos: pd.DataFrame) -> pd.DataFrame:
    """Equal weight per company within a year; a company in two rising areas counts once."""
    t = pos[(pos["status"] == "traded") & pos["rising"]].drop_duplicates(["request_fy", "ticker"])
    return t.groupby("request_fy").agg(companies=("ticker", "nunique"),
                                       mean_net_excess=("net_excess", "mean"),
                                       median_net_excess=("net_excess", "median"),
                                       hit_rate=("net_excess", lambda x: float((x > 0).mean()))).reset_index()


def random_area_placebo(pos: pd.DataFrame, signals: pd.DataFrame, cfg: dict) -> dict:
    """Draw, in each year, as many areas as actually rose, at random from all areas with an
    exposed, tradable company; compare the real mean net excess with the draws."""
    rng = np.random.default_rng(int(cfg["tests"]["random_seed"]))
    draws = int(cfg["tests"]["placebo_draws"])
    traded = pos[pos["status"] == "traded"]
    real = traded[traded["rising"]].drop_duplicates(["request_fy", "ticker"])
    if real.empty:
        return {"companies": 0}
    years = sorted(real["request_fy"].unique())
    rising_traded = traded[traded["rising"]]
    per_year = {}
    for y in years:
        # as many random areas as there were rising areas with at least one tradable company
        k = int(rising_traded[rising_traded["request_fy"] == y]["area"].nunique())
        ty = traded[traded["request_fy"] == y]
        area_lists = {a: g["ticker"].tolist() for a, g in ty.groupby("area")}
        excess = ty.drop_duplicates("ticker").set_index("ticker")["net_excess"].to_dict()
        per_year[y] = (k, list(area_lists), area_lists, excess)
    dist = np.empty(draws)
    for d in range(draws):
        vals = []
        for y, (k, names, lists, excess) in per_year.items():
            if not names:
                continue
            pick = rng.choice(len(names), size=min(k, len(names)), replace=False)
            tickers = {t for j in pick for t in lists[names[j]]}
            vals.extend(excess[t] for t in tickers)
        dist[d] = np.mean(vals) if vals else np.nan
    actual = float(real["net_excess"].mean())
    dist = dist[~np.isnan(dist)]
    return {"companies": int(len(real)), "actual_mean_net_excess": actual,
            "placebo_mean": float(dist.mean()), "share_of_placebos_beaten": float((dist < actual).mean()),
            "passes_95pct": bool((dist < actual).mean() >= 0.95)}


def company_awards_v2(sources: pd.DataFrame, matches: pd.DataFrame, areas: dict[str, list[str]]) -> pd.DataFrame:
    """Version 2 award rows: every source, every usable match, area tags on the cleaned text.
    A company is keyed by CIK, so its divisions and subsidiaries pool into one history."""
    m = matches[["recipient_name", "ticker", "cik", "delisting_date", "match_type", "valid_from", "valid_until"]]
    df = sources.merge(m, on="recipient_name", how="inner")
    df = df[(df["action_date"] >= pd.to_datetime(df["valid_from"])) &
            (df["action_date"] < pd.to_datetime(df["valid_until"]))].copy()
    df["key"] = df["cik"].astype(int).astype(str)
    df["areas"] = tag_areas(df["text"].fillna(""), areas)
    df["delisting_date"] = pd.to_datetime(df["delisting_date"], errors="coerce")
    return df[["key", "ticker", "cik", "delisting_date", "action_date", "available_date", "amount", "areas",
               "recipient_name", "source", "match_type"]].reset_index(drop=True)

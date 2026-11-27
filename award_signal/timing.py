"""Timing check: did rising-area companies move before the model bought them?

Design frozen 2026-11-27, before this check was run.

The model dates every budget signal 1 July of the year before the fiscal year.
DoD publishes the line-item request (R-1, P-1) earlier, on the dates below. If
the market reads the request when it comes out, the gain would sit between the
release and 1 July, where the backtest never looks.

For each request year, companies exposed to an area (version 2 award history
and exclusions, exposure measured with awards public before the release date,
market cap $100M to $5B at the window start) are scored in four windows of
trading sessions around R, the first session on or after the release date:

  pre      open of R-60 to open of R-5     (anticipation)
  release  open of R-5 to open of R+5      (the announcement itself)
  drift    open of R+5 to the first open after 1 July (release to the model's entry)
  hold     the model's 252-session hold, already measured by the backtest

Each window's return is measured against IWM over the same sessions, with no
trading costs. Rising-area companies (counted once per year) are compared with
the same random-area draws the backtest uses.

Reading, decided in advance: a window where rising areas beat at least 95% of
random-area draws, with a positive mean, means the budget request carries news
the market prices before 1 July, so the next step is earlier sources. No such
window means the 16 keyword areas carry no usable news, so the next step is
linking companies to individual budget lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stages import exposure, random_area_placebo

# Date DoD published each request's line-item detail (R-1/P-1). FY2026 and FY2027 from
# press coverage of the Comptroller postings; earlier years from the DoD budget rollouts.
RELEASE_DATES = {
    2015: "2014-03-04", 2016: "2015-02-02", 2017: "2016-02-09", 2018: "2017-05-23",
    2019: "2018-02-12", 2020: "2019-03-12", 2021: "2020-02-10", 2022: "2021-05-28",
    2023: "2022-03-28", 2024: "2023-03-13", 2025: "2024-03-11", 2026: "2025-06-25",
    2027: "2026-04-21",
}
WINDOWS = ("pre", "release", "drift")


def window_rows(signals: pd.DataFrame, awards: pd.DataFrame, fundamentals, provider, cfg: dict,
                log=print) -> pd.DataFrame:
    sig_cfg = cfg["signal"]
    bench = provider.get(cfg["prices"]["benchmark"])
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for rfy, g in signals.groupby("request_fy"):
        if int(rfy) not in RELEASE_DATES:
            continue
        rel = pd.Timestamp(RELEASE_DATES[int(rfy)])
        entry_date = pd.Timestamp(g["signal_date"].iloc[0])
        rising = set(g.loc[g["rising"], "area"])
        exp = exposure(awards, rel, cfg)
        exp = exp[exp["exposed"]]
        log(f"  FY{rfy} (released {rel.date()}): {len(rising)} rising areas, {exp['key'].nunique()} exposed companies")
        for r in exp.itertuples(index=False):
            rec = {"request_fy": rfy, "release_date": rel, "area": r.area, "rising": r.area in rising,
                   "ticker": r.ticker, "cik": r.cik, "status": "", "market_cap": np.nan}
            rec.update({f"{w}_excess": np.nan for w in WINDOWS})
            if pd.notna(r.delisting_date) and r.delisting_date <= rel:
                rec["status"] = "delisted before release"
                rows.append(rec)
                continue
            if r.ticker not in cache:
                px = provider.get(r.ticker)
                cache[r.ticker] = px.dropna(subset=["open", "close"]) if px is not None and len(px) > 80 else None
            px = cache[r.ticker]
            if px is None:
                rec["status"] = "no price data"
                rows.append(rec)
                continue
            px = px[px["open"] > 0]
            d = px.index
            b_open = bench["open"].reindex(d).ffill().to_numpy(float)
            o = px["open"].to_numpy(float)
            R = int(d.searchsorted(rel, side="left"))
            E = int(d.searchsorted(entry_date, side="right"))
            if R - 60 < 0 or R + 5 >= len(d):
                rec["status"] = "outside price history"
                rows.append(rec)
                continue
            shares = fundamentals.shares_asof(r.cik, d[R - 60]) if pd.notna(r.cik) else None
            mcap = (shares or np.nan) * px["close"].iloc[R - 61] if R - 61 >= 0 else np.nan
            rec["market_cap"] = mcap
            if np.isnan(mcap):
                rec["status"] = "market cap unknown"
            elif not (sig_cfg["min_market_cap"] <= mcap <= sig_cfg["max_market_cap"]):
                rec["status"] = "outside market-cap band"
            else:
                rec["status"] = "traded"

                def xs(a, b):
                    if b <= a or b >= len(d):
                        return np.nan
                    return (o[b] / o[a] - 1) - (b_open[b] / b_open[a] - 1)

                rec["pre_excess"] = xs(R - 60, R - 5)
                rec["release_excess"] = xs(R - 5, R + 5)
                rec["drift_excess"] = xs(R + 5, E)
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame, signals: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    for w in WINDOWS:
        col = f"{w}_excess"
        sub = rows.assign(net_excess=rows[col])
        sub.loc[sub["net_excess"].isna(), "status"] = "window empty"
        real = sub[(sub["status"] == "traded") & sub["rising"]].drop_duplicates(["request_fy", "ticker"])
        res = random_area_placebo(sub, signals, cfg)
        res["years"] = int(real["request_fy"].nunique())
        res["by_year"] = real.groupby("request_fy")["net_excess"].mean().round(4).to_dict()
        out[w] = res
    signal_in = [w for w in ("pre", "release", "drift")
                 if out[w].get("passes_95pct") and (out[w].get("actual_mean_net_excess") or 0) > 0]
    out["reading"] = ("budget request carries news priced before 1 July: pursue earlier sources (" +
                      ", ".join(signal_in) + ")") if signal_in else \
        "no window shows rising areas ahead of random areas: pursue line-level linking"
    return out

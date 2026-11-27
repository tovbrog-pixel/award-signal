"""Event study: 90-trading-day excess return over the benchmark, after costs.

Timing rule: a signal dated D is traded at the open of the first trading day
after D (plus any test delay) and sold at the open `horizon` trading days later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def half_spread_bps(adv_dollars: float, table: list) -> float:
    for upper, bps in table:
        if adv_dollars < float(upper):
            return float(bps)
    return float(table[-1][1])


def round_trip_cost(adv_dollars: float, cfg: dict) -> float:
    """Round-trip cost as a fraction of the position."""
    c = cfg["costs"]
    if not adv_dollars or adv_dollars <= 0 or np.isnan(adv_dollars):
        adv_dollars = 1.0
    hs = half_spread_bps(adv_dollars, c["half_spread_bps_by_adv"])
    impact = float(c["impact_coefficient_bps"]) * np.sqrt(float(c["position_dollars"]) / adv_dollars)
    return 2 * (hs + impact) / 10_000


class TickerFrame:
    """Price arrays for one ticker with the benchmark aligned to its calendar."""

    def __init__(self, px: pd.DataFrame, bench: pd.DataFrame, horizon: int):
        px = px.dropna(subset=["open", "close"])
        px = px[px["open"] > 0]
        self.dates = px.index
        self.open = px["open"].to_numpy(float)
        self.close = px["close"].to_numpy(float)
        dollar_vol = (px["close"] * px["volume"]).to_numpy(float)
        # ADV for entry at i = mean dollar volume over the 20 sessions before i
        self.adv = pd.Series(dollar_vol).rolling(20).mean().shift(1).to_numpy()
        b_open = bench["open"].reindex(self.dates).ffill().to_numpy(float)
        h = horizon
        n = len(self.dates)
        self.ret = np.full(n, np.nan)
        self.bench_ret = np.full(n, np.nan)
        if n > h:
            self.ret[: n - h] = self.open[h:] / self.open[: n - h] - 1
            self.bench_ret[: n - h] = b_open[h:] / b_open[: n - h] - 1
        self.excess = self.ret - self.bench_ret

    def entry_index(self, signal_date, delay: int) -> int | None:
        i = int(self.dates.searchsorted(pd.Timestamp(signal_date), side="right")) + delay
        return i if i < len(self.dates) else None


def run_study(events: pd.DataFrame, provider, cfg: dict, delay: int = 0) -> pd.DataFrame:
    s = cfg["signal"]
    h = int(s["horizon_days"])
    bench = provider.get(cfg["prices"]["benchmark"])
    if bench is None:
        raise RuntimeError(f"No price data for benchmark {cfg['prices']['benchmark']}")
    frames: dict[str, TickerFrame | None] = {}
    out = []
    for ev in events.itertuples(index=False):
        rec = ev._asdict()
        rec.update(status="", entry_date=pd.NaT, exit_date=pd.NaT, entry_index=np.nan,
                   market_cap=np.nan, adv_dollars=np.nan, ret=np.nan, bench_ret=np.nan,
                   excess=np.nan, cost=np.nan, net_excess=np.nan)
        if not ev.passes_ratio:
            rec["status"] = "below ratio"
        elif ev.in_cooldown:
            rec["status"] = "cooldown"
        else:
            if ev.ticker not in frames:
                px = provider.get(ev.ticker)
                frames[ev.ticker] = TickerFrame(px, bench, h) if px is not None and len(px) > h + 21 else None
            tf = frames[ev.ticker]
            i = tf.entry_index(ev.signal_date, delay) if tf is not None else None
            if tf is None:
                rec["status"] = "no price data"
            elif i is None or i < 21 or np.isnan(tf.ret[i]):
                rec["status"] = "outside price history"
            else:
                mcap = (ev.shares_outstanding or np.nan) * tf.close[i - 1]
                rec.update(entry_date=tf.dates[i], exit_date=tf.dates[i + h], entry_index=i,
                           market_cap=mcap, adv_dollars=tf.adv[i])
                if np.isnan(mcap):
                    rec["status"] = "market cap unknown"
                elif not (s["min_market_cap"] <= mcap <= s["max_market_cap"]):
                    rec["status"] = "outside market-cap band"
                else:
                    cost = round_trip_cost(tf.adv[i], cfg)
                    rec.update(status="traded", ret=tf.ret[i], bench_ret=tf.bench_ret[i],
                               excess=tf.excess[i], cost=cost, net_excess=tf.excess[i] - cost)
        out.append(rec)
    res = pd.DataFrame(out)
    res["year"] = pd.to_datetime(res["signal_date"]).dt.year
    res.attrs["frames"] = frames
    return res


def yearly_summary(res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["tests"]
    years = sorted(set(t["in_sample_years"]) | set(t["test_years"]))
    traded = res[res["status"] == "traded"]
    rows = []
    for y in years:
        g = traded[traded["year"] == y]
        rows.append({
            "year": y,
            "sample": "test" if y in t["test_years"] else "in-sample",
            "trades": len(g),
            "mean_net_excess": g["net_excess"].mean() if len(g) else np.nan,
            "median_net_excess": g["net_excess"].median() if len(g) else np.nan,
            "hit_rate": (g["net_excess"] > 0).mean() if len(g) else np.nan,
            "mean_cost": g["cost"].mean() if len(g) else np.nan,
            "beat_benchmark": bool(len(g) and g["net_excess"].mean() > 0),
        })
    return pd.DataFrame(rows)


def gate_check(yearly: pd.DataFrame, cfg: dict) -> dict:
    test = yearly[yearly["sample"] == "test"]
    positive = int(test["beat_benchmark"].sum())
    need = int(cfg["tests"]["min_positive_years"])
    return {"test_years": len(test), "years_beating_benchmark": positive,
            "years_required": need, "passed": positive >= need}


def placebo_test(res: pd.DataFrame, cfg: dict) -> dict:
    """Compare the real mean net excess with random entry dates on the same
    stocks, drawn from their whole price history away from real signal windows.

    Drawing from the whole history (instead of the same calendar year) keeps
    each placebo pick as variable as a real 90-day window; a narrow pool of
    overlapping windows understates the spread and flags noise as signal."""
    t = cfg["tests"]
    h = int(cfg["signal"]["horizon_days"])
    rng = np.random.default_rng(int(t["random_seed"]))
    draws = int(t["placebo_draws"])
    traded = res[res["status"] == "traded"]
    if traded.empty:
        return {"events": 0}
    frames = res.attrs["frames"]
    real_idx: dict[str, list[int]] = {}
    for ev in traded.itertuples(index=False):
        real_idx.setdefault(ev.ticker, []).append(int(ev.entry_index))
    samples = []
    for ev in traded.itertuples(index=False):
        tf = frames[ev.ticker]
        n = len(tf.dates)
        ok = np.zeros(n, dtype=bool)
        ok[21: n - h] = True
        ok &= ~np.isnan(tf.excess)
        for j in real_idx[ev.ticker]:
            ok[max(0, j - h): j + h + 1] = False
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            continue
        pick = rng.choice(cand, size=draws, replace=True)
        costs = np.array([round_trip_cost(a, cfg) for a in tf.adv[pick]])
        samples.append(tf.excess[pick] - costs)
    if not samples:
        return {"events": len(traded), "note": "no placebo dates available"}
    mat = np.vstack(samples)
    dist = mat.mean(axis=0)
    actual = traded["net_excess"].mean()
    return {
        "events": int(len(traded)),
        "events_with_placebo": int(mat.shape[0]),
        "actual_mean_net_excess": float(actual),
        "placebo_mean": float(dist.mean()),
        "placebo_p95": float(np.percentile(dist, 95)),
        "share_of_placebos_beaten": float((dist < actual).mean()),
        "passes_95pct": bool((dist < actual).mean() >= 0.95),
    }

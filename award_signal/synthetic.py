"""Synthetic market for tests and the offline demo.

Builds fake prices, awards and fundamentals with a known planted effect, so we
can confirm the pipeline finds an effect that exists and finds nothing when
the effect is switched off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DictProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def get(self, ticker: str):
        return self.frames.get(ticker)


class StubFundamentals:
    def __init__(self, revenue: dict[int, float], shares: dict[int, float]):
        self.revenue, self.shares = revenue, shares

    def revenue_asof(self, cik, date):
        return self.revenue.get(int(cik))

    def shares_asof(self, cik, date):
        return self.shares.get(int(cik))


def _path(rng, n, drift, vol, start_price):
    steps = rng.normal(drift, vol, n)
    return start_price * np.exp(np.cumsum(steps))


def make_market(seed: int = 1, n_tickers: int = 40, effect: float = 0.15,
                start: str = "2014-01-01", end: str = "2026-06-30", n_delisted: int = 3) -> dict:
    """effect = extra total return spread over the 90 sessions after each
    qualifying award's signal date (0 switches the planted effect off).

    The last `n_delisted` companies stop trading partway through the sample,
    and one more has no price history at all, so the survivorship check has
    something to find."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    bench_close = _path(rng, n, 0.0003, 0.012, 150.0)
    frames = {"IWM": _frame(dates, bench_close, rng, 30_000_000)}

    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    revenue, shares, rows, mapping = {}, {}, [], []
    awards = []
    base_paths = {}
    for k, tk in enumerate(tickers):
        cik = 1000 + k
        rev = float(rng.uniform(40e6, 600e6))
        price0 = float(rng.uniform(8, 40))
        revenue[cik] = rev
        shares[cik] = float(rng.uniform(1.0, 3.0) * rev / price0)  # market cap ~1-3x revenue
        idio = rng.normal(-0.5 * 0.022 ** 2, 0.022, n)  # zero expected simple return
        log_ret = 1.1 * np.diff(np.log(bench_close), prepend=np.log(bench_close[0])) + idio
        base_paths[tk] = log_ret
        name = f"Synthetic Systems {k:02d} Inc."
        mapping.append({"recipient_name": name, "cik": cik, "ticker": tk, "sec_title": name.upper(),
                        "score": 1.0, "confirmed": "Y"})
        # a few awards per company per year, some large relative to revenue
        for year in range(2015, 2026):
            for _ in range(rng.poisson(1.2)):
                action = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 364)))
                ratio = float(rng.choice([0.03, 0.06, 0.12, 0.2, 0.35], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
                awards.append({"award_id": f"W{k:02d}{year}{len(awards)}", "recipient_name": name,
                               "recipient_uei": f"UEI{k:04d}", "amount": max(ratio * rev, 7.6e6),
                               "action_date": action, "sub_agency": "Department of the Army",
                               "description": "synthetic award"})
    aw = pd.DataFrame(awards)
    aw["signal_date"] = aw["action_date"] + pd.Timedelta(days=90)

    # plant the effect after each qualifying award
    for tk, log_ret in base_paths.items():
        k = int(tk[3:])
        rev = revenue[1000 + k]
        mine = aw[(aw["recipient_name"] == f"Synthetic Systems {k:02d} Inc.") & (aw["amount"] / rev >= 0.10)]
        for sd in mine["signal_date"]:
            i = int(dates.searchsorted(sd, side="right"))
            if i + 91 < n:
                log_ret[i + 1: i + 91] += np.log1p(effect) / 90
        close = float(rng.uniform(8, 40)) * np.exp(np.cumsum(log_ret))
        frames[tk] = _frame(dates, close, rng, float(rng.uniform(2e5, 2e6)))
    for k in range(n_tickers - n_delisted, n_tickers):
        cut = pd.Timestamp(int(rng.integers(2017, 2025)), int(rng.integers(1, 13)), 15)
        frames[f"SYN{k:02d}"] = frames[f"SYN{k:02d}"].loc[:cut]
    if n_delisted:
        frames.pop(f"SYN{n_tickers - n_delisted - 1:02d}", None)
    return {"awards": aw, "map": pd.DataFrame(mapping), "provider": DictProvider(frames),
            "fundamentals": StubFundamentals(revenue, shares)}


def _frame(dates, close, rng, base_volume):
    gap = rng.normal(0, 0.004, len(close))
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(close))))
    vol = base_volume * np.exp(rng.normal(0, 0.4, len(close)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=pd.Index(dates, name="date"))

"""Daily price providers. Each returns a DataFrame indexed by date with
columns open, high, low, close, volume (split- and dividend-adjusted).

yfinance is free but only covers stocks still trading, which biases results
upward (survivorship bias). The csv provider reads files exported from a
survivorship-free source such as Sharadar SEP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


class CsvProvider:
    def __init__(self, csv_dir: str | Path):
        self.dir = Path(csv_dir)
        self._cache: dict[str, pd.DataFrame | None] = {}

    def get(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._cache:
            p = self.dir / f"{ticker}.csv"
            if not p.exists():
                self._cache[ticker] = None
            else:
                df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
                df.columns = [c.lower() for c in df.columns]
                self._cache[ticker] = df[COLS]
        return self._cache[ticker]


class YFinanceProvider(CsvProvider):
    """Downloads from Yahoo Finance once, then reads the saved CSV."""

    def __init__(self, csv_dir: str | Path, start: str, end: str):
        super().__init__(csv_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start, self.end = start, end

    def get(self, ticker: str) -> pd.DataFrame | None:
        p = self.dir / f"{ticker}.csv"
        if not p.exists():
            import yfinance as yf

            raw = yf.download(ticker, start=self.start, end=self.end, auto_adjust=True,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                self._cache[ticker] = None
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            raw.index.name = "date"
            raw[COLS].to_csv(p)
        return super().get(ticker)


def make_provider(cfg: dict):
    p = cfg["prices"]
    if p["provider"] == "csv":
        return CsvProvider(p["csv_dir"])
    if p["provider"] == "yfinance":
        start = (pd.Timestamp(cfg["awards"]["start_date"]) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(cfg["awards"]["end_date"]) + pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        return YFinanceProvider(p["csv_dir"], start, end)
    raise ValueError(f"unknown price provider {p['provider']!r}")

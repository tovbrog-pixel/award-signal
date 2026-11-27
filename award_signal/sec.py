"""SEC EDGAR data with point-in-time rules.

Every XBRL fact carries a `filed` date. A value counts as known on date D
only if it was filed on or before D, so later restatements never leak back.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .net import Http

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueServicesNet",
]
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "20-F", "40-F"}


def load_tickers(http: Http) -> pd.DataFrame:
    data = http.json("GET", TICKERS_URL)
    df = pd.DataFrame(list(data.values()))
    df = df.rename(columns={"cik_str": "cik", "title": "sec_title"})
    df["ticker"] = df["ticker"].str.upper().str.replace(".", "-", regex=False)
    return df[["cik", "ticker", "sec_title"]]


class Fundamentals:
    """Point-in-time revenue and shares outstanding from SEC company facts."""

    def __init__(self, http: Http):
        self.http = http
        self._facts: dict[int, dict] = {}

    def facts(self, cik: int) -> dict:
        if cik not in self._facts:
            try:
                self._facts[cik] = self.http.json("GET", FACTS_URL.format(cik=int(cik)))
            except Exception:
                self._facts[cik] = {}
        return self._facts[cik]

    def revenue_asof(self, cik: int, date) -> float | None:
        return annual_revenue_asof(self.facts(cik), pd.Timestamp(date))

    def shares_asof(self, cik: int, date) -> float | None:
        return shares_outstanding_asof(self.facts(cik), pd.Timestamp(date))


def _entries(facts: dict, taxonomy: str, concept: str, unit: str) -> list[dict]:
    try:
        return facts["facts"][taxonomy][concept]["units"][unit]
    except (KeyError, TypeError):
        return []


def annual_revenue_asof(facts: dict, date: pd.Timestamp) -> float | None:
    """Latest full-year revenue filed on or before `date`."""
    best = None  # (period_end, filed, value)
    for concept in REVENUE_CONCEPTS:
        for e in _entries(facts, "us-gaap", concept, "USD"):
            if e.get("form") not in ANNUAL_FORMS or "start" not in e:
                continue
            filed = pd.Timestamp(e["filed"])
            if filed > date:
                continue
            start, end = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            if not 300 <= (end - start).days <= 400:
                continue
            key = (end, filed, float(e["val"]))
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else None


def shares_outstanding_asof(facts: dict, date: pd.Timestamp) -> float | None:
    best = None  # (filed, end, value)
    for e in _entries(facts, "dei", "EntityCommonStockSharesOutstanding", "shares"):
        filed = pd.Timestamp(e["filed"])
        if filed > date:
            continue
        key = (filed, pd.Timestamp(e["end"]), float(e["val"]))
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def save_tickers(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)

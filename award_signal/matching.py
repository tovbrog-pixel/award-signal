"""Match contract recipient names to SEC-listed companies.

Exact matches on a normalized name are accepted automatically. Close matches
go to a candidates file for a person to confirm, because a wrong match puts a
fake signal into the test.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd

SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LTD",
    "LIMITED", "LP", "LLP", "PLC", "HOLDINGS", "HOLDING", "GROUP", "THE", "NV", "SA", "AG",
    "DBA", "USA", "US", "NA",
}


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    words = [w for w in s.split() if w not in SUFFIXES]
    return " ".join(words)


def build_candidates(recipients: pd.Series, tickers: pd.DataFrame, cutoff: float = 0.88) -> pd.DataFrame:
    t = tickers.copy()
    t["norm"] = t["sec_title"].map(normalize)
    t = t[t["norm"] != ""].drop_duplicates("norm")
    by_norm = t.set_index("norm")
    norms = list(by_norm.index)
    out = []
    for name in sorted(set(recipients.dropna())):
        n = normalize(name)
        if not n:
            continue
        if n in by_norm.index:
            r = by_norm.loc[n]
            out.append((name, int(r["cik"]), r["ticker"], r["sec_title"], 1.0, "Y"))
            continue
        close = difflib.get_close_matches(n, norms, n=1, cutoff=cutoff)
        if close:
            r = by_norm.loc[close[0]]
            score = difflib.SequenceMatcher(None, n, close[0]).ratio()
            out.append((name, int(r["cik"]), r["ticker"], r["sec_title"], round(score, 3), ""))
    return pd.DataFrame(out, columns=["recipient_name", "cik", "ticker", "sec_title", "score", "confirmed"])


def load_map(data_dir: str | Path) -> pd.DataFrame:
    """Confirmed recipient map: data/recipient_map.csv if present, else exact matches only."""
    data_dir = Path(data_dir)
    reviewed = data_dir / "recipient_map.csv"
    src = reviewed if reviewed.exists() else data_dir / "recipient_map_candidates.csv"
    if not src.exists():
        raise FileNotFoundError("Run `python -m award_signal match` first.")
    df = pd.read_csv(src, dtype={"confirmed": str})
    df = df[df["confirmed"].fillna("").str.upper().str.startswith("Y")]
    return df[["recipient_name", "cik", "ticker"]]

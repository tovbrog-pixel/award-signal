"""Match contract recipient names to SEC-listed companies.

Pass 1 checks SEC's current ticker list: exact matches on a normalized name
are accepted, close matches wait for a person to confirm.

Pass 2 handles recipients pass 1 missed. It looks for an exact name match in
SEC's full filer list, which still holds companies acquired or delisted since
their awards. If an Alpha Vantage delisted list is supplied, the company's old
ticker and delisting date come from it.

A wrong match puts a fake signal into the test, so anything less than an exact
name match stays unconfirmed.
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

COLUMNS = ["recipient_name", "cik", "ticker", "sec_title", "score", "confirmed", "source",
           "delisting_date", "note"]


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.upper()
    s = re.sub(r"/[A-Z]{2,3}/?\s*$", " ", s)   # SEC state-of-incorporation tags like /DE/
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    words = [w for w in s.split() if w not in SUFFIXES]
    return " ".join(words)


def _index_by_norm(df: pd.DataFrame, col: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, n in enumerate(df[col].map(normalize)):
        if n:
            out.setdefault(n, []).append(i)
    return out


def build_candidates(recipients: pd.Series, tickers: pd.DataFrame,
                     all_names: pd.DataFrame | None = None,
                     delisted: pd.DataFrame | None = None,
                     cutoff: float = 0.88) -> pd.DataFrame:
    t = tickers.reset_index(drop=True)
    t_idx = _index_by_norm(t, "sec_title")
    t_norms = list(t_idx)
    a_idx = _index_by_norm(all_names.reset_index(drop=True), "sec_title") if all_names is not None else {}
    d = delisted.reset_index(drop=True) if delisted is not None else None
    d_idx = _index_by_norm(d, "name") if d is not None else {}

    out = []
    for name in sorted(set(recipients.dropna())):
        n = normalize(name)
        if not n:
            continue
        # pass 1: current SEC ticker list
        if n in t_idx:
            r = t.iloc[t_idx[n][0]]
            if len(n.split()) == 1:
                # one-word names collide easily (NOVA GROUP vs NOVA LTD), so a person checks them
                out.append([name, int(r["cik"]), r["ticker"], r["sec_title"], 1.0, "", "sec_tickers", None,
                            "one-word name: check it is the same company"])
            else:
                out.append([name, int(r["cik"]), r["ticker"], r["sec_title"], 1.0, "Y", "sec_tickers", None, ""])
            continue
        close = difflib.get_close_matches(n, t_norms, n=1, cutoff=cutoff)
        if close:
            r = t.iloc[t_idx[close[0]][0]]
            score = round(difflib.SequenceMatcher(None, n, close[0]).ratio(), 3)
            out.append([name, int(r["cik"]), r["ticker"], r["sec_title"], score, "", "sec_tickers",
                        None, "close match: check it is the same company"])
            continue
        # pass 2: every EDGAR filer, then the delisted ticker list
        if n not in a_idx:
            continue
        rows = all_names.iloc[a_idx[n]]
        ciks = sorted(set(rows["cik"].astype(int)))
        cik, title = ciks[0], rows.iloc[0]["sec_title"]
        note = "" if len(ciks) == 1 else f"{len(ciks)} SEC filers share this name: {ciks}"
        hits = d.iloc[d_idx[n]] if n in d_idx else None
        if hits is not None and len(hits) == 1 and len(ciks) == 1 and len(n.split()) > 1:
            h = hits.iloc[0]
            out.append([name, cik, h["ticker"], title, 1.0, "Y", "sec_all_names+av_delisted",
                        h["delisting_date"], ""])
        elif hits is not None and len(hits) > 0:
            h = hits.sort_values("delisting_date").iloc[-1]
            out.append([name, cik, h["ticker"], title, 1.0, "", "sec_all_names+av_delisted",
                        h["delisting_date"], (note + "; " if note else "") +
                        f"{len(hits)} delisted tickers share this name: check which one"])
        else:
            out.append([name, cik, "", title, 1.0, "", "sec_all_names", None,
                        (note + "; " if note else "") +
                        "SEC filer with no current ticker: may be private, or add its old ticker by hand"])
    return pd.DataFrame(out, columns=COLUMNS)


def load_map(data_dir: str | Path) -> pd.DataFrame:
    """Confirmed recipient map: data/recipient_map.csv if present, else exact matches only."""
    data_dir = Path(data_dir)
    reviewed = data_dir / "recipient_map.csv"
    src = reviewed if reviewed.exists() else data_dir / "recipient_map_candidates.csv"
    if not src.exists():
        raise FileNotFoundError("Run `python -m award_signal match` first.")
    df = pd.read_csv(src, dtype={"confirmed": str, "ticker": str})
    df = df[df["confirmed"].fillna("").str.upper().str.startswith("Y")]
    df = df[df["ticker"].fillna("").str.strip() != ""]
    if "delisting_date" not in df.columns:
        df["delisting_date"] = pd.NaT
    df["delisting_date"] = pd.to_datetime(df["delisting_date"], errors="coerce")
    return df[["recipient_name", "cik", "ticker", "delisting_date"]]

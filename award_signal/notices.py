"""Pre-award signal: SAM.gov notices of intent to award sole-source.

Agencies post a notice before awarding a contract without competition, and
the notice usually names the company. This module streams SAM.gov's yearly
archived-opportunity files, keeps the sole-source and intent-to-award notices,
and finds the public companies they name by looking up company names from
SEC's ticker list and Alpha Vantage's delisted list directly in the notice text.

Each file is large, so it is read as a stream and only matching rows are kept
(data/notices/FYxxxx.csv); a rerun skips years already filtered.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

import pandas as pd

from .matching import SUFFIXES, normalize

ARCHIVE_URL = ("https://sam.gov/api/prod/fileextractservices/v1/api/download/"
               "Contract%20Opportunities/Archived%20Data/FY{fy}_archived_opportunities.csv?privacy=Public")

INTENT_RE = re.compile(
    r"sole[\s\-]*source|intent(?:ion)?\s+to\s+(?:award|negotiate|sole|issue|contract)|"
    r"intends?\s+to\s+(?:award|negotiate|issue|enter|procure|contract)|notice\s+of\s+intent|"
    r"other\s+than\s+full\s+and\s+open|only\s+one\s+responsible\s+source", re.I)
MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(million|m\b|billion|b\b)?", re.I)

KEEP = ["NoticeId", "Title", "Type", "PostedDate", "Department/Ind.Agency", "Sub-Tier",
        "NaicsCode", "Awardee", "Description"]

# Words too common to identify a company on their own
GENERIC = {
    "AMERICAN", "AMERICA", "UNITED", "STATES", "NATIONAL", "FIRST", "GENERAL", "GLOBAL",
    "INTERNATIONAL", "SERVICES", "SERVICE", "SYSTEMS", "SYSTEM", "TECHNOLOGIES", "TECHNOLOGY",
    "SOLUTIONS", "DEFENSE", "INDUSTRIES", "INDUSTRIAL", "ENTERPRISES", "PRODUCTS", "RESOURCES",
    "ENERGY", "POWER", "DATA", "HEALTH", "MEDICAL", "SCIENCE", "SCIENCES", "ENGINEERING", "AND",
    "OF", "NORTH", "SOUTH", "EAST", "WEST", "PACIFIC", "ATLANTIC", "CAPITAL", "TRUST", "BANK",
    "FINANCIAL", "PARTNERS", "MANAGEMENT", "SUPPLY", "ELECTRIC", "ELECTRONICS", "NETWORK",
    "NETWORKS", "SOFTWARE", "COMMUNICATIONS", "SECURITY", "AEROSPACE", "MARINE", "AIR",
    "WATER", "OIL", "GAS", "STEEL", "METALS", "CHEMICAL", "LABS", "LABORATORIES", "DIGITAL",
    "ADVANCED", "APPLIED", "INNOVATIONS", "CORE", "PRIME", "NEW", "WORLD", "STAR", "TOTAL",
    "ONE", "THE", "FEDERAL", "GOVERNMENT", "PROGRAM", "PROJECT", "CONTRACT", "AWARD",
    "SOURCE", "SOLE", "NOTICE", "INTENT", "OFFICE", "ARMY", "NAVY", "FORCE", "CENTER",
}
CORP_SUFFIX_RE = r"(?:,?\s*(?:inc|incorporated|corp|corporation|co|company|llc|ltd|limited|plc|technologies|holdings))\b"


def filter_year(fy: int, out_path: Path, session, log=print) -> int:
    """Stream one fiscal-year archive and keep the sole-source and intent rows."""
    url = ARCHIVE_URL.format(fy=fy)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    kept = seen = 0
    tmp = out_path.with_suffix(".part")
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        text = io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=KEEP)
            writer.writeheader()
            for row in reader:
                seen += 1
                if (row.get("Type") or "").lower().startswith("award"):
                    continue
                desc = (row.get("Description") or "")[:6000]
                if INTENT_RE.search(row.get("Title") or "") or INTENT_RE.search(desc):
                    rec = {k: row.get(k, "") for k in KEEP}
                    rec["Description"] = desc
                    writer.writerow(rec)
                    kept += 1
                if seen % 200_000 == 0:
                    log(f"    FY{fy}: {seen:,} notices read, {kept:,} kept")
    tmp.replace(out_path)
    log(f"  FY{fy}: {seen:,} notices, {kept:,} sole-source or intent notices")
    return kept


def build_name_index(tickers: pd.DataFrame, delisted: pd.DataFrame | None,
                     all_names: pd.DataFrame | None = None) -> dict[str, dict]:
    """Normalized company name -> company record. Generic names are left out.
    Delisted companies get their CIK from SEC's full filer list when one name matches."""
    index: dict[str, dict] = {}
    cik_by_name: dict[str, set] = {}
    if all_names is not None:
        for t, c in zip(all_names["sec_title"], all_names["cik"]):
            cik_by_name.setdefault(normalize(t), set()).add(int(c))
    for r in tickers.itertuples(index=False):
        n = normalize(r.sec_title)
        if _usable(n):
            index.setdefault(n, {"cik": int(r.cik), "ticker": r.ticker, "sec_title": r.sec_title,
                                 "delisting_date": None})
    if delisted is not None:
        for r in delisted.itertuples(index=False):
            n = normalize(r.name)
            ciks = cik_by_name.get(n, set())
            if _usable(n) and n not in index and len(ciks) == 1:
                index[n] = {"cik": next(iter(ciks)), "ticker": r.ticker, "sec_title": r.name,
                            "delisting_date": r.delisting_date}
    return index


def _usable(n: str) -> bool:
    words = n.split()
    if not words or all(w in GENERIC or w.isdigit() for w in words):
        return False
    if len(words) == 1 and len(n) < 6:
        return False
    return True


def _tokens(text: str) -> list[str]:
    s = text.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return [w for w in s.split() if w not in SUFFIXES]


def first_words(index: dict[str, dict]) -> dict[str, list[int]]:
    """First word of each name -> the word counts of names starting with it, longest first."""
    out: dict[str, set] = {}
    for k in index:
        w = k.split()
        out.setdefault(w[0], set()).add(len(w))
    return {k: sorted(v, reverse=True) for k, v in out.items()}


def find_companies(text: str, index: dict[str, dict], starts: dict[str, list[int]] | None = None) -> list[str]:
    starts = starts if starts is not None else first_words(index)
    toks = _tokens(text)
    found = []
    for i, tok in enumerate(toks):
        for n in starts.get(tok, ()):
            if i + n > len(toks):
                continue
            key = " ".join(toks[i:i + n])
            if key in index:
                if n == 1:
                    # a single word counts only when the text writes it as a company name
                    if not re.search(r"\b" + re.escape(key) + CORP_SUFFIX_RE, text, re.I):
                        continue
                found.append(key)
                break
    return sorted(set(found))


def estimated_value(text: str) -> float | None:
    m = MONEY_RE.search(text)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit.startswith("m"):
        val *= 1e6
    elif unit.startswith("b"):
        val *= 1e9
    return val if val >= 1000 else None


def scan_notices(notice_files: list[Path], index: dict[str, dict], cfg: dict,
                 max_companies: int = 3, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (awards-style rows, recipient map) for notices naming public companies."""
    a = cfg["awards"]
    start, end = pd.Timestamp(a["start_date"]), pd.Timestamp(a["end_date"])
    rows = []
    starts = first_words(index)
    for f in notice_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df = df.rename(columns={"Department/Ind.Agency": "Department", "Sub-Tier": "SubTier"})
        for r in df.itertuples(index=False):
            posted = pd.to_datetime(str(r.PostedDate)[:10], errors="coerce")
            if pd.isna(posted) or not (start <= posted <= end):
                continue
            text = f"{r.Title} . {r.Awardee} . {r.Description}"
            names = find_companies(text, index, starts)
            if not names or len(names) > max_companies:
                continue
            value = estimated_value(r.Description)
            for n in names:
                rows.append({"award_id": r.NoticeId, "recipient_name": index[n]["sec_title"],
                             "amount": value if value else 0.0, "action_date": posted,
                             "signal_date": posted, "sub_agency": r.SubTier,
                             "description": str(r.Title)[:200], "notice_type": r.Type,
                             "recipient_uei": None})
        log(f"  {f.name}: {len(rows):,} company mentions so far")
    awards = pd.DataFrame(rows)
    if awards.empty:
        return awards, pd.DataFrame(columns=["recipient_name", "cik", "ticker", "delisting_date", "confirmed"])
    awards = awards.drop_duplicates(subset=["award_id", "recipient_name"]).sort_values("signal_date")
    used = {index[k]["sec_title"]: index[k] for k in index}
    rmap = pd.DataFrame([{"recipient_name": t, "cik": used[t]["cik"], "ticker": used[t]["ticker"],
                          "delisting_date": used[t]["delisting_date"], "confirmed": "Y"}
                         for t in sorted(set(awards["recipient_name"]))])
    return awards, rmap

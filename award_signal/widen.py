"""Stage 2, version 2: more exposure sources and wider recipient matching.

Rules frozen 2026-11-27 before any version 2 return was computed.

Sources of award dollars (each row: recipient, award date, date the public
could see it, dollars, text to tag):
  announcements  DoD daily contract announcements, $7.5M and up. Seen the same
                 day. Tagged on the full paragraph after removing the sentences
                 that name the contracting office, the funds and the work
                 location, plus a few organization names that contain area
                 words (Space and Naval Warfare Systems Command buys IT).
  sam            SAM.gov archived award notices from DoD, $100,000 up to $7.5M
                 (larger DoD actions are already in the announcements). Seen on
                 the posting date. Tagged on title plus the first 500
                 characters of the description.
  sbir           SBIR/STTR awards from DoD under $7.5M, from SBIR.gov's bulk
                 file. SBIR.gov posts awards late, so each counts as seen 365
                 days after the award date. Tagged on the award title.

Subsidiaries come from Exhibit 21 of 10-K filings, sampled every two years,
for companies with SEC public float between $50M and $10B in any year and an
SIC code in a defense-related industry. Matched companies in finance,
insurance, real estate, blank-check shells or tobacco are dropped.
"""

from __future__ import annotations

import csv
import html
import io
import re
import sys
from pathlib import Path

import pandas as pd

from .matching import normalize
from .net import Http, HttpError
from .notices import _usable

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/CY{year}Q2I.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"
SBIR_URL = "https://data.www.sbir.gov/mod_awarddatapublic_no_abstract/award_data_no_abstract.csv"

SAM_MIN, SAM_MAX = 100_000, 7_500_000
SBIR_LAG_DAYS = 365
FLOAT_MIN, FLOAT_MAX = 50e6, 10e9
EX21_YEARS = (2013, 2015, 2017, 2019, 2021, 2023, 2025)

# Industries searched for subsidiaries (SIC code ranges, inclusive)
EX21_SIC = [(1380, 1389), (1600, 1799), (2800, 2832), (2840, 2899), (3000, 3999), (4500, 4599),
            (4800, 4899), (7370, 7379), (8700, 8748)]
# Industries whose name matches are dropped
DROP_SIC = [(2100, 2199), (6000, 6799)]

# Removed from announcement text before tagging
NOISE_SENTENCES = [
    r"[^.]*\bcontracting activit(?:y|ies)\b[^.]*\.?",
    r"[^.]*\bobligated\b[^.]*\.?",
    r"[^.]*\bfunds?\b[^.]*\.?",
    r"[^.]*\bwork will be performed\b[^.]*\.?",
]
NOISE_ORGS = [
    r"space and naval warfare systems(?: center| command)?", r"aviation and missile (?:command|life cycle management command|center)",
]
NOISE_PLACES = [r"(?:[A-Z][a-z]+ ){1,3}Missile Range", r"(?:[A-Z][a-z]+ ){1,3}Space Force (?:Base|Station)"]
_NOISE = re.compile("|".join(NOISE_SENTENCES + NOISE_ORGS), re.I)
_PLACES = re.compile("|".join(NOISE_PLACES))


def announcement_tag_text(text: str) -> str:
    return _NOISE.sub(" ", _PLACES.sub(" ", str(text or "")))


def _in(code, ranges) -> bool:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return False
    return any(lo <= c <= hi for lo, hi in ranges)


# ---------------------------------------------------------------- SEC extras

def float_ciks(http: Http, years=range(2014, 2026), log=print) -> set[int]:
    """CIKs whose public float was between $50M and $10B in at least one year."""
    out: set[int] = set()
    for y in years:
        try:
            data = http.json("GET", FRAMES_URL.format(year=y))
        except HttpError as exc:
            log(f"  public float CY{y}: {exc}")
            continue
        for d in data.get("data", []):
            if FLOAT_MIN <= float(d.get("val", 0)) <= FLOAT_MAX:
                out.add(int(d["cik"]))
    return out


def submissions(http: Http, cik: int) -> dict:
    try:
        return http.json("GET", SUBMISSIONS_URL.format(cik=int(cik)))
    except HttpError:
        return {}


def tenk_filings(sub: dict) -> list[tuple[pd.Timestamp, str]]:
    r = sub.get("filings", {}).get("recent", {})
    out = []
    for form, date, acc in zip(r.get("form", []), r.get("filingDate", []), r.get("accessionNumber", [])):
        if form in ("10-K", "10-K405", "10-KT"):
            out.append((pd.Timestamp(date), acc))
    return sorted(out)


def sampled_filings(filings: list[tuple[pd.Timestamp, str]]) -> list[tuple[pd.Timestamp, str]]:
    """First 10-K filed in each sample year."""
    picked = {}
    for d, acc in filings:
        if d.year in EX21_YEARS and d.year not in picked:
            picked[d.year] = (d, acc)
    return [picked[y] for y in sorted(picked)]


STATES = {normalize(s) for s in (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho "
    "Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota "
    "Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah "
    "Vermont Virginia Washington Wisconsin Wyoming Canada Mexico Germany France Israel Japan China India "
    "Australia Netherlands Singapore Luxembourg Bermuda Ireland Belgium Sweden Norway Denmark Switzerland "
    "Italy Spain Brazil Korea Taiwan").split()} | {"NEW YORK", "NEW JERSEY", "NEW MEXICO", "NEW HAMPSHIRE",
    "NORTH CAROLINA", "SOUTH CAROLINA", "NORTH DAKOTA", "SOUTH DAKOTA", "RHODE ISLAND", "WEST VIRGINIA",
    "DISTRICT OF COLUMBIA", "UNITED KINGDOM", "CAYMAN ISLANDS", "BRITISH VIRGIN ISLANDS", "HONG KONG"}
HEADINGS = re.compile(r"subsidiar|jurisdiction|incorporat|organiz|percent|ownership|exhibit|name of|"
                      r"state or|country|doing business|^\(?\d+\)?$", re.I)


def parse_ex21(text: str) -> list[str]:
    """Subsidiary names from an Exhibit 21 (HTML or plain text)."""
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    t = re.sub(r"(?i)</(tr|p|div|li|h\d)>|<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</t[dh]>", "\t", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\xa0", " ")
    names = []
    for line in t.splitlines():
        cells = [c.strip() for c in re.split(r"\t|\s{3,}", line) if c.strip()]
        for c in cells:
            c = re.sub(r"\s*\(.*$", "", c)                 # "(a Delaware corporation)", "(100%)"
            c = re.sub(r"[\.…]{3,}.*$", "", c)        # dotted leaders
            c = re.sub(r"^[\-•\*\d\.\s]+", "", c).strip()  # bullets, numbering
            if not re.search(r"[A-Za-z]{2}", c) or HEADINGS.search(c):
                continue
            n = normalize(c)
            if n and n not in STATES and _usable(n) and len(c) <= 120:
                names.append(c)
            break                                           # the name is the first usable cell
    return names


def ex21_names(http: Http, cik: int, acc: str) -> list[str]:
    a = acc.replace("-", "")
    try:
        idx = http.json("GET", INDEX_URL.format(cik=int(cik), acc=a))
    except HttpError:
        return []
    items = idx.get("directory", {}).get("item", [])
    docs = [i["name"] for i in items if re.search(r"ex[-_]?21", i.get("name", ""), re.I)
            and i["name"].lower().endswith((".htm", ".html", ".txt"))]
    out = []
    for name in docs[:2]:
        try:
            out.extend(parse_ex21(http.text(DOC_URL.format(cik=int(cik), acc=a, name=name))))
        except HttpError:
            continue
    return out


def build_subsidiaries(http: Http, ciks: list[int], log=print) -> pd.DataFrame:
    """key, sub_name, cik, valid_from, valid_until (award dates the link holds for)."""
    rows = []
    for k, cik in enumerate(ciks):
        sub = submissions(http, cik)
        filings = sampled_filings(tenk_filings(sub))
        seen: dict[str, list] = {}
        for i, (d, acc) in enumerate(filings):
            for nm in set(ex21_names(http, cik, acc)):
                seen.setdefault(normalize(nm), []).append((i, d, nm))
        for key, hits in seen.items():
            first_i, first_d, nm = hits[0]
            last_i = hits[-1][0]
            until = filings[last_i + 1][0] if last_i + 1 < len(filings) else hits[-1][1] + pd.DateOffset(years=2)
            rows.append({"key": key, "sub_name": nm, "cik": int(cik), "valid_from": first_d, "valid_until": until})
        if (k + 1) % 50 == 0:
            log(f"  subsidiary lists: {k + 1:,} of {len(ciks):,} companies, {len(rows):,} names")
    df = pd.DataFrame(rows, columns=["key", "sub_name", "cik", "valid_from", "valid_until"])
    if df.empty:
        return df
    # a name listed by two companies over overlapping years is ambiguous: drop it
    bad = set()
    for key, g in df.groupby("key"):
        if g["cik"].nunique() > 1:
            g = g.sort_values("valid_from")
            if (g["valid_from"].iloc[1:].to_numpy() < g["valid_until"].iloc[:-1].to_numpy()).any():
                bad.add(key)
    return df[~df["key"].isin(bad)].reset_index(drop=True)


def company_info(http: Http, ciks, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SIC code and dated former names for each CIK."""
    rows, names = [], []
    for cik in ciks:
        s = submissions(http, cik)
        rows.append({"cik": int(cik), "sic": s.get("sic"), "sic_description": s.get("sicDescription"),
                     "name": s.get("name")})
        for f in s.get("formerNames", []) or []:
            names.append({"cik": int(cik), "key": normalize(f.get("name")),
                          "from": pd.to_datetime(f.get("from"), errors="coerce", utc=True),
                          "to": pd.to_datetime(f.get("to"), errors="coerce", utc=True)})
    info = pd.DataFrame(rows, columns=["cik", "sic", "sic_description", "name"])
    former = pd.DataFrame(names, columns=["cik", "key", "from", "to"])
    for c in ("from", "to"):
        former[c] = former[c].dt.tz_localize(None)
    return info, former


# ---------------------------------------------------------------- SBIR

def _col(df: pd.DataFrame, *cands) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_sbir(path: Path, session=None, log=print) -> pd.DataFrame:
    """DoD SBIR/STTR awards under $7.5M from SBIR.gov's bulk file."""
    if not path.exists():
        if session is None:
            return pd.DataFrame()
        log("Downloading SBIR.gov award file (about 65 MB)...")
        tmp = path.with_suffix(".part")
        with session.get(SBIR_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, encoding_errors="replace")
    c_co, c_ag = _col(df, "Company", "Firm"), _col(df, "Agency")
    c_title, c_amt = _col(df, "Award Title", "Title"), _col(df, "Award Amount")
    c_date = _col(df, "Proposal Award Date", "Award Date", "Date of Notification")
    c_year, c_contract = _col(df, "Award Year"), _col(df, "Contract")
    df = df[df[c_ag].str.contains("defense", case=False, na=False)]
    date = pd.to_datetime(df[c_date], errors="coerce") if c_date else pd.Series(pd.NaT, index=df.index)
    if c_year:
        date = date.fillna(pd.to_datetime(df[c_year].str[:4] + "-07-01", errors="coerce"))
    amt = pd.to_numeric(df[c_amt].str.replace(r"[$,]", "", regex=True), errors="coerce")
    out = pd.DataFrame({"source": "sbir", "recipient_name": df[c_co], "action_date": date,
                        "available_date": date + pd.Timedelta(days=SBIR_LAG_DAYS), "amount": amt,
                        "text": df[c_title] if c_title else "",
                        "contract": df[c_contract] if c_contract else ""})
    return out[(out["amount"] > 0) & (out["amount"] < SAM_MAX) & out["action_date"].notna()]


# ---------------------------------------------------------------- SAM award notices

SAM_AWARD_KEEP = ["NoticeId", "Title", "PostedDate", "Department/Ind.Agency", "Sub-Tier", "AwardNumber",
                  "AwardDate", "Award$", "Awardee", "Description"]


def keep_award_notice(row: dict) -> bool:
    if not (row.get("Type") or "").lower().startswith("award"):
        return False
    if "defense" not in (row.get("Department/Ind.Agency") or "").lower():
        return False
    try:
        v = float(re.sub(r"[$,\s]", "", row.get("Award$") or "") or 0)
    except ValueError:
        return False
    return SAM_MIN <= v < SAM_MAX


def load_sam_awards(files: list[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if df.empty:
            continue
        posted = pd.to_datetime(df["PostedDate"].str[:10], errors="coerce")
        awardee = df["Awardee"].str.split(r"\s+-\s+|\(|\bUEI\b", regex=True).str[0]
        parts.append(pd.DataFrame({
            "source": "sam", "recipient_name": awardee.str.strip(" ,"), "action_date": posted,
            "available_date": posted,
            "amount": pd.to_numeric(df["Award$"].str.replace(r"[$,\s]", "", regex=True), errors="coerce"),
            "text": df["Title"] + " . " + df["Description"].str[:500], "contract": df["AwardNumber"]}))
    if not parts:
        return pd.DataFrame(columns=["source", "recipient_name", "action_date", "available_date", "amount",
                                     "text", "contract"])
    return pd.concat(parts, ignore_index=True)


def contract_key(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str[:13]


# ---------------------------------------------------------------- all sources

def announcement_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[~raw["multi_award"].astype(bool)]
    d = pd.to_datetime(df["action_date"])
    return pd.DataFrame({"source": "announcements", "recipient_name": df["recipient_name"], "action_date": d,
                         "available_date": d, "amount": pd.to_numeric(df["face_value"], errors="coerce"),
                         "text": df["description"].fillna("").map(announcement_tag_text),
                         "contract": df["contract_number"]})


def combine_sources(ann: pd.DataFrame, sam: pd.DataFrame, sbir: pd.DataFrame) -> pd.DataFrame:
    """Stack the sources; an SBIR award already posted as a SAM notice counts once."""
    if len(sam) and len(sbir):
        k = set(contract_key(sam["contract"])) - {""}
        sbir = sbir[~contract_key(sbir["contract"]).isin(k)]
    df = pd.concat([x for x in (ann, sam, sbir) if len(x)], ignore_index=True)
    return df[(df["amount"].fillna(0) > 0) & df["action_date"].notna()].reset_index(drop=True)

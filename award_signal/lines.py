"""Stage 2, version 4: link companies to individual budget lines.

Design frozen 2026-11-27 before any version 4 return was computed.

Each R-1 or P-1 line gets program keys taken from its title:
  - an acronym in parentheses of three or more characters (Maneuver Support
    Vessel (MSV) -> MSV), unless it is a generic budget word
  - a designator such as F-35, MK-54, E-4B, AH-64E, AN/TPY-2
(Title phrases such as "Maneuver Support Vessel" were tried and turned off before
any return: too many common phrases.) A DoD announcement paragraph or SBIR award
title that contains a key (whole words; acronyms and
designators match case-sensitively) links its awardee to that line. A company
is exposed to a line when, over the three years before the budget release,
award dollars that mention the line reach $7.5M and 10% of all its award
dollars. Lines come from the stage 1 flags: a ramp (request at least 1.5 times
the enacted amount and at least $20M) or a new start (at least $10M).

Trade: open of the 6th session after the line-item release, held 252 sessions,
market cap $100M to $5B at entry, the usual trading costs. Each company counts
once a year however many flagged lines it is exposed to. Compared with the same
number of randomly drawn unflagged lines that have exposed companies.
Pass: beat 95% of random draws and at least 60% of years beat IWM.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .study import round_trip_cost
from .v3 import RELEASE_DATES, shares_for_band

FILLER = {
    "ADVANCED", "ADV", "AND", "OF", "THE", "FOR", "IN", "TO", "MODS", "MOD", "MODIFICATION", "MODIFICATIONS",
    "MODERNIZATION", "PROCUREMENT", "PROC", "DEVELOPMENT", "DEV", "DEMONSTRATION", "DEMO", "TECHNOLOGY", "TECH",
    "SYSTEM", "SYSTEMS", "SYS", "PROGRAM", "PROGRAMS", "SUPPORT", "EQUIPMENT", "EQUIP", "ENGINEERING", "ENG",
    "INITIAL", "SPARES", "SPARE", "REPAIR", "PARTS", "COMPONENTS", "UPGRADE", "UPGRADES", "IMPROVEMENT",
    "IMPROVEMENTS", "PRODUCT", "PRODUCTION", "ALL", "TYPES", "OTHER", "MISC", "MISCELLANEOUS", "LESS", "THAN",
    "ITEMS", "ITEM", "SERIES", "FAMILY", "INTEGRATION", "TEST", "EVALUATION", "RESEARCH", "APPLIED", "BASIC",
    "INITIATIVES", "INITIATIVE", "CAPABILITY", "CAPABILITIES", "OPERATIONAL", "OPERATIONS", "ACTIVITIES",
    "MANAGEMENT", "HQ", "SUMMARY", "COST", "FACILITIES", "FIELD", "TRAINING", "TRAINERS", "DEVICES", "SET",
    "SETS", "KITS", "KIT", "ARMY", "NAVY", "AIR", "FORCE", "MARINE", "CORPS", "DEFENSE", "DEFENSE-WIDE", "DOD",
    "JOINT", "TACTICAL", "COMBAT", "WEAPONS", "WEAPON", "VEHICLE", "VEHICLES", "AIRCRAFT", "SHIP", "SHIPS",
    "COMMUNICATIONS", "ELECTRONICS", "SOFTWARE", "HARDWARE", "INFORMATION", "TECHNICAL", "SPECIAL", "GENERAL",
    "PURPOSE", "NEW", "FUTURE", "NEXT", "GENERATION", "SPACE", "MISSILE", "MISSILES", "AMMUNITION", "AMMO",
    "COMMAND", "CONTROL", "CENTER", "SECURITY", "SURVEILLANCE", "INTELLIGENCE", "SENSOR", "SENSORS", "RADAR",
    "NETWORK", "NETWORKS", "DATA", "LINK", "BASE", "INSTALLATION", "ENHANCEMENTS", "ENHANCED", "LIFE", "CYCLE",
    "EXTENSION", "SERVICE", "SERVICES", "SUSTAINMENT", "ACQUISITION", "PLATFORM", "PLATFORMS", "AUTOMATED",
    "AUTOMATION", "PROTOTYPE", "PROTOTYPES", "PROTOTYPING", "RAPID", "SCIENCE", "STUDIES", "ANALYSIS", "SMALL",
    "LARGE", "MEDIUM", "LIGHT", "HEAVY", "CLASS", "MULTI", "MISSION", "SUPPORTING", "LOGISTICS", "MATERIEL",
    "MEDICAL", "CYBER", "FIGHTER", "BOMBER", "TANKER", "TRAINER", "HELICOPTER", "SUBMARINE", "ROTARY", "STRATEGIC",
    "SQUADRONS", "SQUADRON", "WING", "GROUND", "SURFACE", "UNDERSEA", "ELECTRONIC", "WARFARE", "SATELLITE",
}
GENERIC_ACRONYMS = {
    "RDT", "OCO", "USAF", "USMC", "USA", "USN", "DOD", "OSD", "DARPA", "SOCOM", "SOF", "MDA", "DISA", "DLA", "NGB",
    "ARNG", "USAR", "ANG", "OPA", "WPN", "PMC", "OMA", "ROK", "NATO", "CONUS", "OCONUS", "DHP", "HQ", "TBD", "DRP",
    "FMS", "MIP", "NIP", "ISR", "IED", "IT", "CR", "CRA", "R&D", "DWR", "GPS", "UAS", "UAV", "IR", "EW", "C2",
    "C3", "C4", "C3I", "C4I", "C4ISR", "S&T", "T&E", "TPY", "HMMWV", "JLTV", "SBIR", "STTR",
}
# Title phrases were tried on the announcement text and turned off before any return was
# computed: phrases such as "Coast Guard", "Unmanned Aerial" and "Situational Awareness" tied
# companies to lines they do not serve. Only acronyms and designators are used.
USE_TITLE_PHRASES = False
DESIGNATOR = re.compile(r"\b(?:AN/[A-Z]{2,4}-\d+[A-Z]?|[A-Z]{1,4}-\d{1,3}[A-Z]{0,2}|MK\s?\d{1,3}|M\d{1,3}A\d)\b")
PAREN = re.compile(r"\(([A-Za-z0-9/&\-]{3,12})\)")


def line_keys(title: str) -> list[tuple[str, bool]]:
    """(key, case_sensitive) pairs for one budget line title."""
    t = str(title or "")
    keys = []
    for a in PAREN.findall(t):
        if (a.upper() in GENERIC_ACRONYMS or a.upper() in FILLER or a.isdigit() or not re.search(r"[A-Za-z]", a)
                or re.search(r"[a-z]{2}", a)):          # "(Medical)" is a word, not an acronym
            continue
        keys.append((a, True))
    for d in DESIGNATOR.findall(t.upper()):
        d = re.sub(r"\s", " ", d)
        if d.split("-")[0] not in {"FY", "BA", "PE", "TO", "NO"}:
            keys.append((d, True))
    words = [w for w in re.split(r"[^A-Za-z0-9\-/&]+", re.sub(r"\([^)]*\)", " ", t)) if w]
    def filler(w):
        return w.upper() in FILLER or not re.search(r"[A-Za-z]", w)
    while words and filler(words[0]):
        words.pop(0)
    while words and filler(words[-1]):
        words.pop()
    if USE_TITLE_PHRASES and len(words) >= 2 and any(len(w) >= 5 and w.upper() not in FILLER for w in words):
        keys.append((" ".join(words), False))
    seen, out = set(), []
    for k, cs in keys:
        if k.lower() not in seen:
            seen.add(k.lower())
            out.append((k, cs))
    return out


def key_pattern(k: str, cs: bool) -> re.Pattern:
    p = r"(?<![A-Za-z0-9])" + re.escape(k).replace(r"\ ", r"[\s\-]+") + r"(?![A-Za-z0-9])"
    return re.compile(p if cs else f"(?i:{p})")


MAX_TITLE_KEY_COMPANIES = 25   # a title phrase found in more companies' awards is a common phrase


def link_awards(lines: pd.DataFrame, awards: pd.DataFrame) -> pd.DataFrame:
    """Rows (line_key_id, award row index) for every award paragraph mentioning a line's keys.
    A line is identified across years by its key set, so last year's awards count toward
    this year's line of the same program."""
    lines = lines.copy()
    lines["keys"] = lines["title"].map(line_keys)
    lines["key_id"] = lines["keys"].map(lambda ks: "|".join(sorted(k.lower() for k, _ in ks)))
    uniq = lines[lines["key_id"] != ""].drop_duplicates("key_id")
    text = awards["text"].fillna("").tolist()
    names = awards["recipient_name"].to_numpy()
    tok = re.compile(r"[A-Za-z0-9/&\-]+")
    index_cs: dict[str, set] = {}
    index_ci: dict[str, set] = {}
    for i, t in enumerate(text):
        for w in set(tok.findall(t)):
            index_cs.setdefault(w, set()).add(i)
            index_ci.setdefault(w.lower(), set()).add(i)
    hits_by_key: dict[tuple[str, bool], np.ndarray] = {}
    for ks in uniq["keys"]:
        for k, cs in ks:
            if (k, cs) in hits_by_key:
                continue
            parts = tok.findall(k)
            if not parts:
                hits_by_key[(k, cs)] = np.array([], dtype=int)
                continue
            first = parts[0] if cs else parts[0].lower()
            cand = (index_cs if cs else index_ci).get(first, set())
            if len(parts) > 1 or not cs:
                pat = key_pattern(k, cs)
                cand = {i for i in cand if pat.search(text[i])}
            idx = np.array(sorted(cand), dtype=int)
            if not cs and len(set(names[idx])) > MAX_TITLE_KEY_COMPANIES:
                idx = idx[:0]
            hits_by_key[(k, cs)] = idx
    rows = []
    for r in uniq.itertuples(index=False):
        idx = np.unique(np.concatenate([hits_by_key[(k, cs)] for k, cs in r.keys]))
        rows.extend((r.key_id, awards.index[i]) for i in idx)
    return lines, pd.DataFrame(rows, columns=["key_id", "award_idx"])


def exposures(lines: pd.DataFrame, links: pd.DataFrame, awards: pd.DataFrame, rel: pd.Timestamp, rfy: int,
              cfg: dict, share: float = 0.10, min_dollars: float = 7.5e6) -> pd.DataFrame:
    s = cfg["stages"]
    seen = awards["available_date"] if "available_date" in awards.columns else awards["action_date"]
    w = awards[(awards["action_date"] < rel) & (seen < rel) &
               (awards["action_date"] >= rel - pd.DateOffset(years=int(s["lookback_years"])))]
    tot = w.groupby("key")["amount"].sum()
    lk = links[links["award_idx"].isin(w.index)].merge(
        w[["key", "ticker", "cik", "delisting_date", "amount"]], left_on="award_idx", right_index=True)
    g = lk.groupby(["key_id", "key", "ticker", "cik"], dropna=False).agg(
        line_dollars=("amount", "sum"), delisting_date=("delisting_date", "first")).reset_index()
    g["share"] = g["line_dollars"] / g["key"].map(tot)
    g = g[(g["line_dollars"] >= min_dollars) & (g["share"] >= share)]
    yl = lines[lines["request_fy"] == rfy][["key_id", "ramp", "new_start"]]
    yl = yl[yl["key_id"] != ""].groupby("key_id")[["ramp", "new_start"]].any().reset_index()
    yl["flagged"] = yl["ramp"] | yl["new_start"]
    return g.merge(yl[["key_id", "flagged"]], on="key_id", how="inner")


def positions(lines, links, awards, fundamentals, provider, cfg, years, log=print,
              min_dollars: float = 7.5e6) -> pd.DataFrame:
    sig = cfg["signal"]
    bench = provider.get(cfg["prices"]["benchmark"])
    cache, rows = {}, []
    hold = int(cfg["stages"]["hold_days"])
    for rfy in years:
        if rfy not in RELEASE_DATES:
            continue
        rel = pd.Timestamp(RELEASE_DATES[rfy])
        ex = exposures(lines, links, awards, rel, rfy, cfg, min_dollars=min_dollars)
        log(f"  FY{rfy} (released {rel.date()}): {ex.loc[ex['flagged'], 'key_id'].nunique()} flagged lines with "
            f"exposed companies, {ex['key_id'].nunique()} lines in all, {ex['ticker'].nunique()} companies")
        for r in ex.itertuples(index=False):
            rec = {"request_fy": rfy, "area": r.key_id, "rising": bool(r.flagged), "ticker": r.ticker, "cik": r.cik,
                   "share": r.share, "line_dollars": r.line_dollars, "status": "", "entry_date": pd.NaT,
                   "market_cap": np.nan, "ret": np.nan, "bench_ret": np.nan, "cost": np.nan, "net_excess": np.nan}
            if pd.notna(r.delisting_date) and pd.Timestamp(r.delisting_date) <= rel:
                rows.append({**rec, "status": "delisted before release"})
                continue
            if r.ticker not in cache:
                px = provider.get(r.ticker)
                cache[r.ticker] = px.dropna(subset=["open", "close"]) if px is not None and len(px) > 30 else None
            px = cache[r.ticker]
            if px is None:
                rows.append({**rec, "status": "no price data"})
                continue
            px = px[px["open"] > 0]
            d = px.index
            i = int(d.searchsorted(rel, side="left")) + 5
            if i < 21 or i >= len(d):
                rows.append({**rec, "status": "outside price history"})
                continue
            j = i + hold
            facts = fundamentals.facts(r.cik) if pd.notna(r.cik) else {}
            mcap = (shares_for_band(facts, d[i]) or np.nan) * px["close"].iloc[i - 1]
            rec.update(entry_date=d[i], market_cap=mcap)
            if np.isnan(mcap):
                rows.append({**rec, "status": "market cap unknown"})
                continue
            if not (sig["min_market_cap"] <= mcap <= sig["max_market_cap"]):
                rows.append({**rec, "status": "outside market-cap band"})
                continue
            if j >= len(d):
                o = px["open"].to_numpy(float)
                b_open = bench["open"].reindex(d).ffill().to_numpy(float)
                rows.append({**rec, "status": "hold not finished",
                             "ret": o[-1] / o[i] - 1, "bench_ret": b_open[-1] / b_open[i] - 1})
                continue
            o = px["open"].to_numpy(float)
            b_open = bench["open"].reindex(d).ffill().to_numpy(float)
            adv = float((px["close"] * px["volume"]).iloc[max(0, i - 20):i].mean())
            cost = round_trip_cost(adv, cfg)
            ret, bret = o[j] / o[i] - 1, b_open[j] / b_open[i] - 1
            rows.append({**rec, "status": "traded", "ret": ret, "bench_ret": bret, "cost": cost,
                         "net_excess": ret - bret - cost})
    return pd.DataFrame(rows)


def build_award_table(sources: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Matched award rows that keep their text, for linking (same validity rules as version 2)."""
    m = matches[["recipient_name", "ticker", "cik", "delisting_date", "valid_from", "valid_until"]]
    df = sources.merge(m, on="recipient_name", how="inner")
    df = df[(df["action_date"] >= pd.to_datetime(df["valid_from"])) &
            (df["action_date"] < pd.to_datetime(df["valid_until"]))].copy()
    df["key"] = df["cik"].astype(int).astype(str)
    df["delisting_date"] = pd.to_datetime(df["delisting_date"], errors="coerce")
    return df.reset_index(drop=True)

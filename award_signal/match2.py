"""Recipient matching, version 2 (frozen 2026-11-27 before any v2 return was computed).

Version 1 matched a recipient only when its normalized name equaled a public
company's name. That missed divisions and subsidiaries ("Lockheed Martin
Rotary and Mission Systems"), renamed companies (Vectrus, now V2X) and
subsidiaries that carry their own names. Version 2 tries, in order:

1. exact      the recipient's normalized name equals a public company's current
              name, a delisted company's name, or a former name SEC lists for a
              listed company's CIK (one CIK only)
2. prefix     a public company's name of two or more words starts the recipient's
              name (the name needs one word of four or more letters that is not
              generic); a one-word company name (six or more letters, not
              generic) counts only when that company also wins awards under its
              own exact name or works in a defense-related industry (widen.EX21_SIC),
              and only for names from the DoD announcements (added 2026-11-27 after
              reviewing the first match list, before any return: SBIR names such as
              "Quantum Design" and "Coherent Technologies" went to unrelated companies)
3. subsidiary the recipient appears in the company's Exhibit 21 subsidiary list;
              usable only between the filing that first lists it and the next
              sampled filing that drops it

A first-word rule (Kratos Unmanned Aerial Systems -> Kratos) was tried on the
announcement names and dropped before any return was computed: most of its
matches were unrelated companies sharing a first word.

Each match carries valid_from / valid_until: the award dates it holds for.
A subsidiary link starts on the filing date of the first 10-K that lists it,
so an award only counts after the parent had disclosed the subsidiary; a
match through a former name holds only while SEC shows that name in use
(plus one year for late paperwork). Companies in finance, insurance, real
estate, shells or tobacco are dropped (see widen.DROP_SIC).
"""

from __future__ import annotations

import pandas as pd

from .matching import normalize
from .notices import GENERIC, _usable

FAR_PAST = pd.Timestamp("1900-01-01")
FAR_FUTURE = pd.Timestamp("2200-01-01")


def _distinctive(word: str) -> bool:
    return len(word) >= 6 and word not in GENERIC and not word.isdigit()


class Matcher:
    def __init__(self, index: dict[str, dict], tickers: pd.DataFrame | None = None,
                 all_names: pd.DataFrame | None = None, subsidiaries: pd.DataFrame | None = None):
        self.index = dict(index)
        self.former_keys: set[str] = set()
        self.by_cik = {int(r["cik"]): r for r in self.index.values() if pd.notna(r.get("cik"))}
        # former names of listed companies: SEC's filer list keeps every name a CIK has used
        if all_names is not None and tickers is not None:
            # every listed CIK, including ones whose current name is too short to match alone (V2X)
            listed = {}
            for r in tickers.itertuples(index=False):        # first share class listed wins
                listed.setdefault(int(r.cik), {"cik": int(r.cik), "ticker": r.ticker, "sec_title": r.sec_title,
                                               "delisting_date": None})
            listed.update(self.by_cik)
            ciks_by_name: dict[str, set] = {}
            for t, c in zip(all_names["sec_title"], all_names["cik"]):
                n = normalize(t)
                if n:
                    ciks_by_name.setdefault(n, set()).add(int(c))
            for n, ciks in ciks_by_name.items():
                if n in self.index or not _usable(n) or len(ciks) != 1:
                    continue
                c = next(iter(ciks))
                if c in listed:
                    self.index[n] = listed[c]
                    self.former_keys.add(n)
                    self.by_cik.setdefault(c, listed[c])
        self.confirmed: set[int] = set()   # CIKs that win awards under their own exact name
        self.subs: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
        if subsidiaries is not None and len(subsidiaries):
            for r in subsidiaries.itertuples(index=False):
                if int(r.cik) in self.by_cik:
                    self.subs.setdefault(r.key, []).append(
                        (int(r.cik), pd.Timestamp(r.valid_from), pd.Timestamp(r.valid_until)))

    def match(self, name: str) -> list[dict]:
        """All usable matches for one recipient (a subsidiary can belong to different
        parents in different years). Empty when nothing matches."""
        n = normalize(name)
        if not n:
            return []
        rec = self.index.get(n)
        if rec is not None:
            return [self._out(rec, "exact", n in self.former_keys and n)]
        toks = n.split()
        for k in range(len(toks) - 1, 0, -1):
            key = " ".join(toks[:k])
            rec = self.index.get(key)
            if rec is None or not any(len(w) >= 4 and w not in GENERIC for w in key.split()):
                continue
            if k >= 2 or _distinctive(key):
                out = self._out(rec, "prefix", key in self.former_keys and key)
                out["one_word"] = k == 1
                out["one_word_unconfirmed"] = k == 1 and int(rec["cik"]) not in self.confirmed
                return [out]
        if n in self.subs:
            return [self._out(self.by_cik[c], "subsidiary", False, f, u) for c, f, u in self.subs[n]]
        return []

    @staticmethod
    def _out(rec: dict, how: str, former_key=False, valid_from=FAR_PAST, valid_until=FAR_FUTURE) -> dict:
        return {"ticker": rec["ticker"], "cik": int(rec["cik"]), "sec_title": rec["sec_title"],
                "delisting_date": rec.get("delisting_date"), "match_type": how, "one_word": False,
                "one_word_unconfirmed": False,
                "former_name": former_key or "", "valid_from": valid_from, "valid_until": valid_until}


def match_table(names: pd.Series, matcher: Matcher) -> pd.DataFrame:
    """One row per (recipient name, match); unmatched names are left out."""
    rows = []
    unique = sorted(set(names.dropna()))
    matcher.confirmed = {int(matcher.index[normalize(n)]["cik"]) for n in unique if normalize(n) in matcher.index}
    for name in unique:
        for m in matcher.match(name):
            rows.append({"recipient_name": name, **m})
    cols = ["recipient_name", "ticker", "cik", "sec_title", "delisting_date", "match_type",
            "one_word", "one_word_unconfirmed", "former_name", "valid_from", "valid_until"]
    return pd.DataFrame(rows, columns=cols)


def apply_company_rules(mt: pd.DataFrame, info: pd.DataFrame, former: pd.DataFrame, drop_sic,
                        announcement_names: set | None = None) -> pd.DataFrame:
    """Date former-name matches, limit one-word prefixes, then drop companies in excluded
    industries. A one-word prefix counts only for DoD announcement recipients (SBIR and SAM.gov
    hold thousands of small firms named "Quantum ..." or "Coherent ..."), and, when the company
    never wins under its own exact name, only in a defense-related industry."""
    from .widen import EX21_SIC, _in

    mt = mt.copy()
    keep = []
    fn = {(int(c), k): (f, t) for c, k, f, t in zip(former["cik"], former["key"], former["from"], former["to"])}
    for i, r in mt.iterrows():
        if r["former_name"]:
            span = fn.get((int(r["cik"]), r["former_name"]))
            if span is None:
                keep.append(False)
                continue
            f, t = span
            mt.at[i, "valid_from"] = f if pd.notna(f) else FAR_PAST
            mt.at[i, "valid_until"] = (t + pd.DateOffset(years=1)) if pd.notna(t) else FAR_FUTURE
        keep.append(True)
    mt = mt[keep]
    sic = info.set_index("cik")["sic"].to_dict()
    mt["sic"] = mt["cik"].map(sic)
    mt = mt[~(mt["one_word_unconfirmed"].astype(bool) & ~mt["sic"].map(lambda c: _in(c, EX21_SIC)))]
    if announcement_names is not None:
        mt = mt[~(mt["one_word"].astype(bool) & ~mt["recipient_name"].isin(announcement_names))]
    return mt[~mt["sic"].map(lambda c: _in(c, drop_sic))].reset_index(drop=True)

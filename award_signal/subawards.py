"""Version 5: reach small suppliers through prime contractors' subcontract reports.

Design frozen 2026-11-27 before any version 5 return was computed.

Data. USAspending's subaward search (FSRS reports that prime contractors file):
DoD subcontracts of $250,000 and up, January 2013 to the latest month. A
subcontract counts as public 90 days after its date (primes report by the end of
the following month, and the public file trails that). Rows above $250M are
dropped as reporting errors (one prime reported a $3.7 billion subcontract to a
small engineering firm).

Text for linking. Each subcontract carries its own description plus the text of
every DoD announcement paragraph for the same prime contract (contract numbers
compared on their first 13 letters and digits), so a supplier on an F-35
contract inherits the F-35 mention.

Matching. Subcontractor names go through the version 2 matcher and exclusions;
like USAspending prime names they are legal names, so one-word prefixes are
allowed. New companies get a review list, written before any return, and a
frozen exclusion file (match_exclusions_v5.yaml).

Exposure. Over the three years before the budget release, a company's
line-linked dollars (prime awards plus subcontracts) reach $2M and 10% of all
its DoD award dollars (prime plus sub). Lines and the trade are as in version 4:
flagged lines are ramps or new starts; buy at the open of the 6th session after
the line-item release, hold 252 sessions, $100M to $5B market cap at entry,
usual costs; compare with randomly drawn unflagged lines; pass at 95% of draws
and 60% of years beating IWM.
"""

from __future__ import annotations

import pandas as pd

from .net import Http, HttpError
from .widen import announcement_tag_text, contract_key

URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
FIELDS = ["Sub-Award ID", "Sub-Awardee Name", "Sub-Award Date", "Sub-Award Amount", "Prime Award ID",
          "Prime Recipient Name", "Sub-Award Description"]
MIN_AMOUNT = 250_000
MAX_AMOUNT = 250_000_000
LAG_DAYS = 90
START = "2013-01-01"
LINE_MIN_DOLLARS = 2_000_000


def _body(s: str, e: str, page: int) -> dict:
    return {"subawards": True,
            "filters": {"award_type_codes": ["A", "B", "C", "D"],
                        "agencies": [{"type": "awarding", "tier": "toptier", "name": "Department of Defense"}],
                        "time_period": [{"start_date": s, "end_date": e}],
                        "award_amounts": [{"lower_bound": MIN_AMOUNT}]},
            "fields": FIELDS, "page": page, "limit": 100, "sort": "Sub-Award Amount", "order": "desc"}


def _window(http: Http, s: str, e: str, log, skipped: list, depth: int = 0) -> list:
    rows, page = [], 1
    while True:
        try:
            data = http.json("POST", URL, _body(s, e, page), retries=5)
        except HttpError as err:
            a, b = pd.Timestamp(s), pd.Timestamp(e)
            if (b - a).days >= 1 and depth < 6:
                mid = pd.Timestamp((a + (b - a) / 2).date())
                return (_window(http, s, mid.strftime("%Y-%m-%d"), log, skipped, depth + 1) +
                        _window(http, (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), e, log, skipped, depth + 1))
            log(f"    SKIPPED {s} to {e} page {page}: {str(err)[:120]}")
            skipped.append((s, e, page))
            return rows
        rows.extend(data.get("results", []))
        if not data.get("page_metadata", {}).get("hasNext"):
            return rows
        page += 1


def fetch(http: Http, end: str, log=print) -> tuple[pd.DataFrame, list]:
    rows, skipped = [], []
    for m in pd.date_range(START, end, freq="MS"):
        s, e = m.strftime("%Y-%m-%d"), (m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        rows.extend(_window(http, s, e, log, skipped))
        log(f"  {s[:7]}: {len(rows):,} subcontracts so far")
    return pd.DataFrame(rows), skipped


def to_sources(raw: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    """Subcontract rows in the common source layout, with the prime contract's announcement text."""
    d = pd.to_datetime(raw["Sub-Award Date"], errors="coerce")
    amt = pd.to_numeric(raw["Sub-Award Amount"], errors="coerce")
    ann = announcements[~announcements["multi_award"].astype(bool)].copy()
    ann["ck"] = contract_key(ann["contract_number"])
    ann["clean"] = ann["description"].fillna("").map(announcement_tag_text)   # same cleanup as version 4
    prime_text = ann[ann["ck"] != ""].groupby("ck")["clean"].apply(lambda x: " . ".join(x.astype(str)))
    ck = contract_key(raw["Prime Award ID"])
    text = raw["Sub-Award Description"].fillna("").astype(str) + " . " + ck.map(prime_text).fillna("")
    out = pd.DataFrame({"source": "subaward", "recipient_name": raw["Sub-Awardee Name"], "action_date": d,
                        "available_date": d + pd.Timedelta(days=LAG_DAYS), "amount": amt, "text": text,
                        "contract": raw["Prime Award ID"]})
    return out[(out["amount"] >= MIN_AMOUNT) & (out["amount"] <= MAX_AMOUNT) & out["action_date"].notna()
               & out["recipient_name"].notna()].reset_index(drop=True)

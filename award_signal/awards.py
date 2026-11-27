"""DoD contract awards: USAspending download or a CSV (HigherGov export, etc.).

Output columns:
    award_id, recipient_name, recipient_uei, amount, action_date, signal_date,
    sub_agency, description
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .net import Http, HttpError

USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

BASE_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Recipient UEI",
    "Award Amount",
    "Start Date",
    "Awarding Sub Agency",
    "Description",
]
DATE_FIELD = "Base Obligation Date"

COLUMNS = ["award_id", "recipient_name", "recipient_uei", "amount", "action_date",
           "signal_date", "sub_agency", "description"]


def _month_windows(start: str, end: str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur <= e:
        nxt = min(cur + pd.offsets.MonthEnd(0), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


def request_body(cfg: dict, w_start: str, w_end: str, fields: list, page: int, new_only: bool) -> dict:
    a = cfg["awards"]
    period = {"start_date": w_start, "end_date": w_end}
    if new_only:
        # Keep only awards whose first (base) transaction falls in the window,
        # so a modification to an old contract never shows up as a new award.
        period["date_type"] = "new_awards_only"
    return {
        "filters": {
            "award_type_codes": a["award_type_codes"],
            "agencies": [{"type": "awarding", "tier": "toptier", "name": a["awarding_agency"]}],
            "time_period": [period],
            "award_amounts": [{"lower_bound": a["min_amount"]}],
        },
        "fields": fields,
        "page": page,
        "limit": 100,
        "sort": "Award Amount",
        "order": "desc",
    }


def _fetch_window(cfg: dict, http: Http, w_start: str, w_end: str, state: dict, log, depth: int = 0) -> list:
    """All pages for one date window. If USAspending keeps dropping the
    connection, split the window in half and try each half; a single day that
    still fails is skipped and recorded in state['skipped']."""
    rows = []
    page = 1
    while True:
        body = request_body(cfg, w_start, w_end, state["fields"], page, state["new_only"])
        try:
            data = http.json("POST", USASPENDING_URL, body, retries=4)
        except HttpError as err:
            if err.status == 400:
                if state["new_only"] and "date_type" in str(err):
                    log("USAspending rejected the new-awards-only filter; using all action dates. "
                        "The base-date check still drops old contracts.")
                    state["new_only"] = False
                    continue
                if DATE_FIELD in state["fields"]:
                    log(f"USAspending rejected '{DATE_FIELD}'; falling back to 'Start Date'.")
                    state["fields"] = list(BASE_FIELDS)
                    continue
                raise
            s, e = pd.Timestamp(w_start), pd.Timestamp(w_end)
            if (e - s).days >= 1 and depth < 6:
                mid = s + (e - s) / 2
                mid = pd.Timestamp(mid.date())
                log(f"    connection trouble for {w_start} to {w_end}; splitting the window")
                return (_fetch_window(cfg, http, w_start, mid.strftime("%Y-%m-%d"), state, log, depth + 1) +
                        _fetch_window(cfg, http, (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                      w_end, state, log, depth + 1))
            log(f"    SKIPPED {w_start} to {w_end} (page {page}) after repeated connection failures")
            state["skipped"].append({"start": w_start, "end": w_end, "page": page, "error": str(err)[:200]})
            return rows
        rows.extend(data.get("results", []))
        if not data.get("page_metadata", {}).get("hasNext"):
            return rows
        page += 1


def fetch_usaspending(cfg: dict, http: Http, log=print) -> pd.DataFrame:
    a = cfg["awards"]
    state = {"fields": BASE_FIELDS + [DATE_FIELD], "new_only": bool(a.get("new_awards_only", True)),
             "skipped": []}
    rows = []
    for w_start, w_end in _month_windows(a["start_date"], a["end_date"]):
        rows.extend(_fetch_window(cfg, http, w_start, w_end, state, log))
        log(f"  {w_start[:7]}: {len(rows):,} award rows so far")
    if state["skipped"]:
        log(f"WARNING: {len(state['skipped'])} date windows were skipped; see data/awards_skipped.csv. "
            "Run `awards` again later to fill them.")
    df = normalize_usaspending(pd.DataFrame(rows), cfg)
    df.attrs["skipped"] = state["skipped"]
    return df


def normalize_usaspending(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    date_col = DATE_FIELD if DATE_FIELD in raw.columns else "Start Date"
    df = pd.DataFrame({
        "award_id": raw["Award ID"],
        "recipient_name": raw["Recipient Name"],
        "recipient_uei": raw.get("Recipient UEI"),
        "amount": pd.to_numeric(raw["Award Amount"], errors="coerce"),
        "action_date": pd.to_datetime(raw[date_col], errors="coerce"),
        "sub_agency": raw.get("Awarding Sub Agency"),
        "description": raw.get("Description"),
    })
    return finalize(df, cfg)


def load_awards_csv(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Load awards from a CSV with at least recipient_name, amount, action_date.

    If the CSV already carries a signal_date (for example the date a DoD
    announcement was published), that date is kept and no lag is added.
    """
    df = pd.read_csv(path)
    missing = {"recipient_name", "amount", "action_date"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    for col in ("award_id", "recipient_uei", "sub_agency", "description"):
        if col not in df.columns:
            df[col] = None
    return finalize(df, cfg)


def finalize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = cfg["awards"]
    start, end = pd.Timestamp(a["start_date"]), pd.Timestamp(a["end_date"])
    df = df.dropna(subset=["recipient_name", "amount", "action_date"])
    df = df[(df["action_date"] >= start) & (df["action_date"] <= end)]
    df = df[df["amount"] >= a["min_amount"]]
    lag = pd.Timedelta(days=int(a["reporting_lag_days"]))
    if "signal_date" not in df.columns:
        df["signal_date"] = df["action_date"] + lag
    else:
        df["signal_date"] = df["signal_date"].fillna(df["action_date"] + lag)
    df = df.drop_duplicates(subset=["award_id", "recipient_name", "action_date", "amount"])
    return df[COLUMNS].sort_values("signal_date").reset_index(drop=True)

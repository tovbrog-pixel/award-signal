"""Turn matched awards into dated signal events."""

from __future__ import annotations

import pandas as pd


def build_events(awards: pd.DataFrame, rmap: pd.DataFrame, fundamentals, cfg: dict,
                 log=print) -> pd.DataFrame:
    """One row per (ticker, signal_date) with award total, revenue and ratio.

    Returns every matched event with a `passes_ratio` flag so the excluded
    ones stay visible for review.
    """
    s = cfg["signal"]
    rmap = rmap.copy()
    if "delisting_date" not in rmap.columns:
        rmap["delisting_date"] = pd.NaT
    rmap["delisting_date"] = pd.to_datetime(rmap["delisting_date"], errors="coerce")
    m = awards.merge(rmap[["recipient_name", "cik", "ticker", "delisting_date"]], on="recipient_name", how="inner")
    if m.empty:
        log("No awards matched a listed company.")
        return pd.DataFrame()
    grouped = (m.groupby(["ticker", "cik", "signal_date"], as_index=False)
                 .agg(amount=("amount", "sum"),
                      n_awards=("amount", "size"),
                      first_action_date=("action_date", "min"),
                      delisting_date=("delisting_date", "max"),
                      recipients=("recipient_name", lambda x: "; ".join(sorted(set(x))))))
    revenues, shares = [], []
    for row in grouped.itertuples(index=False):
        revenues.append(fundamentals.revenue_asof(row.cik, row.signal_date))
        shares.append(fundamentals.shares_asof(row.cik, row.signal_date))
    grouped["revenue"] = revenues
    grouped["shares_outstanding"] = shares
    grouped["award_to_revenue"] = grouped["amount"] / grouped["revenue"]
    grouped["passes_ratio"] = (grouped["revenue"] > 0) & (grouped["award_to_revenue"] >= s["min_award_to_revenue"])
    grouped = apply_cooldown(grouped, int(s["cooldown_days"]))
    return grouped.sort_values("signal_date").reset_index(drop=True)


def apply_cooldown(events: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    """Among passing events, drop repeats on the same ticker inside the cooldown window."""
    events = events.sort_values(["ticker", "signal_date"]).copy()
    events["in_cooldown"] = False
    last: dict[str, pd.Timestamp] = {}
    for idx, row in events.iterrows():
        if not row["passes_ratio"]:
            continue
        prev = last.get(row["ticker"])
        if prev is not None and (row["signal_date"] - prev).days < cooldown_days:
            events.at[idx, "in_cooldown"] = True
            continue
        last[row["ticker"]] = row["signal_date"]
    return events

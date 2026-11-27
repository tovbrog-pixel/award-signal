import numpy as np
import pandas as pd
import pytest

from award_signal.awards import finalize, load_awards_csv
from award_signal.events import build_events
from award_signal.matching import build_candidates, normalize
from award_signal.sec import annual_revenue_asof, shares_outstanding_asof
from award_signal.study import (TickerFrame, gate_check, placebo_test, round_trip_cost,
                                run_study, yearly_summary)
from award_signal.synthetic import make_market

CFG = {
    "awards": {"start_date": "2015-01-01", "end_date": "2025-12-31", "min_amount": 7_500_000,
               "reporting_lag_days": 90},
    "signal": {"min_award_to_revenue": 0.10, "min_market_cap": 1e8, "max_market_cap": 5e9,
               "horizon_days": 90, "cooldown_days": 126, "delay_test_days": 5},
    "prices": {"benchmark": "IWM"},
    "costs": {"position_dollars": 4000, "half_spread_bps_by_adv": [[1e6, 150], [1e7, 50], [1e18, 15]],
              "impact_coefficient_bps": 100},
    "tests": {"in_sample_years": [2015, 2016, 2017], "test_years": list(range(2018, 2026)),
              "min_positive_years": 5, "placebo_draws": 300, "random_seed": 7},
}


def facts(entries_rev, entries_shares=()):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": list(entries_rev)}}},
                      "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": list(entries_shares)}}}}}


def test_revenue_is_point_in_time():
    f = facts([
        {"start": "2019-01-01", "end": "2019-12-31", "val": 100, "form": "10-K", "filed": "2020-03-01"},
        {"start": "2019-01-01", "end": "2019-12-31", "val": 90, "form": "10-K/A", "filed": "2021-06-01"},
        {"start": "2020-01-01", "end": "2020-12-31", "val": 150, "form": "10-K", "filed": "2021-03-01"},
        {"start": "2021-01-01", "end": "2021-03-31", "val": 60, "form": "10-Q", "filed": "2021-05-01"},
    ])
    assert annual_revenue_asof(f, pd.Timestamp("2020-02-01")) is None      # nothing filed yet
    assert annual_revenue_asof(f, pd.Timestamp("2020-06-01")) == 100       # original 10-K
    assert annual_revenue_asof(f, pd.Timestamp("2021-04-01")) == 150       # newer year
    assert annual_revenue_asof(f, pd.Timestamp("2021-05-15")) == 150       # quarterly ignored


def test_shares_point_in_time():
    f = facts([], [{"end": "2020-01-31", "val": 10e6, "filed": "2020-03-01"},
                   {"end": "2021-01-31", "val": 12e6, "filed": "2021-03-01"}])
    assert shares_outstanding_asof(f, pd.Timestamp("2020-12-31")) == 10e6
    assert shares_outstanding_asof(f, pd.Timestamp("2021-03-01")) == 12e6


def test_awards_lag_and_filters(tmp_path):
    df = pd.DataFrame({"award_id": ["a", "b", "c"], "recipient_name": ["X", "X", "Y"],
                       "recipient_uei": None, "amount": [8e6, 5e6, 9e6],
                       "action_date": pd.to_datetime(["2016-01-10", "2016-02-01", "2014-05-01"]),
                       "sub_agency": None, "description": None})
    out = finalize(df, CFG)
    assert list(out["award_id"]) == ["a"]                       # small and out-of-window dropped
    assert out["signal_date"].iloc[0] == pd.Timestamp("2016-04-09")   # 90-day reporting lag

    p = tmp_path / "aw.csv"
    pd.DataFrame({"recipient_name": ["Z"], "amount": [1e7], "action_date": ["2017-01-05"],
                  "signal_date": ["2017-01-06"]}).to_csv(p, index=False)
    out = load_awards_csv(p, CFG)
    assert out["signal_date"].iloc[0] == pd.Timestamp("2017-01-06")   # announcement date kept


def test_matching():
    assert normalize("Kratos Defense & Security Solutions, Inc.") == "KRATOS DEFENSE AND SECURITY SOLUTIONS"
    tickers = pd.DataFrame({"cik": [1, 2], "ticker": ["AAA", "BBB"],
                            "sec_title": ["ALPHA SYSTEMS INC", "BRAVO AEROSPACE CORP"]})
    cand = build_candidates(pd.Series(["Alpha Systems, Inc.", "Bravo Aerospce Corporation", "Nobody LLC"]), tickers)
    by = cand.set_index("recipient_name")
    assert by.loc["Alpha Systems, Inc.", "confirmed"] == "Y"
    assert by.loc["Bravo Aerospce Corporation", "ticker"] == "BBB"
    assert by.loc["Bravo Aerospce Corporation", "confirmed"] == ""   # close match needs review
    assert "Nobody LLC" not in by.index


def test_entry_is_next_session_open():
    dates = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"open": np.arange(1, 301, dtype=float), "high": 1.0, "low": 1.0,
                       "close": np.arange(1, 301, dtype=float), "volume": 1e6}, index=dates)
    tf = TickerFrame(px, px, horizon=90)
    signal = dates[100]                       # signal published on session 100
    assert tf.entry_index(signal, 0) == 101   # trade at session 101's open
    assert tf.entry_index(signal, 5) == 106
    assert tf.ret[101] == pytest.approx(px["open"].iloc[191] / px["open"].iloc[101] - 1)


def test_costs_rise_for_thin_stocks():
    assert round_trip_cost(500_000, CFG) > round_trip_cost(50_000_000, CFG)
    assert round_trip_cost(50_000_000, CFG) == pytest.approx(2 * (15 + 100 * np.sqrt(4000 / 5e7)) / 1e4)


def _study(effect, seed):
    m = make_market(seed=seed, effect=effect)
    ev = build_events(m["awards"], m["map"], m["fundamentals"], CFG, log=lambda *a: None)
    res = run_study(ev, m["provider"], CFG)
    return res, gate_check(yearly_summary(res, CFG), CFG), placebo_test(res, CFG)


def test_finds_planted_effect():
    res, gate, placebo = _study(0.15, seed=1)
    assert (res["status"] == "traded").sum() > 100
    assert gate["passed"] and placebo["passes_95pct"]


def test_null_market_fails_placebo():
    _, _, placebo = _study(0.0, seed=1)
    assert not placebo["passes_95pct"]


def test_usaspending_normalization():
    from award_signal.awards import normalize_usaspending
    raw = pd.DataFrame([{"Award ID": "W1", "Recipient Name": "ALPHA SYSTEMS INC", "Recipient UEI": "U1",
                         "Award Amount": 12_000_000, "Start Date": "2018-01-01",
                         "Base Obligation Date": "2018-02-01", "Awarding Sub Agency": "Navy",
                         "Description": "x"}])
    out = normalize_usaspending(raw, CFG)
    assert out["action_date"].iloc[0] == pd.Timestamp("2018-02-01")   # base obligation date preferred
    assert out["signal_date"].iloc[0] == pd.Timestamp("2018-05-02")

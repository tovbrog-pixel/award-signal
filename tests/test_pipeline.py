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


def test_new_awards_only_request():
    from award_signal.awards import request_body
    cfg = {"awards": {**CFG["awards"], "award_type_codes": ["A"], "awarding_agency": "Department of Defense"}}
    body = request_body(cfg, "2018-01-01", "2018-01-31", ["Award ID"], 1, new_only=True)
    assert body["filters"]["time_period"][0]["date_type"] == "new_awards_only"
    body = request_body(cfg, "2018-01-01", "2018-01-31", ["Award ID"], 1, new_only=False)
    assert "date_type" not in body["filters"]["time_period"][0]


def test_parse_sec_and_alpha_vantage_files():
    from award_signal.sec import parse_all_names, parse_av_delisted
    names = parse_all_names("#1 A LIFESAFER HOLDINGS, INC.:0001509607:\nODD: NAME CORP:0000000042:\njunk line\n")
    assert list(names["cik"]) == [1509607, 42]
    assert names["sec_title"].iloc[1] == "ODD: NAME CORP"
    av = parse_av_delisted("symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
                           "GONE,Gone Systems Inc,NYSE,Stock,2010-01-01,2019-06-30,Delisted\n"
                           "GONEW,Gone Warrant,NYSE,ETF,2010-01-01,2019-06-30,Delisted\n")
    assert list(av["ticker"]) == ["GONE"]
    assert av["delisting_date"].iloc[0] == pd.Timestamp("2019-06-30")


def test_match_finds_delisted_company(tmp_path):
    from award_signal.matching import load_map
    tickers = pd.DataFrame({"cik": [1], "ticker": ["AAA"], "sec_title": ["ALPHA SYSTEMS INC"]})
    all_names = pd.DataFrame({"sec_title": ["ALPHA SYSTEMS INC", "GONE SYSTEMS INC", "PRIVATE WIDGETS LLC"],
                              "cik": [1, 77, 88]})
    delisted = pd.DataFrame({"ticker": ["GONE"], "name": ["Gone Systems, Inc."],
                             "delisting_date": [pd.Timestamp("2019-06-30")]})
    recips = pd.Series(["Gone Systems Inc", "Private Widgets, LLC"])
    cand = build_candidates(recips, tickers, all_names, delisted).set_index("recipient_name")
    assert cand.loc["Gone Systems Inc", "ticker"] == "GONE"
    assert cand.loc["Gone Systems Inc", "confirmed"] == "Y"
    assert cand.loc["Private Widgets, LLC", "ticker"] == ""
    assert cand.loc["Private Widgets, LLC", "confirmed"] == ""
    cand.reset_index().to_csv(tmp_path / "recipient_map_candidates.csv", index=False)
    m = load_map(tmp_path)
    assert list(m["ticker"]) == ["GONE"]                      # no-ticker rows are left out
    assert m["delisting_date"].iloc[0] == pd.Timestamp("2019-06-30")


def test_worst_case_and_delisting_statuses():
    from award_signal.study import worst_case
    res, gate, placebo = _study(0.15, seed=1)
    statuses = set(res["status"])
    assert "stopped trading during hold" in statuses or "no price data" in statuses
    wc = worst_case(res, CFG)
    assert wc["unpriced_trades"] > 0
    assert wc["mean_net_excess"] < res.loc[res["status"] == "traded", "net_excess"].mean()


def test_reused_ticker_is_flagged():
    from award_signal.synthetic import DictProvider
    dates = pd.bdate_range("2016-01-01", "2022-12-30")
    px = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6}, index=dates)
    ev = pd.DataFrame([{"ticker": "OLD", "cik": 1, "signal_date": pd.Timestamp("2018-03-01"), "amount": 2e7,
                        "n_awards": 1, "first_action_date": pd.Timestamp("2017-12-01"),
                        "delisting_date": pd.Timestamp("2019-01-31"), "recipients": "Old Co",
                        "revenue": 1e8, "shares_outstanding": 2e7, "award_to_revenue": 0.2,
                        "passes_ratio": True, "in_cooldown": False}])
    res = run_study(ev, DictProvider({"OLD": px, "IWM": px}), CFG)
    assert res["status"].iloc[0] == "ticker reused after delisting"


def test_usaspending_splits_and_skips_failing_windows():
    from award_signal.awards import fetch_usaspending
    from award_signal.net import HttpError

    class FakeHttp:
        def json(self, method, url, body, retries=8):
            p = body["filters"]["time_period"][0]
            if p["start_date"] <= "2018-03-15" <= p["end_date"]:
                raise HttpError(0, "Remote end closed connection")
            return {"results": [{"Award ID": p["start_date"], "Recipient Name": "X", "Recipient UEI": None,
                                 "Award Amount": 9e6, "Start Date": p["start_date"],
                                 "Base Obligation Date": p["start_date"], "Awarding Sub Agency": "Army",
                                 "Description": ""}],
                    "page_metadata": {"hasNext": False}}

    cfg = {"awards": {**CFG["awards"], "start_date": "2018-02-01", "end_date": "2018-04-30",
                      "award_type_codes": ["A"], "awarding_agency": "Department of Defense"}}
    df = fetch_usaspending(cfg, FakeHttp(), log=lambda *a: None)
    skipped = df.attrs["skipped"]
    assert [(s["start"], s["end"]) for s in skipped] == [("2018-03-15", "2018-03-15")]
    assert "2018-02-01" in set(df["award_id"]) and "2018-04-01" in set(df["award_id"])
    assert len(df) > 3        # March was split into pieces around the bad day


def test_normalize_drops_state_tags():
    assert normalize("L3HARRIS TECHNOLOGIES, INC. /DE/") == normalize("L3Harris Technologies, Inc")
    assert normalize("GOODYEAR TIRE & RUBBER CO /OH/") == normalize("Goodyear Tire & Rubber Company, The")


def test_award_after_delisting_is_not_a_trade():
    from award_signal.synthetic import DictProvider
    from award_signal.study import worst_case
    dates = pd.bdate_range("2016-01-01", "2018-12-31")
    px = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6}, index=dates)
    ev = pd.DataFrame([{"ticker": "GONE", "cik": 1, "signal_date": pd.Timestamp("2020-03-01"), "amount": 2e7,
                        "n_awards": 1, "first_action_date": pd.Timestamp("2019-12-01"),
                        "delisting_date": pd.Timestamp("2017-06-30"), "recipients": "Gone Co",
                        "revenue": 1e8, "shares_outstanding": 2e7, "award_to_revenue": 0.2,
                        "passes_ratio": True, "in_cooldown": False}])
    res = run_study(ev, DictProvider({"IWM": px}), CFG)
    assert res["status"].iloc[0] == "delisted before signal"
    assert worst_case(res, CFG)["unpriced_trades"] == 0


ARTICLE_HTML = """<html><body><div class="body">
<p style="text-align: center;"><strong>AIR FORCE</strong></p>
<p>Boeing Defense, Space &amp; Security, a division of The Boeing Co., Seattle, Washington, has been awarded a $2,819,046,560 modification to a previously awarded contract (FA8625-11-C-6600) for 15 KC-46A Lot 13 production aircraft. Fiscal 2027 aircraft procurement funds in the amount of $2,819,046,560 are being obligated at the time of award.</p>
<p>Amentum Technology Inc., Tullahoma, Tennessee, has been awarded a cost-plus-fixed-fee $68,276,868 task order (FA4890-27-F-0001) for squadron support. Fiscal 2027 operations funds in the amount of $5,000,000 are being obligated at the time of award.</p>
<p style="text-align: center;"><strong><span>NAVY</span></strong></p>
<p>Alpha Widgets Inc.,* Huntsville, Alabama (N00019-27-D-0101); Bravo Systems LLC, Dayton, Ohio (N00019-27-D-0102); and Charlie Corp., Austin, Texas (N00019-27-D-0103), are each awarded a $50,000,000 indefinite-delivery/indefinite-quantity contract for engineering services.</p>
<p>Delta Sensors Corp.,* Tucson, Arizona, was awarded a $12,500,000 firm-fixed-price contract (N68335-27-C-0042) for radar test sets.</p>
<p>Work will be performed in Tucson, Arizona.</p>
</div></body></html>"""


def test_announcement_parser():
    from award_signal.announcements import parse_article, parse_listing, slug_date, to_awards
    assert slug_date("contracts-for-sept-30-2021") == pd.Timestamp("2021-09-30")
    assert slug_date("contracts-for-dec-10-2015") == pd.Timestamp("2015-12-10")
    listing = 'article-url="https://www.war.gov/News/Contracts/Contract/Article/633702/contracts-for-dec-10-2015/"'
    assert parse_listing(listing) == [("633702", "contracts-for-dec-10-2015", pd.Timestamp("2015-12-10"))]

    rows = pd.DataFrame(parse_article(ARTICLE_HTML, pd.Timestamp("2026-11-24"), "1"))
    by = rows.set_index("recipient_name")
    assert by.loc["Boeing Defense", "modification"]
    assert by.loc["Boeing Defense", "sub_agency"] == "Air Force"
    assert not by.loc["Amentum Technology Inc", "modification"]
    assert by.loc["Amentum Technology Inc", "face_value"] == 68_276_868
    assert by.loc["Amentum Technology Inc", "obligated"] == 5_000_000
    assert set(rows[rows["multi_award"]]["recipient_name"]) == {"Alpha Widgets Inc", "Bravo Systems LLC", "Charlie Corp"}
    assert by.loc["Alpha Widgets Inc", "small_business"]
    assert by.loc["Delta Sensors Corp", "sub_agency"] == "Navy"
    assert by.loc["Delta Sensors Corp", "contract_number"] == "N68335-27-C-0042"
    assert len(rows) == 6                                   # the "Work will be performed" line is skipped

    cfg = {"announcements": {"amount": "face", "include_modifications": False, "include_multi_award": False}}
    sel = to_awards(rows, cfg)
    assert set(sel["recipient_name"]) == {"Amentum Technology Inc", "Delta Sensors Corp"}


def test_one_word_names_need_review():
    tickers = pd.DataFrame({"cik": [5], "ticker": ["NVMI"], "sec_title": ["NOVA LTD."]})
    cand = build_candidates(pd.Series(["NOVA GROUP, INC."]), tickers)
    assert cand["confirmed"].iloc[0] == "" and "one-word" in cand["note"].iloc[0]
    t2 = pd.DataFrame({"cik": [7], "ticker": ["AVAV"], "sec_title": ["AeroVironment Inc"]})
    assert build_candidates(pd.Series(["AEROVIRONMENT, INC."]), t2)["confirmed"].iloc[0] == "Y"


def test_notice_company_finder_and_filter(tmp_path):
    import io as _io
    from award_signal.notices import build_name_index, estimated_value, filter_year, find_companies
    tickers = pd.DataFrame({"cik": [1, 2, 3, 4], "ticker": ["AVAV", "KTOS", "UIS", "AAPL"],
                            "sec_title": ["AeroVironment Inc", "Kratos Defense & Security Solutions, Inc.",
                                          "Unisys Corp", "Apple Inc."]})
    index = build_name_index(tickers, None)
    assert "UNISYS" in index and "APPLE" not in index          # five letters: too short alone
    text = ("Notice of Intent to Sole Source. The Army intends to award a sole source contract to "
            "AeroVironment, Inc. for Puma spares. Kratos Defense and Security Solutions may also respond.")
    assert find_companies(text, index) == ["AEROVIRONMENT", "KRATOS DEFENSE AND SECURITY SOLUTIONS"]
    assert find_companies("unisys systems are aging", index) == []   # one word without a company suffix
    assert estimated_value("estimated at $2.5 million over five years") == 2.5e6

    csv_text = ('"NoticeId","Title","Type","PostedDate","Description","Awardee"\n'
                '"a1","Notice of Intent to Sole Source - AeroVironment","Special Notice","2020-03-05","x",""\n'
                '"a2","Office furniture","Solicitation","2020-03-06","desks",""\n'
                '"a3","Award: sole source","Award Notice","2020-03-07","sole source",""\n')

    class Raw(_io.BytesIO):
        decode_content = True

    class Resp:
        def __init__(self):
            self.raw = Raw(csv_text.encode())
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Session:
        def get(self, url, stream=True, timeout=300): return Resp()

    out = tmp_path / "FY2020.csv"
    assert filter_year(2020, out, Session(), log=lambda *a: None) == 1
    assert list(pd.read_csv(out)["NoticeId"]) == ["a1"]

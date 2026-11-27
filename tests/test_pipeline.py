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


def _xlsx(rows):
    import io as _io
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active
    for r in rows:
        ws.append(r)
    buf = _io.BytesIO(); wb.save(buf)
    return buf.getvalue()


BUDGET_CFG = {"budget": {"ramp_multiple": 1.5, "ramp_min_request": 20e6, "new_start_min_request": 10e6,
                         "area_excess_growth": 0.15, "area_min_increase": 100e6,
                         "signal_month_day": "07-01"}}


def test_budget_parse_and_flags():
    from award_signal.budget import area_signals, find_files, flag_lines, line_summary, parse_exhibit
    html = ('<a href="/Portals/45/Documents/defbudget/FY2026/r1_display.xlsx">x</a>'
            '<a href="/Portals/45/Documents/defbudget/fy2020/r1.xlsx">x</a>'
            '<a href="/Portals/45/Documents/defbudget/fy2020/r1_display.xlsx">x</a>')
    assert find_files(html)["r1"].endswith("fy2020/r1.xlsx")       # full file preferred over display

    # FY2020 request, single header row
    y20 = _xlsx([["Department of Defense FY 2020 Budget"], [],
                 ["Account", "Account Title", "Organization", "Budget Activity", "Program Element",
                  "Program Element Title", "FY 2018 Actuals", "FY 2019 Enacted", "FY 2020 Request"],
                 [400, "RDT&E Army", "Army", 4, "0604123A", "Counter Small Unmanned Aircraft Systems", 5000, 10000, 60000],
                 [400, "RDT&E Army", "Army", 4, "0604555A", "Facilities Support", 90000, 95000, 96000],
                 [400, "RDT&E Army", "Army", 4, "", "Total Budget Activity", 1, 2, 3]])
    # FY2021 request, two-row header (Base / OCO / Total under each year)
    y21 = _xlsx([["FY 2021 RDT&E Programs (R-1)"],
                 ["Account", "Organization", "Program Element", "Program Element Title", "FY 2020 Enacted", None, None, "FY 2021 Request", None, None],
                 [None, None, None, None, "Base", "OCO", "Total", "Base", "OCO", "Total"],
                 [400, "Army", "0604123A", "Counter Small Unmanned Aircraft Systems", 55000, 5000, 60000, 80000, 0, 80000],
                 [400, "Army", "0604999A", "High Energy Laser Weapon Prototype", None, None, None, 45000, 0, 45000],
                 [400, "Army", "0604555A", "Facilities Support", 96000, 0, 96000, 97000, 0, 97000]])
    long = pd.concat([parse_exhibit(y20, "r1", 2020), parse_exhibit(y21, "r1", 2021)])
    summ = line_summary(long).set_index(["request_fy", "line_id"])
    assert summ.loc[(2020, "0604123A"), "enacted"] == 10_000_000
    assert summ.loc[(2020, "0604123A"), "request"] == 60_000_000
    assert summ.loc[(2021, "0604123A"), "enacted"] == 60_000_000     # the Total column, not Base + Total
    assert summ.loc[(2021, "0604999A"), "request"] == 45_000_000
    assert "Total Budget Activity" not in set(long["title"])

    flagged = flag_lines(line_summary(long), BUDGET_CFG).set_index(["request_fy", "line_id"])
    assert flagged.loc[(2020, "0604123A"), "ramp"]                    # $60M vs $10M enacted
    assert not flagged.loc[(2021, "0604123A"), "ramp"]                # $80M vs $60M is under 1.5x
    assert flagged.loc[(2021, "0604999A"), "new_start"]
    assert not flagged.loc[(2021, "0604555A"), "new_start"]

    areas = {"counter_uas": ["counter small unmanned"], "directed_energy": ["high energy laser"],
             "unmanned_autonomy": ["unmanned"]}
    sig = area_signals(flag_lines(line_summary(long), BUDGET_CFG), areas, BUDGET_CFG).set_index(["request_fy", "area"])
    assert sig.loc[(2020, "counter_uas"), "ramps"] == 1
    assert sig.loc[(2020, "unmanned_autonomy"), "lines"] == 1          # one line can count in two areas
    assert sig.loc[(2021, "directed_energy"), "new_starts"] == 1
    assert sig.loc[(2021, "directed_energy"), "signal_date"] == pd.Timestamp("2020-07-01")


def test_area_keywords_match_word_starts():
    from award_signal.budget import tag_areas
    tags = tag_areas(pd.Series(["Electronic Warfare Development", "New Equipment Review", "EW Planning"]),
                     {"electronic_warfare": ["electronic warfare", "ew"]})
    assert list(tags) == [["electronic_warfare"], [], ["electronic_warfare"]]


def test_budget_reads_hand_saved_files(tmp_path):
    from award_signal.budget import local_files
    (tmp_path / "FY2026").mkdir()
    (tmp_path / "FY2026" / "r1_display.xlsx").write_bytes(b"PK")
    (tmp_path / "FY2026" / "p1.xlsx").write_bytes(b"PK")
    found = local_files(tmp_path, 2026)
    assert found["r1"].name == "r1_display.xlsx" and found["p1"].name == "p1.xlsx"
    assert local_files(tmp_path, 2025) == {}


def test_rising_area_is_judged_against_department_growth():
    from award_signal.budget import area_signals
    flagged = pd.DataFrame([
        {"request_fy": 2022, "exhibit": "r1", "line_id": "A", "title": "Hypersonic Prototype", "enacted": 500e6,
         "request": 900e6, "ramp": False, "new_start": False},
        {"request_fy": 2022, "exhibit": "r1", "line_id": "B", "title": "Facilities", "enacted": 9_500e6,
         "request": 9_600e6, "ramp": False, "new_start": False},
        {"request_fy": 2022, "exhibit": "r1", "line_id": "C", "title": "Space Sensor", "enacted": 1_000e6,
         "request": 1_050e6, "ramp": False, "new_start": False}])
    sig = area_signals(flagged, {"hypersonics": ["hypersonic"], "space": ["space"]}, BUDGET_CFG).set_index("area")
    assert sig.loc["hypersonics", "rising"]          # +80% against about +4% overall
    assert not sig.loc["space", "rising"]            # +5%: barely above the department, and only $50M


def _stage_market(effect):
    from award_signal.synthetic import DictProvider, StubFundamentals, _frame
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2014-01-01", "2026-11-20")
    bench = _frame(dates, 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))), rng, 3e7)
    areas = ["hypersonics", "space", "cyber", "undersea", "munitions", "nuclear"]
    frames, index, rows, revenue, shares = {"IWM": bench}, {}, [], {}, {}
    sig_rows = []
    rising_by_year = {y: set(rng.choice(areas, size=2, replace=False)) for y in range(2017, 2027)}
    for k in range(36):
        area = areas[k % len(areas)]
        tk, cik, name = f"ST{k:02d}", 500 + k, f"Stage Test Company {k:02d} Inc"
        index[normalize(name)] = {"cik": cik, "ticker": tk, "sec_title": name, "delisting_date": None}
        shares[cik] = 2e7
        for y in range(2015, 2026):
            rows.append({"recipient_name": name, "face_value": 3e7, "multi_award": False, "modification": False,
                         "description": f"{area} work", "action_date": f"{y}-03-15"})
        lr = np.log1p(bench["close"].pct_change().fillna(0).to_numpy()) + rng.normal(-0.5 * 0.02**2, 0.02, len(dates))
        for y, ris in rising_by_year.items():
            if area in ris:
                i = int(dates.searchsorted(pd.Timestamp(f"{y - 1}-07-01"), side="right"))
                lr[i + 1:i + 253] += np.log1p(effect) / 252
        frames[tk] = _frame(dates, 20 * np.exp(np.cumsum(lr)), rng, 5e5)
    for y, ris in rising_by_year.items():
        for a in areas:
            sig_rows.append({"request_fy": y, "area": a, "rising": a in ris, "signal_date": pd.Timestamp(f"{y - 1}-07-01")})
    return (pd.DataFrame(rows), index, pd.DataFrame(sig_rows), DictProvider(frames),
            StubFundamentals({}, shares), {a: [a] for a in areas})


STAGE_CFG = {**CFG, "stages": {"lookback_years": 3, "exposure_share": 0.25, "exposure_min_dollars": 7.5e6,
                               "hold_days": 252}}


def test_stages_find_planted_rising_areas():
    from award_signal.stages import build_positions, company_awards, portfolio_by_year, random_area_placebo
    raw, index, signals, provider, fund, areas = _stage_market(effect=0.30)
    awards = company_awards(raw, index, areas)
    assert awards["key"].nunique() == 36
    pos = build_positions(signals, awards, fund, provider, STAGE_CFG, log=lambda *a: None)
    traded = pos[(pos["status"] == "traded") & pos["rising"]]
    assert traded["request_fy"].nunique() >= 9
    assert portfolio_by_year(pos)["mean_net_excess"].mean() > 0.1
    assert random_area_placebo(pos, signals, STAGE_CFG)["passes_95pct"]


def test_stages_null_fails_placebo():
    from award_signal.stages import build_positions, company_awards, random_area_placebo
    raw, index, signals, provider, fund, areas = _stage_market(effect=0.0)
    pos = build_positions(signals, company_awards(raw, index, areas), fund, provider, STAGE_CFG, log=lambda *a: None)
    assert not random_area_placebo(pos, signals, STAGE_CFG)["passes_95pct"]


def test_exposure_uses_only_earlier_awards():
    from award_signal.stages import exposure
    aw = pd.DataFrame({"key": ["A", "A", "A"], "ticker": "AAA", "cik": 1, "delisting_date": pd.NaT,
                       "action_date": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-08-01"]),
                       "amount": [10e6, 10e6, 500e6], "areas": [["space"], ["space"], ["cyber"]],
                       "recipient_name": "A Co"})
    ex = exposure(aw, pd.Timestamp("2021-07-01"), STAGE_CFG).set_index("area")
    assert ex.loc["space", "share"] == 1.0 and "cyber" not in ex.index   # the August 2021 award is not known yet


# ---------------------------------------------------------------- stage 2, version 2

def _v2_index():
    from award_signal.notices import build_name_index
    tickers = pd.DataFrame({"cik": [10, 11, 12, 13], "ticker": ["LMT", "VTX", "VVX", "KTOS"],
                            "sec_title": ["Lockheed Martin Corp", "Vertex, Inc.", "V2X, Inc.",
                                          "Kratos Defense & Security Solutions, Inc."]})
    all_names = pd.DataFrame({"sec_title": ["VECTRUS, INC.", "LOCKHEED MARTIN CORP"], "cik": [12, 10]})
    return tickers, all_names, build_name_index(tickers, None, all_names)


def test_v2_matcher_rules():
    from award_signal.match2 import Matcher, match_table
    tickers, all_names, index = _v2_index()
    subs = pd.DataFrame({"key": [normalize("Kratos Unmanned Aerial Systems, Inc.")], "cik": [13],
                         "valid_from": [pd.Timestamp("2016-03-01")], "valid_until": [pd.Timestamp("2018-03-01")]})
    m = Matcher(index, tickers, all_names, subs)
    names = pd.Series(["Lockheed Martin Rotary and Mission Systems", "Vectrus Systems Corp", "Vertex Aerospace LLC",
                       "Kratos Unmanned Aerial Systems Inc", "Vertex Inc", "Unrelated Widgets LLC"])
    mt = match_table(names, m).set_index("recipient_name")
    assert mt.loc["Lockheed Martin Rotary and Mission Systems", "ticker"] == "LMT"
    assert mt.loc["Lockheed Martin Rotary and Mission Systems", "match_type"] == "prefix"
    assert mt.loc["Vectrus Systems Corp", "ticker"] == "VVX"                   # former name, then prefix
    assert mt.loc["Vectrus Systems Corp", "former_name"] == "VECTRUS"
    assert mt.loc["Vectrus Systems Corp", "one_word_unconfirmed"]
    assert mt.loc["Kratos Unmanned Aerial Systems Inc", "match_type"] == "subsidiary"
    assert mt.loc["Kratos Unmanned Aerial Systems Inc", "valid_from"] == pd.Timestamp("2016-03-01")
    assert "Unrelated Widgets LLC" not in mt.index
    # one-word prefix: confirmed when the company wins awards under its own exact name
    assert not mt.loc["Vertex Aerospace LLC", "one_word_unconfirmed"]
    mt2 = match_table(names.drop(4), Matcher(index, tickers, all_names, subs)).set_index("recipient_name")
    assert mt2.loc["Vertex Aerospace LLC", "one_word_unconfirmed"]


def test_v2_company_rules_date_former_names_and_drop_industries():
    from award_signal.match2 import Matcher, apply_company_rules, match_table
    from award_signal.widen import DROP_SIC
    tickers, all_names, index = _v2_index()
    mt = match_table(pd.Series(["Vectrus Systems Corp", "Lockheed Martin Space", "Vertex Aerospace LLC"]),
                     Matcher(index, tickers, all_names))
    info = pd.DataFrame({"cik": [10, 11, 12], "sic": ["6331", "5961", "8744"]})
    former = pd.DataFrame({"cik": [12], "key": ["VECTRUS"], "from": [pd.Timestamp("2014-05-01")],
                           "to": [pd.Timestamp("2022-07-05")]})
    out = apply_company_rules(mt, info, former, DROP_SIC).set_index("recipient_name")
    assert "Lockheed Martin Space" not in out.index                          # insurance SIC: dropped
    assert "Vertex Aerospace LLC" not in out.index      # unconfirmed one-word prefix, retail SIC: dropped
    assert out.loc["Vectrus Systems Corp", "valid_until"] == pd.Timestamp("2023-07-05")
    no_dates = apply_company_rules(mt, info, former.iloc[0:0], DROP_SIC)
    assert "Vectrus Systems Corp" not in set(no_dates["recipient_name"])   # undated former name: not used


def test_v2_announcement_text_cleanup():
    from award_signal.widen import announcement_tag_text
    t = ("Acme Corp., Huntsville, Alabama, is awarded $9,000,000 for IT help desk support. Work will be performed "
         "in San Diego, California, and is expected to be completed by June 2027. Fiscal 2024 Space Force "
         "operations funds of $1,000,000 are being obligated at the time of award. Space and Naval Warfare "
         "Systems Command, San Diego, California, is the contracting activity (N00039-24-C-0001).")
    clean = announcement_tag_text(t).lower()
    assert "help desk" in clean and "space" not in clean
    t2 = "Beta Inc., is awarded $8,000,000 for hypersonic glide body testing at White Sands Missile Range."
    assert "hypersonic" in announcement_tag_text(t2).lower() and "missile" not in announcement_tag_text(t2).lower()


def test_v2_parse_ex21_html_and_text():
    from award_signal.widen import parse_ex21
    html = ("<p>SUBSIDIARIES OF THE REGISTRANT</p><table><tr><td>Name</td><td>Jurisdiction</td></tr>"
            "<tr><td>Kratos Unmanned Aerial Systems, Inc.</td><td>California</td></tr>"
            "<tr><td>Kratos SRE, Inc. (100%)</td><td>Delaware</td></tr>"
            "<tr><td>Delaware</td><td></td></tr></table>")
    assert parse_ex21(html) == ["Kratos Unmanned Aerial Systems, Inc.", "Kratos SRE, Inc."]
    text = "Exhibit 21\nAdvanced Acoustic Concepts, LLC      New York\nMicrosemi Frequency Products ...... Texas\n"
    assert parse_ex21(text) == ["Advanced Acoustic Concepts, LLC", "Microsemi Frequency Products"]


def test_v2_subsidiary_windows_and_ambiguity():
    from award_signal import widen

    class FakeHttp:
        def json(self, method, url, body=None):
            if "submissions" in url:
                cik = int(url.split("CIK")[1][:10])
                return {"filings": {"recent": {"form": ["10-K", "10-K", "10-K"],
                                                "filingDate": ["2015-03-01", "2017-03-01", "2019-03-01"],
                                                "accessionNumber": [f"{cik}-15", f"{cik}-17", f"{cik}-19"]}}}
            return {"directory": {"item": [{"name": "ex21.htm"}]}}

        def text(self, url):
            acc = url.split("/")[-2]
            cik = url.split("/")[-3]
            lists = {"1": {"115": "Alpha Sub Inc", "117": "Alpha Sub Inc\nShared Name Inc", "119": "Beta Sub Inc"},
                     "2": {"215": "Shared Name Inc", "217": "Shared Name Inc", "219": ""}}
            return lists[cik][acc]

    subs = widen.build_subsidiaries(FakeHttp(), [1, 2], log=lambda *a: None).set_index("key")
    assert subs.loc["ALPHA SUB", "valid_from"] == pd.Timestamp("2015-03-01")
    assert subs.loc["ALPHA SUB", "valid_until"] == pd.Timestamp("2019-03-01")   # dropped from the 2019 list
    assert subs.loc["BETA SUB", "valid_until"] == pd.Timestamp("2021-03-01")
    assert "SHARED NAME" not in subs.index                                   # two parents in 2015-2017


def test_v2_sources_sbir_sam_and_exposure_lag(tmp_path):
    from award_signal import widen
    from award_signal.stages import company_awards_v2, exposure
    p = tmp_path / "sbir.csv"
    pd.DataFrame({"Company": ["Acme Corp", "Acme Corp", "Acme Corp"], "Award Title": ["Counter-UAS radar", "x", "y"],
                  "Agency": ["Department of Defense", "Department of Energy", "Department of Defense"],
                  "Proposal Award Date": ["2019-02-01", "2019-02-01", "2019-03-01"],
                  "Award Amount": ["$1,500,000", "900000", "9,000,000"], "Contract": ["FA8650-19-C-1111", "", ""]
                  }).to_csv(p, index=False)
    sbir = widen.load_sbir(p)
    assert len(sbir) == 1 and sbir["available_date"].iloc[0] == pd.Timestamp("2020-02-01")
    assert widen.keep_award_notice({"Type": "Award Notice", "Department/Ind.Agency": "DEPT OF DEFENSE", "Award$": "250000"})
    assert not widen.keep_award_notice({"Type": "Award Notice", "Department/Ind.Agency": "DEPT OF DEFENSE", "Award$": "9000000"})
    sam = pd.DataFrame({"source": "sam", "recipient_name": "Acme Corp", "action_date": [pd.Timestamp("2019-06-01")],
                        "available_date": [pd.Timestamp("2019-06-01")], "amount": [2e6], "text": ["Counter-UAS kit"],
                        "contract": ["FA8650-19-C-1111"]})
    src = widen.combine_sources(sam.iloc[0:0], sam, sbir)
    assert list(src["source"]) == ["sam"]                                    # same contract posted twice
    src = widen.combine_sources(sam.iloc[0:0], sam.assign(contract="OTHER"), sbir)
    mt = pd.DataFrame({"recipient_name": ["Acme Corp"], "ticker": ["ACME"], "cik": [9], "delisting_date": [None],
                       "match_type": ["exact"], "valid_from": [pd.Timestamp("1900-01-01")],
                       "valid_until": [pd.Timestamp("2200-01-01")]})
    aw = company_awards_v2(src, mt, {"counter_uas": ["counter-uas"]})
    ex = exposure(aw, pd.Timestamp("2019-07-01"), STAGE_CFG)
    assert ex["area_dollars"].sum() == 2e6           # the SBIR award is not public until February 2020
    ex = exposure(aw, pd.Timestamp("2020-07-01"), STAGE_CFG)
    assert ex["area_dollars"].sum() == 3.5e6


def test_v2_one_word_prefix_only_for_announcement_names():
    from award_signal.match2 import Matcher, apply_company_rules, match_table
    from award_signal.widen import DROP_SIC, parse_ex21
    tickers, all_names, index = _v2_index()
    names = pd.Series(["Vertex Aerospace LLC", "VERTEX SOLUTIONS, LLC", "Vertex Inc"])
    mt = match_table(names, Matcher(index, tickers, all_names))
    info = pd.DataFrame({"cik": [11], "sic": ["3720"]})
    out = apply_company_rules(mt, info, pd.DataFrame(columns=["cik", "key", "from", "to"]), DROP_SIC,
                              {"Vertex Aerospace LLC", "Vertex Inc"})
    assert set(out["recipient_name"]) == {"Vertex Aerospace LLC", "Vertex Inc"}
    assert parse_ex21("<tr><td>The State of California</td></tr><tr><td>Acme Widgets LLC</td></tr>") == ["Acme Widgets LLC"]


def test_timing_check_finds_a_release_jump():
    from award_signal.synthetic import DictProvider, StubFundamentals, _frame
    from award_signal.timing import RELEASE_DATES, summarize, window_rows
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2014-01-01", "2026-11-20")
    bench = _frame(dates, 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))), rng, 3e7)
    areas = ["hypersonics", "space", "cyber", "undersea", "munitions", "nuclear"]
    rising_by_year = {y: set(rng.choice(areas, size=2, replace=False)) for y in range(2017, 2027)}
    frames, rows, shares, aw = {"IWM": bench}, [], {}, []
    for k in range(36):
        area, tk, cik = areas[k % 6], f"TT{k:02d}", 700 + k
        shares[cik] = 2e7
        lr = np.log1p(bench["close"].pct_change().fillna(0).to_numpy()) + rng.normal(0, 0.02, len(dates))
        for y, ris in rising_by_year.items():
            if area in ris:
                i = int(dates.searchsorted(pd.Timestamp(RELEASE_DATES[y])))
                lr[i] += np.log(1.15)                       # a 15% jump on release day
        frames[tk] = _frame(dates, 20 * np.exp(np.cumsum(lr)), rng, 5e5)
        for y in range(2014, 2026):
            aw.append({"key": str(cik), "ticker": tk, "cik": cik, "delisting_date": pd.NaT,
                       "action_date": pd.Timestamp(f"{y}-01-15"), "available_date": pd.Timestamp(f"{y}-01-15"),
                       "amount": 3e7, "areas": [area], "recipient_name": tk})
    sig = pd.DataFrame([{"request_fy": y, "area": a, "rising": a in ris, "signal_date": pd.Timestamp(f"{y - 1}-07-01")}
                        for y, ris in rising_by_year.items() for a in areas])
    out = window_rows(sig, pd.DataFrame(aw), StubFundamentals({}, shares), DictProvider(frames), STAGE_CFG,
                      log=lambda *a: None)
    s = summarize(out, sig, STAGE_CFG)
    assert s["release"]["passes_95pct"] and not s["pre"]["passes_95pct"]
    assert "earlier sources" in s["reading"]


def test_v3_rule_trades_the_release_window_and_band_fallback():
    from award_signal.synthetic import DictProvider, _frame
    from award_signal.v3 import RELEASE_DATES, judge, shares_for_band, trades
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"filed": "2012-03-01", "end": "2011-12-31", "val": 5e6}]}}}}}
    assert shares_for_band(facts, pd.Timestamp("2011-06-01")) == 5e6      # first count filed within 18 months
    assert shares_for_band(facts, pd.Timestamp("2010-01-01")) is None
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2007-01-01", "2026-11-20")
    bench = _frame(dates, 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))), rng, 3e7)
    areas = ["hypersonics", "space", "cyber", "undersea", "munitions", "nuclear"]
    rising_by_year = {y: set(rng.choice(areas, size=2, replace=False)) for y in range(2009, 2017)}
    frames, aw = {"IWM": bench}, []

    class Fund:
        def facts(self, cik):
            return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
                {"filed": "2006-01-01", "end": "2005-12-31", "val": 2e7}]}}}}}

    for k in range(36):
        area, tk, cik = areas[k % 6], f"HO{k:02d}", 900 + k
        lr = np.log1p(bench["close"].pct_change().fillna(0).to_numpy()) + rng.normal(0, 0.02, len(dates))
        for y, ris in rising_by_year.items():
            if area in ris:
                i = int(dates.searchsorted(pd.Timestamp(RELEASE_DATES[y]))) + 10
                lr[i] += np.log(1.12)                      # a 12% rise inside the window
        frames[tk] = _frame(dates, 20 * np.exp(np.cumsum(lr)), rng, 5e5)
        for y in range(2005, 2016):
            aw.append({"key": str(cik), "ticker": tk, "cik": cik, "delisting_date": pd.NaT,
                       "action_date": pd.Timestamp(f"{y}-01-05"), "available_date": pd.Timestamp(f"{y}-01-06"),
                       "amount": 3e7, "areas": [area], "recipient_name": tk})
    sig = pd.DataFrame([{"request_fy": y, "area": a, "rising": a in ris, "signal_date": pd.Timestamp(f"{y - 1}-07-01")}
                        for y, ris in rising_by_year.items() for a in areas])
    cfg = {**STAGE_CFG, "stages": {**STAGE_CFG["stages"], "min_positive_share": 0.6}}
    rows = trades(sig, pd.DataFrame(aw), Fund(), DictProvider(frames), cfg, log=lambda *a: None)
    res = judge(rows, sig, cfg)
    assert res["passed"] and res["yearly_test"]["years"] == 8


def test_v4_line_keys_and_links():
    from award_signal.lines import line_keys, link_awards
    assert ("MSV", True) in line_keys("Maneuver Support Vessel (MSV)")
    assert ("MK-54", True) in line_keys("MK-54 Torpedo Mods")
    assert ("F-35", True) in line_keys("F-35 STOVL")
    assert line_keys("Physical Security Equipment") == []
    assert all(k != "Medical" for k, _ in line_keys("Readiness Support (Medical)"))
    lines = pd.DataFrame({"request_fy": [2024, 2024], "exhibit": ["p1", "p1"], "line_id": ["A", "B"],
                          "title": ["Maneuver Support Vessel (MSV)", "F-35 STOVL"], "ramp": [True, False],
                          "new_start": [False, False]})
    aw = pd.DataFrame({"text": ["Acme Corp. is awarded $9,000,000 for MSV hull work.",
                                "Beta Inc. is awarded $8,000,000 for F-35 parts.", "Gamma for msv-like work."],
                       "recipient_name": ["Acme", "Beta", "Gamma"]})
    ln, links = link_awards(lines, aw)
    assert set(map(tuple, links.to_numpy())) == {("msv", 0), ("f-35", 1)}

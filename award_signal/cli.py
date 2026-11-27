"""Command line: python -m award_signal <command>

    awards   download DoD awards (or load a CSV) into data/awards.csv
    match    match award recipients to SEC-listed companies
    widen    stage 2 version 2 data: subsidiaries, SBIR, SAM award notices (no returns)
    backtest stages 2 and 3 (--rules v1 or v2)
    run      build events, run the event study, write output/report.html
    demo     run the whole pipeline on synthetic data (no internet needed)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import awards as awards_mod
from .config import load_config
from .events import build_events
from .matching import build_candidates, load_map
from .net import Http
from .prices import make_provider
from .report import write_outputs
from .study import gate_check, placebo_test, run_study, worst_case, yearly_summary


def analyze(awards: pd.DataFrame, rmap: pd.DataFrame, fundamentals, provider, cfg: dict,
            out_dir: str | Path, log=print) -> dict:
    log(f"Awards: {len(awards):,}; confirmed recipient matches: {len(rmap):,}")
    events = build_events(awards, rmap, fundamentals, cfg, log=log)
    if events.empty:
        log("No events to test.")
        return {}
    log(f"Events: {len(events):,} matched; {int(events['passes_ratio'].sum()):,} pass the ratio test")
    res = run_study(events, provider, cfg, delay=0)
    delayed = run_study(events, provider, cfg, delay=int(cfg["signal"]["delay_test_days"]))
    yearly = yearly_summary(res, cfg)
    gate = gate_check(yearly, cfg)
    placebo = placebo_test(res, cfg)
    wc = worst_case(res, cfg)
    summary = write_outputs(out_dir, res, delayed, yearly, gate, placebo, cfg, wc)
    log(f"Trades: {summary['trades']}; yearly test {gate['years_beating_benchmark']} of {gate['test_years']} "
        f"years beat the benchmark (needs {gate['years_required']})")
    if "share_of_placebos_beaten" in placebo:
        log(f"Placebo: real signals beat {placebo['share_of_placebos_beaten']:.0%} of random-date portfolios")
    wg = wc["gate"]
    log(f"Survivorship check: {wc['unpriced_trades']} unpriced trades counted as total losses; "
        f"{wg['years_beating_benchmark']} of {wg['test_years']} test years still beat the benchmark")
    log(f"Phase 1 {'PASSED' if summary['phase1_passed'] else 'NOT PASSED'} (yearly test and placebo test both required)")
    log(f"Report: {Path(out_dir) / 'report.html'}")
    return summary


def _http(cfg: dict, sub: str) -> Http:
    # The SEC_USER_AGENT environment variable, when set, overrides config.yaml.
    ua = os.environ.get("SEC_USER_AGENT", "").strip() or cfg["sec_user_agent"]
    if "example.com" in ua:
        sys.exit("Set the SEC_USER_AGENT environment variable, or sec_user_agent in config.yaml, "
                 "to a name and contact email first (SEC requires it).")
    return Http(Path(cfg["data_dir"]) / "cache" / sub, ua)


def cmd_awards(cfg: dict) -> None:
    out = Path(cfg["data_dir"]) / "awards.csv"
    source = cfg["awards"]["source"]
    if source == "csv":
        df = awards_mod.load_awards_csv(cfg["awards"]["csv_path"], cfg)
    elif source == "announcements":
        from .announcements import fetch_announcements, to_awards

        print("Reading DoD daily contract announcements from war.gov (four pages at a time)...")
        http = Http(Path(cfg["data_dir"]) / "cache" / "wargov",
                    "award-signal research", min_interval=0.3)
        raw = fetch_announcements(cfg, http)
        raw.to_csv(Path(cfg["data_dir"]) / "announcements_raw.csv", index=False)
        n_mod = int(raw["modification"].sum()) if len(raw) else 0
        n_multi = int(raw["multi_award"].sum()) if len(raw) else 0
        print(f"{len(raw):,} award paragraphs: {n_mod:,} modifications, {n_multi:,} multiple-award rows")
        sel = to_awards(raw, cfg)
        for col in ("recipient_uei",):
            sel = sel.assign(**{col: None})
        df = awards_mod.finalize(sel, cfg)
    else:
        print("Downloading DoD awards from USAspending, one month at a time...")
        df = awards_mod.fetch_usaspending(cfg, _http(cfg, "usaspending"))
        skipped = Path(cfg["data_dir"]) / "awards_skipped.csv"
        pd.DataFrame(df.attrs.get("skipped", []), columns=["start", "end", "page", "error"]).to_csv(skipped, index=False)
    df.to_csv(out, index=False)
    df.to_csv(Path(cfg["data_dir"]) / f"awards_{source}.csv", index=False)
    print(f"Saved {len(df):,} awards to {out}")


def cmd_match(cfg: dict) -> None:
    from .sec import load_all_names, load_av_delisted, load_tickers

    aw = pd.read_csv(Path(cfg["data_dir"]) / "awards.csv")
    http = _http(cfg, "sec")
    tickers = load_tickers(http)
    print("Loading SEC's full filer list (large file, downloaded once)...")
    all_names = load_all_names(http)
    delisted = None
    key = str(cfg.get("alphavantage_api_key") or "").strip()
    if key:
        delisted = load_av_delisted(Http(Path(cfg["data_dir"]) / "cache" / "alphavantage", "award-signal"), key)
        print(f"Alpha Vantage delisted list: {len(delisted):,} stocks")
    else:
        print("No alphavantage_api_key in config.yaml: delisted companies will match SEC names but get no ticker.")
    cand = build_candidates(aw["recipient_name"], tickers, all_names, delisted)
    out = Path(cfg["data_dir"]) / "recipient_map_candidates.csv"
    cand.to_csv(out, index=False)
    exact = int((cand["confirmed"] == "Y").sum())
    gone = int((cand["source"] == "sec_all_names+av_delisted").sum())
    print(f"{len(cand):,} candidate matches ({exact:,} confirmed automatically; "
          f"{gone:,} are delisted companies) saved to {out}")
    print("Review rows with a blank 'confirmed' column (see the 'note' column), put Y in the right ones, "
          "and save the file as data/recipient_map.csv. Without that file, only automatic matches are used.")


def cmd_run(cfg: dict, awards_file: str | None = None, map_file: str | None = None) -> None:
    from .sec import Fundamentals

    aw_path = Path(awards_file) if awards_file else Path(cfg["data_dir"]) / "awards.csv"
    aw = pd.read_csv(aw_path, parse_dates=["action_date", "signal_date"])
    if map_file:
        rmap = pd.read_csv(map_file, dtype={"confirmed": str, "ticker": str})
        rmap = rmap[rmap["confirmed"].fillna("").str.upper().str.startswith("Y")]
        rmap["delisting_date"] = pd.to_datetime(rmap.get("delisting_date"), errors="coerce")
        rmap = rmap.dropna(subset=["cik", "ticker"])
    else:
        rmap = load_map(cfg["data_dir"])
    analyze(aw, rmap, Fundamentals(_http(cfg, "sec")), make_provider(cfg), cfg, cfg["output_dir"])


def cmd_budget(cfg: dict) -> None:
    """Load DoD budget requests and flag rising technology areas (stage 1)."""
    import yaml

    from .budget import area_signals, fetch_all, flag_lines, line_summary

    areas = yaml.safe_load(Path(cfg["budget"]["areas_file"]).read_text(encoding="utf-8"))["areas"]
    http = Http(Path(cfg["data_dir"]) / "cache" / "comptroller", "award-signal research", min_interval=0.5)
    print("Reading DoD budget requests (R-1 and P-1) from the Comptroller site...")
    long = fetch_all(cfg, http)
    if long.empty:
        sys.exit("No budget files could be read.")
    data = Path(cfg["data_dir"])
    long.to_csv(data / "budget_lines_long.csv", index=False)
    flagged = flag_lines(line_summary(long), cfg)
    sig = area_signals(flagged, areas, cfg)
    out = Path(cfg["output_dir"]) / "budget"
    out.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out / "lines_flagged.csv", index=False)
    sig.to_csv(out / "areas_by_year.csv", index=False)
    for rfy, g in sig[sig["rising"]].groupby("request_fy"):
        print(f"  FY{rfy} request (signal {g['signal_date'].iloc[0].date()}): " + ", ".join(g["area"]))
    print(f"Saved {out / 'areas_by_year.csv'} and {out / 'lines_flagged.csv'}")


def _name_inputs(cfg: dict):
    from .notices import build_name_index
    from .sec import load_all_names, load_av_delisted, load_tickers

    data = Path(cfg["data_dir"])
    http = _http(cfg, "sec")
    key = str(cfg.get("alphavantage_api_key") or "").strip()
    delisted = load_av_delisted(Http(data / "cache" / "alphavantage", "award-signal"), key) if key else None
    tickers, all_names = load_tickers(http), load_all_names(http)
    return http, tickers, all_names, build_name_index(tickers, delisted, all_names)


def cmd_widen(cfg: dict) -> None:
    """Stage 2, version 2 data: subsidiaries, SBIR, SAM award notices, wider matching.
    Writes the matched award history and a match review file. Computes no returns."""
    import requests
    import yaml

    from . import widen
    from .match2 import Matcher, apply_company_rules, match_table
    from .stages import company_awards_v2

    data = Path(cfg["data_dir"])
    out = Path(cfg["output_dir"]) / "stages_v2"
    out.mkdir(parents=True, exist_ok=True)
    areas = yaml.safe_load(Path(cfg["budget"]["areas_file"]).read_text(encoding="utf-8"))["areas"]
    http, tickers, all_names, index = _name_inputs(cfg)
    http.min_interval = 0.11                       # SEC allows 10 requests a second

    subs_path = data / "subsidiaries.csv"
    if subs_path.exists():
        subs = pd.read_csv(subs_path, parse_dates=["valid_from", "valid_until"])
    else:
        print("Finding companies with public float between $50M and $10B (SEC frames)...")
        scope = widen.float_ciks(http)
        print(f"  {len(scope):,} companies; reading their SEC profiles for industry codes...")
        info, _ = widen.company_info(http, sorted(scope))
        scope = sorted(info.loc[info["sic"].map(lambda c: widen._in(c, widen.EX21_SIC)), "cik"].astype(int))
        print(f"  {len(scope):,} in defense-related industries; reading Exhibit 21 subsidiary lists...")
        subs = widen.build_subsidiaries(http, scope)
        subs.to_csv(subs_path, index=False)
    print(f"Subsidiary names: {len(subs):,} for {subs['cik'].nunique() if len(subs) else 0:,} companies")

    session = requests.Session()
    session.headers["User-Agent"] = "award-signal research"
    (data / "sbir").mkdir(exist_ok=True)
    try:
        sbir = widen.load_sbir(data / "sbir" / "award_data_no_abstract.csv", session)
    except Exception as exc:
        print(f"SBIR file not available ({exc}). Save award_data_no_abstract.csv from sbir.gov in data\\sbir and rerun.")
        sbir = widen.load_sbir(data / "sbir" / "missing.csv")
    sam_files = sorted((data / "notices").glob("FY*_awards.csv"))
    sam = widen.load_sam_awards(sam_files)
    raw = pd.read_csv(data / "announcements_raw.csv", low_memory=False)
    if raw["description"].fillna("").str.len().max() <= 300:
        sys.exit("announcements_raw.csv still holds 300-character descriptions: run run_announcements.bat first.")
    sources = widen.combine_sources(widen.announcement_rows(raw), sam, sbir)
    print(f"Award rows: {sources['source'].value_counts().to_dict()}")

    matcher = Matcher(index, tickers, all_names, subs)
    mt = match_table(sources["recipient_name"], matcher)
    info, former = widen.company_info(http, sorted(mt["cik"].astype(int).unique()))
    mt = apply_company_rules(mt, info, former, widen.DROP_SIC,
                             set(sources.loc[sources["source"] == "announcements", "recipient_name"]))
    mt.to_csv(data / "v2_matches.csv", index=False)
    aw = company_awards_v2(sources, mt, areas)
    aw.assign(areas=aw["areas"].map(";".join)).to_csv(data / "v2_awards.csv", index=False)

    review = aw.groupby(["cik", "ticker"]).agg(
        dollars=("amount", "sum"), awards=("amount", "size"),
        sources=("source", lambda x: ", ".join(sorted(set(x)))),
        match_types=("match_type", lambda x: ", ".join(sorted(set(x)))),
        recipient_names=("recipient_name", lambda x: " | ".join(sorted(set(x))[:6]))).reset_index()
    review = review.merge(info[["cik", "name", "sic", "sic_description"]], on="cik", how="left")
    review.sort_values("dollars", ascending=False).to_csv(out / "match_review.csv", index=False)
    print(f"{len(aw):,} award rows matched to {aw['key'].nunique():,} public companies "
          f"(match types: {aw['match_type'].value_counts().to_dict()})")
    print(f"Saved data/v2_awards.csv and {out / 'match_review.csv'} (no returns computed)")


def cmd_backtest(cfg: dict, rules: str = "v1") -> None:
    """Stages 2 and 3: exposed companies for each rising area, then the 12-month backtest."""
    import json

    import yaml

    from .report import _table
    from .sec import Fundamentals
    from .stages import build_positions, company_awards, portfolio_by_year, random_area_placebo

    data = Path(cfg["data_dir"])
    out = Path(cfg["output_dir"]) / ("stages" if rules == "v1" else "stages_v2")
    out.mkdir(parents=True, exist_ok=True)
    sig_path = Path(cfg["output_dir"]) / "budget" / "areas_by_year.csv"
    if not sig_path.exists():
        sys.exit("Run `python -m award_signal budget` first.")
    signals = pd.read_csv(sig_path, parse_dates=["signal_date"])
    signals = signals[signals["request_fy"] >= int(cfg["stages"]["first_request_fy"])]
    areas = yaml.safe_load(Path(cfg["budget"]["areas_file"]).read_text(encoding="utf-8"))["areas"]
    if rules == "v1":
        http, _, _, index = _name_inputs(cfg)
        raw = pd.read_csv(data / "announcements_raw.csv", low_memory=False)
        raw["description"] = raw["description"].fillna("").str[:300]   # v1 tagged the first 300 characters
        awards = company_awards(raw, index, areas)
    else:
        http = _http(cfg, "sec")
        awards = pd.read_csv(data / "v2_awards.csv", parse_dates=["action_date", "available_date", "delisting_date"],
                             dtype={"key": str, "ticker": str}, keep_default_na=False, na_values=[""])
        awards["areas"] = awards["areas"].fillna("").map(lambda x: [a for a in str(x).split(";") if a])
        excl = yaml.safe_load(Path("match_exclusions_v2.yaml").read_text(encoding="utf-8")) or {}
        drop = {int(e["cik"]) for e in excl.get("exclude", []) or []}
        drop_names = {str(e["name"]) for e in excl.get("exclude_recipients", []) or []}
        print(f"Excluding {len(drop)} companies and {len(drop_names)} recipient names listed in match_exclusions_v2.yaml")
        awards = awards[~awards["cik"].astype(int).isin(drop) & ~awards["recipient_name"].isin(drop_names)]
    print(f"{len(awards):,} award rows matched to {awards['key'].nunique():,} public companies")
    pos = build_positions(signals, awards, Fundamentals(http), make_provider(cfg), cfg)
    pos.to_csv(out / "positions.csv", index=False)
    years = portfolio_by_year(pos)
    years.to_csv(out / "by_year.csv", index=False)
    placebo = random_area_placebo(pos, signals, cfg)
    done = years[years["request_fy"] <= int(signals["request_fy"].max()) - 1]
    positive = int((done["mean_net_excess"] > 0).sum())
    need = int(np.ceil(cfg["stages"]["min_positive_share"] * len(done))) if len(done) else 0
    current = pos[(pos["request_fy"] == signals["request_fy"].max()) & pos["rising"]].copy()
    current.to_csv(out / "current_picks.csv", index=False)
    traded = pos[(pos["status"] == "traded") & pos["rising"]].drop_duplicates(["request_fy", "ticker"])
    summary = {"author": "Tov Binyamin Brog", "rules": rules, "companies_matched": int(awards["key"].nunique()),
               "positions": int(len(traded)), "mean_net_excess": float(traded["net_excess"].mean()) if len(traded) else None,
               "status_counts": pos[pos["rising"]]["status"].value_counts().to_dict(),
               "yearly_test": {"completed_years": int(len(done)), "years_beating_iwm": positive, "years_required": need,
                               "passed": bool(len(done) and positive >= need)},
               "placebo": placebo,
               "passed": bool(len(done) and positive >= need and placebo.get("passes_95pct", False))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='author' content='Tov Binyamin Brog'>"
            "<title>Award Signal - stages 2 and 3</title></head><body style='font-family:system-ui;max-width:1000px;margin:2rem auto'>"
            f"<h1>Rising-area backtest</h1><p>Tov Binyamin Brog</p><pre>{json.dumps(summary, indent=2, default=str)}</pre>"
            f"<h2>By signal year</h2>{_table(years.round(4))}"
            f"<h2>Current picks (latest request)</h2>{_table(current[['area', 'ticker', 'share', 'market_cap', 'status', 'ret', 'bench_ret']].round(4))}"
            "</body></html>")
    (out / "report.html").write_text(html, encoding="utf-8")
    print(f"Positions: {summary['positions']}; mean net excess over IWM: {summary['mean_net_excess']}")
    print(f"Yearly test: {positive} of {len(done)} completed years beat IWM (needs {need})")
    if "share_of_placebos_beaten" in placebo:
        print(f"Placebo: rising areas beat {placebo['share_of_placebos_beaten']:.0%} of random-area portfolios")
    print(f"{'PASSED' if summary['passed'] else 'NOT PASSED'}. Report: {out / 'report.html'}")


def cmd_notices(cfg: dict, years: str | None = None) -> None:
    """Filter SAM.gov archives for sole-source and intent notices, then find the public companies they name."""
    import requests

    from .notices import build_name_index, filter_year, scan_notices
    from .sec import load_all_names, load_av_delisted, load_tickers

    data = Path(cfg["data_dir"])
    ndir = data / "notices"
    ndir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["awards"]["start_date"])
    end = pd.Timestamp(cfg["awards"]["end_date"])
    first_fy = start.year + (1 if start.month >= 10 else 0)
    last_fy = end.year + 1                         # notices posted late in a year are often archived the next FY
    session = requests.Session()
    session.headers["User-Agent"] = "award-signal research"
    files = []
    todo = [int(y) for y in years.split(",")] if years else range(first_fy, last_fy + 1)
    for fy in todo:
        out = ndir / f"FY{fy}.csv"
        if not out.exists() or not (ndir / f"FY{fy}_awards.csv").exists():
            print(f"Streaming SAM.gov archive FY{fy} (large file; only matching notices are kept)...")
            for attempt in range(3):
                try:
                    filter_year(fy, out, session)
                    break
                except Exception as exc:
                    print(f"  FY{fy} attempt {attempt + 1} failed: {exc}")
            else:
                print(f"  FY{fy} skipped after three failures; run again later to fill it.")
                continue
        files.append(out)
    if years:                                      # --years: filter those archives only
        return
    http = _http(cfg, "sec")
    tickers = load_tickers(http)
    all_names = load_all_names(http)
    key = str(cfg.get("alphavantage_api_key") or "").strip()
    delisted = load_av_delisted(Http(data / "cache" / "alphavantage", "award-signal"), key) if key else None
    index = build_name_index(tickers, delisted, all_names)
    print(f"Looking for {len(index):,} company names in the notices...")
    awards, rmap = scan_notices(files, index, cfg)
    awards.to_csv(data / "awards_notices.csv", index=False)
    rmap.to_csv(data / "recipient_map_notices.csv", index=False)
    print(f"{len(awards):,} notice mentions of {len(rmap):,} public companies saved to data/awards_notices.csv")


def cmd_demo(cfg: dict, effect: float) -> None:
    from .synthetic import make_market

    m = make_market(effect=effect)
    out = Path(cfg["output_dir"]) / "demo"
    print(f"Synthetic market with a planted effect of {effect:+.0%} over 90 sessions.")
    analyze(m["awards"], m["map"], m["fundamentals"], m["provider"], cfg, out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="award_signal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["awards", "match", "notices", "budget", "widen", "backtest", "run", "demo"])
    ap.add_argument("--rules", choices=["v1", "v2"], default="v1", help="backtest only: stage 2 rule version")
    ap.add_argument("--years", help="notices only: filter just these fiscal years (e.g. 2021,2022), no scan")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--effect", type=float, default=0.15, help="demo only: planted effect size")
    ap.add_argument("--source", choices=["usaspending", "announcements", "csv"],
                    help="awards only: override awards.source in config.yaml")
    ap.add_argument("--out", help="run only: write the report to this folder under output/")
    ap.add_argument("--awards", help="run only: awards file to use in place of data/awards.csv")
    ap.add_argument("--map", help="run only: recipient map to use in place of data/recipient_map.csv")
    ap.add_argument("--no-ratio", action="store_true",
                    help="run only: skip the award-to-revenue test (for notices that state no value)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.source:
        cfg["awards"]["source"] = args.source
    if args.out:
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / args.out)
    if args.no_ratio:
        cfg["signal"]["min_award_to_revenue"] = 0.0
    {"awards": lambda: cmd_awards(cfg), "match": lambda: cmd_match(cfg),
     "notices": lambda: cmd_notices(cfg, args.years), "budget": lambda: cmd_budget(cfg), "widen": lambda: cmd_widen(cfg),
     "backtest": lambda: cmd_backtest(cfg, args.rules),
     "run": lambda: cmd_run(cfg, args.awards, args.map), "demo": lambda: cmd_demo(cfg, args.effect)}[args.command]()

"""Command line: python -m award_signal <command>

    awards   download DoD awards (or load a CSV) into data/awards.csv
    match    match award recipients to SEC-listed companies
    run      build events, run the event study, write output/report.html
    demo     run the whole pipeline on synthetic data (no internet needed)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) award-signal research", min_interval=0.3)
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


def cmd_notices(cfg: dict) -> None:
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
    for fy in range(first_fy, last_fy + 1):
        out = ndir / f"FY{fy}.csv"
        if not out.exists():
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
    ap.add_argument("command", choices=["awards", "match", "notices", "run", "demo"])
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
     "notices": lambda: cmd_notices(cfg),
     "run": lambda: cmd_run(cfg, args.awards, args.map), "demo": lambda: cmd_demo(cfg, args.effect)}[args.command]()

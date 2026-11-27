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
    if cfg["awards"]["source"] == "csv":
        df = awards_mod.load_awards_csv(cfg["awards"]["csv_path"], cfg)
    else:
        print("Downloading DoD awards from USAspending, one month at a time...")
        df = awards_mod.fetch_usaspending(cfg, _http(cfg, "usaspending"))
        skipped = Path(cfg["data_dir"]) / "awards_skipped.csv"
        pd.DataFrame(df.attrs.get("skipped", []), columns=["start", "end", "page", "error"]).to_csv(skipped, index=False)
    df.to_csv(out, index=False)
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


def cmd_run(cfg: dict) -> None:
    from .sec import Fundamentals

    aw = pd.read_csv(Path(cfg["data_dir"]) / "awards.csv", parse_dates=["action_date", "signal_date"])
    rmap = load_map(cfg["data_dir"])
    analyze(aw, rmap, Fundamentals(_http(cfg, "sec")), make_provider(cfg), cfg, cfg["output_dir"])


def cmd_demo(cfg: dict, effect: float) -> None:
    from .synthetic import make_market

    m = make_market(effect=effect)
    out = Path(cfg["output_dir"]) / "demo"
    print(f"Synthetic market with a planted effect of {effect:+.0%} over 90 sessions.")
    analyze(m["awards"], m["map"], m["fundamentals"], m["provider"], cfg, out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="award_signal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["awards", "match", "run", "demo"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--effect", type=float, default=0.15, help="demo only: planted effect size")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"awards": lambda: cmd_awards(cfg), "match": lambda: cmd_match(cfg),
     "run": lambda: cmd_run(cfg), "demo": lambda: cmd_demo(cfg, args.effect)}[args.command]()

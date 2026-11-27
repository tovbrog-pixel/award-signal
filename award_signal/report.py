"""Write CSV outputs and a one-page HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

AUTHOR = "Tov Binyamin Brog"


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:+.1f}%"


def _money(x) -> str:
    if pd.isna(x):
        return ""
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:,.0f}"


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_outputs(out_dir: str | Path, res: pd.DataFrame, delayed: pd.DataFrame,
                  yearly: pd.DataFrame, gate: dict, placebo: dict, cfg: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drop = [c for c in ("entry_index",) if c in res.columns]
    res.drop(columns=drop).to_csv(out / "events.csv", index=False)
    yearly.to_csv(out / "yearly.csv", index=False)

    traded = res[res["status"] == "traded"]
    d_traded = delayed[delayed["status"] == "traded"]
    summary = {
        "author": AUTHOR,
        "signal": f"DoD award >= {cfg['signal']['min_award_to_revenue']:.0%} of latest annual revenue",
        "horizon_trading_days": cfg["signal"]["horizon_days"],
        "benchmark": cfg["prices"]["benchmark"],
        "events_matched": int(len(res)),
        "status_counts": res["status"].value_counts().to_dict(),
        "trades": int(len(traded)),
        "mean_net_excess": float(traded["net_excess"].mean()) if len(traded) else None,
        "hit_rate": float((traded["net_excess"] > 0).mean()) if len(traded) else None,
        "delay_test": {
            "delay_trading_days": cfg["signal"]["delay_test_days"],
            "trades": int(len(d_traded)),
            "mean_net_excess": float(d_traded["net_excess"].mean()) if len(d_traded) else None,
        },
        "gate": gate,
        "placebo": placebo,
        # A 5-of-8-years rule alone passes pure noise about 36% of the time
        # (binomial, p = 0.5), so Phase 1 also requires the placebo test.
        "phase1_passed": bool(gate["passed"] and placebo.get("passes_95pct", False)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "report.html").write_text(_html(summary, yearly, traded), encoding="utf-8")
    return summary


def _html(summary: dict, yearly: pd.DataFrame, traded: pd.DataFrame) -> str:
    g, p, d = summary["gate"], summary["placebo"], summary["delay_test"]
    ok = summary["phase1_passed"]
    verdict = "PASSED" if ok else "NOT PASSED"
    y = yearly.copy()
    for c in ("mean_net_excess", "median_net_excess", "mean_cost"):
        y[c] = y[c].map(_pct)
    y["hit_rate"] = y["hit_rate"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    t = traded.sort_values("signal_date", ascending=False).head(200)
    t = pd.DataFrame({
        "signal date": pd.to_datetime(t["signal_date"]).dt.date,
        "ticker": t["ticker"],
        "award": t["amount"].map(_money),
        "revenue": t["revenue"].map(_money),
        "award/revenue": t["award_to_revenue"].map(lambda v: f"{v:.0%}"),
        "market cap": t["market_cap"].map(_money),
        "return": t["ret"].map(_pct),
        "benchmark": t["bench_ret"].map(_pct),
        "cost": t["cost"].map(_pct),
        "net excess": t["net_excess"].map(_pct),
    })
    placebo_line = (
        f"Real signals beat {p['share_of_placebos_beaten']:.0%} of {p.get('events_with_placebo', 0)}-event "
        f"random-date portfolios (target 95%)." if "share_of_placebos_beaten" in p else "Placebo test not available."
    )
    delay_line = (
        f"With signals {d['delay_trading_days']} trading days late: {d['trades']} trades, "
        f"mean net excess {_pct(d['mean_net_excess'])}." if d["mean_net_excess"] is not None else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="author" content="{AUTHOR}">
<title>Award Signal - Phase 1 results</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border-bottom: 1px solid #d0d7de; padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.verdict {{ font-weight: 700; color: {'#1a7f37' if ok else '#cf222e'}; }}
.byline {{ color: #57606a; }}
</style></head><body>
<h1>Award Signal: Phase 1 results</h1>
<p class="byline">{AUTHOR}</p>
<p>Signal: {html.escape(summary['signal'])}. Hold {summary['horizon_trading_days']} trading days,
compared with {summary['benchmark']}, after estimated spread and market-impact costs.</p>
<p class="verdict">Phase 1 {verdict}. Both tests must pass.</p>
<p>Yearly test: {g['years_beating_benchmark']} of {g['test_years']} test years beat the benchmark after costs
(needs {g['years_required']}).</p>
<p>{summary['trades']} trades; mean net excess {_pct(summary['mean_net_excess'])};
hit rate {'' if summary['hit_rate'] is None else f"{summary['hit_rate']:.0%}"}.</p>
<p>Placebo test: {placebo_line}</p>
<p>{delay_line}</p>
<h2>By year</h2>
{_table(y)}
<h2>Trades (newest 200)</h2>
{_table(t)}
<h2>Events excluded, by reason</h2>
{_table(pd.DataFrame(list(summary['status_counts'].items()), columns=['status', 'events']))}
</body></html>
"""

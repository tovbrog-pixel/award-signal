# Award Signal

Author: Tov Binyamin Brog

Phase 1 of the investment opportunity tool. It tests one question: when the Department of Defense awards a contract worth at least 10% of a small public company's annual revenue, does that stock beat the Russell 2000 (IWM) over the next 90 trading days, after trading costs?

## How a test runs

1. **Awards.** `awards` downloads DoD contracts of $7.5M and up from USAspending, 2015 through 2025. DoD contract data reaches USAspending about 90 days after award, so each award's signal date is its action date plus 90 days. A CSV with its own `signal_date` column (for example, DoD announcement dates or a HigherGov export) keeps those dates and skips the lag.
2. **Matching.** `match` pairs recipient names with SEC-listed companies. Exact name matches go in automatically. Close matches wait in a file for you to confirm.
3. **Fundamentals.** For each signal date, the tool pulls the latest annual revenue and share count that SEC had on file by that date. Restatements filed later never leak backward.
4. **Event study.** Each qualifying stock is bought at the next session's open and sold at the open 90 sessions later. The tool charges half the bid-ask spread plus a market-impact estimate on each side, and both grow as daily dollar volume shrinks.
5. **Verdict.** Phase 1 passes only if both tests pass:
    - **Yearly test:** at least 5 of the 8 test years (2018 to 2025) beat IWM after costs.
    - **Placebo test:** the real signals beat at least 95% of 1,000 portfolios built from random dates on the same stocks.

    The placebo test is required because the yearly test alone passes pure noise about 36% of the time.

A delay test reruns everything with each signal 5 trading days late. A big drop there means the edge depends on speed.

## Run it on your laptop (Windows)

Install Python 3.11 or newer from python.org, then open PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m award_signal demo
```

The demo builds a fake market with a known effect and should report `Phase 1 PASSED`. Run `python -m award_signal demo --effect 0` to confirm it reports `NOT PASSED` when no effect exists.

For the real test:

1. Open `config.yaml` and set `sec_user_agent` to your name and email. SEC requires this.
2. Run the commands below in order. The first award download can take an hour or more; everything is cached in `data\`.

```powershell
python -m award_signal awards
python -m award_signal match
python -m award_signal run
```

3. Open `output\report.html`.

To review close matches, open `data\recipient_map_candidates.csv`, put `Y` in the `confirmed` column for correct pairs, and save it as `data\recipient_map.csv`. Then run `run` again.

## Known limits of this first version

| Limit | Effect | Fix planned |
| --- | --- | --- |
| yfinance prices cover only stocks still trading | Results lean optimistic (survivorship bias) | Switch `prices.provider` to `csv` with Sharadar exports |
| SEC ticker list covers current filers only | Companies acquired or delisted since award go unmatched | Add CIKs by hand in `recipient_map.csv`; Sharadar ticker history later |
| USAspending reports award totals | A large modification can look like a new award | Use DoD announcement dates or FPDS action-level data |
| 90-day reporting lag | The test trades later than an announcement reader would | Load DoD announcement dates through the CSV path |

## Files

| Path | Holds |
| --- | --- |
| `config.yaml` | Every threshold, date range and cost assumption |
| `award_signal/awards.py` | USAspending download and CSV loader |
| `award_signal/sec.py` | Point-in-time revenue and shares from SEC company facts |
| `award_signal/matching.py` | Recipient-to-ticker matching |
| `award_signal/prices.py` | yfinance and CSV price providers |
| `award_signal/study.py` | Event study, costs, yearly test, placebo test |
| `award_signal/report.py` | CSV outputs and the HTML report |
| `award_signal/synthetic.py` | Fake market for tests and the demo |

This is a research tool for testing one hypothesis. Treat its output as research.

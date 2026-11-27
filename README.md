# Award Signal

Author: Tov Binyamin Brog

Phase 1 of the investment opportunity tool. It tests one question: when the Department of Defense awards a contract worth at least 10% of a small public company's annual revenue, does that stock beat the Russell 2000 (IWM) over the next 90 trading days, after trading costs?

## How a test runs

1. **Awards.** `awards` downloads DoD contracts of $7.5M and up from USAspending, 2015 through 2025. It keeps only new awards, dated by their first transaction, so a modification to an old contract never counts as a new award. DoD contract data reaches USAspending about 90 days after award, so each award's signal date is its action date plus 90 days. A CSV with its own `signal_date` column (for example, DoD announcement dates or a HigherGov export) keeps those dates and skips the lag.
2. **Matching.** `match` pairs recipient names with SEC-listed companies in two passes.
    - **Current companies:** exact name matches against SEC's ticker list go in automatically; close matches wait for you to confirm.
    - **Companies gone since:** recipients the first pass missed are checked against SEC's full filer list, which keeps acquired and delisted companies. With a free Alpha Vantage key in `config.yaml`, each one also gets its old ticker and delisting date.
3. **Fundamentals.** For each signal date, the tool pulls the latest annual revenue and share count that SEC had on file by that date. Restatements filed later never leak backward.
4. **Event study.** Each qualifying stock is bought at the next session's open and sold at the open 90 sessions later. The tool charges half the bid-ask spread plus a market-impact estimate on each side, and both grow as daily dollar volume shrinks.
5. **Verdict.** Phase 1 passes only if both tests pass:
    - **Yearly test:** at least 5 of the 8 test years (2018 to 2025) beat IWM after costs.
    - **Placebo test:** the real signals beat at least 95% of 1,000 portfolios built from random dates on the same stocks.

    The placebo test is required because the yearly test alone passes pure noise about 36% of the time.

Two checks run alongside the verdict:

- **Delay test:** reruns everything with each signal 5 trading days late. A big drop means the edge depends on speed.
- **Survivorship check:** counts every qualifying trade the tool could not price (usually a delisted stock) as a total loss and reruns the yearly test. If the signal still passes, missing delisted stocks cannot explain the result. A reused ticker, where prices continue more than 30 days past the old company's delisting date, counts as unpriced.

## Pre-award test: sole-source notices

Agencies post a notice before awarding a contract without competition, and the notice usually names the company. `notices` streams SAM.gov's yearly archived-opportunity files (FY2015 onward), keeps the sole-source and intent-to-award notices, and finds the public companies they name by looking up SEC and delisted company names in the notice text. Each notice is dated by its posting date, which comes before any award.

On Windows, double-click `run_notices.bat`. The SAM.gov files are large and are read as a stream; only matching notices are saved, in `data\notices\`, so a rerun skips years already done. The report goes to `output\notices\report.html`. Most notices state no dollar value, so this run skips the award-to-revenue test and uses the market-cap band, cooldown, placebo and yearly tests as before.

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

1. Open `config.yaml` and set `sec_user_agent` to your name and email. SEC requires this. Optionally, add a free key from alphavantage.co as `alphavantage_api_key`.
2. Run the commands below in order. The first award download can take an hour or more; everything is cached in `data\`.

```powershell
python -m award_signal awards
python -m award_signal match
python -m award_signal run
```

3. Open `output\report.html`.

To review matches, open `data\recipient_map_candidates.csv`. Rows with a blank `confirmed` column carry a `note` explaining why. Put `Y` in `confirmed` for correct pairs, type in any old ticker you know, and save the file as `data\recipient_map.csv`. Then run `run` again.

## Known limits of this first version

| Limit | Effect | Fix planned |
| --- | --- | --- |
| yfinance prices cover only stocks still trading | Delisted stocks go unpriced | The survivorship check bounds the damage; delisted prices from Tiingo or EODHD through the `csv` provider remove it |
| Delisted tickers need an Alpha Vantage key | Without one, delisted companies match SEC names but get no ticker | Add the free key, or type old tickers into `recipient_map.csv` |
| USAspending's award amount is the contract's current total | Later modifications can inflate the size the test uses | Use announced values from DoD announcement pages through the CSV path |
| 90-day reporting lag | The test trades later than an announcement reader would | Load DoD announcement dates through the CSV path |

## Files

| Path | Holds |
| --- | --- |
| `config.yaml` | Every threshold, date range and cost assumption |
| `award_signal/awards.py` | USAspending download and CSV loader |
| `award_signal/sec.py` | Point-in-time revenue and shares from SEC company facts |
| `award_signal/matching.py` | Recipient-to-ticker matching |
| `award_signal/prices.py` | yfinance and CSV price providers |
| `award_signal/study.py` | Event study, costs, yearly, placebo and survivorship tests |
| `award_signal/report.py` | CSV outputs and the HTML report |
| `award_signal/synthetic.py` | Fake market for tests and the demo |

This is a research tool for testing one hypothesis. Treat its output as research.

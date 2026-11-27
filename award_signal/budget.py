"""Stage 1 inputs: DoD budget requests (R-1 research lines, P-1 procurement lines).

Each year's request lists, for every line, the prior-year actual, the
current-year enacted amount and the requested amount, in thousands of dollars.
The Comptroller publishes these as Excel files for FY1998 onward; this module
finds each year's files on the Comptroller page, reads them with a header
search that tolerates the format changes between years, and returns one long
table:

    request_fy, exhibit (r1|p1), line_id, title, organization, account_title,
    fy, kind (actual|enacted|request|total), amount (dollars)

Point in time: a request is public from its release date. Requests come out
between February and June, so the detector dates every signal 1 July of the
calendar year before the fiscal year starts (config: budget.signal_month_day).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

PAGE_URL = "https://comptroller.war.gov/Budget-Materials/Budget{fy}/"
SITE = "https://comptroller.war.gov"

HREF_RE = re.compile(r'href="([^"]+?/(r1|p1)(_display)?\.xlsx?)"', re.I)


def find_files(page_html: str) -> dict[str, str]:
    """Pick one file per exhibit: the full file when offered, else the display file."""
    found: dict[str, dict[bool, str]] = {}
    for href, exhibit, display in HREF_RE.findall(page_html):
        found.setdefault(exhibit.lower(), {})[bool(display)] = href
    out = {}
    for ex, d in found.items():
        href = d.get(False) or d.get(True)
        out[ex] = href if href.startswith("http") else SITE + href
    return out


def known_files(fy: int) -> dict[str, str]:
    """File locations as the Comptroller site listed them in November 2026: lower-case
    folder names through FY2021, upper-case after; FY2026 on offer only display files."""
    folder = f"fy{fy}" if fy <= 2021 else f"FY{fy}"
    name = "{ex}_display.xlsx" if fy >= 2026 else "{ex}.xlsx"
    return {ex: f"{SITE}/Portals/45/Documents/defbudget/{folder}/" + name.format(ex=ex) for ex in ("r1", "p1")}


def alternates(url: str) -> list[str]:
    """Fallbacks: the display file, and the other folder-name case."""
    out = []
    if "_display" not in url:
        out.append(url.replace(".xlsx", "_display.xlsx"))
    m = re.search(r"/(fy|FY)(\d{4})/", url)
    if m:
        other = ("FY" if m.group(1) == "fy" else "fy") + m.group(2)
        out += [u.replace(m.group(0), f"/{other}/") for u in [url] + out]
    return out


def _find_header(raw: pd.DataFrame) -> int | None:
    for i in range(min(40, len(raw))):
        cells = [str(c).lower() for c in raw.iloc[i].tolist()]
        joined = " | ".join(cells)
        if ("program element" in joined or "line item" in joined or "budget line" in joined
                or re.search(r"\bpe\b|\bbli\b", joined)) and "fy" in joined:
            return i
    return None


def _labels(raw: pd.DataFrame, h: int) -> list[str]:
    """Header labels, merging up to two rows below when they carry text (multi-row headers)."""
    labels = [str(c) if pd.notna(c) else "" for c in raw.iloc[h].tolist()]
    for extra in (1, 2):
        if h + extra >= len(raw):
            break
        row = raw.iloc[h + extra].tolist()
        texty = [isinstance(c, str) and not re.fullmatch(r"[\d,.\-\s]*", c) for c in row]
        if sum(texty) >= 2 and not any(isinstance(c, (int, float)) and pd.notna(c) and c > 1000 for c in row):
            labels = [(a + " " + (str(b) if isinstance(b, str) else "")).strip() for a, b in zip(labels, row)]
        else:
            break
    # carry a year heading (for example "FY 2020 Enacted") across merged cells to the right
    parts = r"\b(base|oco|total|emergency|dwe|ooc|discretionary|mandatory)\b"
    prefix = ""
    out = []
    for lab in labels:
        if re.search(r"fy\s*\d{4}", lab, re.I):
            prefix = re.sub(parts, "", lab, flags=re.I).strip()
        elif prefix and re.search(parts, lab, re.I):
            lab = f"{prefix} {lab}"
        out.append(re.sub(r"\s+", " ", lab).strip())
    return out


def _col(labels: list[str], *patterns: str, exclude: str = "title") -> int | None:
    for p in patterns:
        for i, lab in enumerate(labels):
            if re.search(p, lab, re.I) and not (exclude and re.search(exclude, lab, re.I)):
                return i
    return None


SKIP = re.compile(r"quantity|qty|less|remaining|classification|include", re.I)


def _usable_amount(lab: str, fy: int) -> bool:
    if not re.search(rf"fy\s*{fy}\b", lab, re.I) or SKIP.search(lab):
        return False
    # an OCO-only column is part of a total, never the total itself
    if re.search(r"\boco\b", lab, re.I) and not re.search(r"base", lab, re.I):
        return False
    return True


def _pick(labels: list[str], fy: int, tests: list) -> int | None:
    cands = [(i, lab) for i, lab in enumerate(labels) if _usable_amount(lab, fy)]
    for test in tests:
        for i, lab in cands:
            if test(lab.lower()):
                return i
    return cands[0][0] if cands else None


def amount_columns(labels: list[str], request_fy: int) -> dict[str, int | None]:
    """Choose the comparable 'current-year enacted' and 'request' columns by priority.

    Request: total, then request, then base. Enacted (prior year): total enacted,
    then total, then enacted/enactment, then a continuing-resolution request.
    Totals are preferred on both sides so the two figures cover the same funds.
    """
    req = _pick(labels, request_fy, [lambda l: "total" in l, lambda l: "request" in l, lambda l: "base" in l])
    enacted = _pick(labels, request_fy - 1, [lambda l: "total" in l and "enact" in l, lambda l: "total" in l,
                                             lambda l: "enact" in l, lambda l: "request" in l or "cr" in l])
    return {"request": req, "enacted": enacted}


def parse_exhibit(content: bytes, exhibit: str, request_fy: int) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), header=None, sheet_name=0)
    h = _find_header(raw)
    if h is None:
        raise ValueError(f"{exhibit} FY{request_fy}: no header row found")
    labels = _labels(raw, h)
    # the line identifier: PE or budget line item number, never the line-number column (it repeats)
    id_col = _col(labels, r"^pe\s*/\s*bli$", r"^(budget )?line item$", r"^pe$", r"^program element$",
                  r"program element.*(number|#|no)", r"line item.*(number|#|no)", r"\bbli\b", r"\bpe\b",
                  r"line number")
    title_col = _col(labels, r"(program element|line item|bli|pe).*title", r"^title$", exclude="")
    org_col = _col(labels, r"^organization", exclude="")
    acct_col = _col(labels, r"account title", exclude="")
    addnon_col = _col(labels, r"add\s*/\s*non", exclude="")
    cols = amount_columns(labels, request_fy)
    if id_col is None or title_col is None or (cols["request"] is None and cols["enacted"] is None):
        raise ValueError(f"{exhibit} FY{request_fy}: columns not recognized: {labels}")
    rows = []
    for _, r in raw.iloc[h + 1:].iterrows():
        line_id, title = r.iloc[id_col], r.iloc[title_col]
        if pd.isna(line_id) or pd.isna(title) or str(title).strip().lower().startswith("total"):
            continue
        if addnon_col is not None and "non" in str(r.iloc[addnon_col]).lower():
            continue   # non-add rows repeat money already counted elsewhere
        base = {"request_fy": request_fy, "exhibit": exhibit, "line_id": str(line_id).strip(),
                "title": str(title).strip(),
                "organization": str(r.iloc[org_col]).strip() if org_col is not None else "",
                "account_title": str(r.iloc[acct_col]).strip() if acct_col is not None else ""}
        for kind, fy in (("enacted", request_fy - 1), ("request", request_fy)):
            i = cols[kind]
            if i is None:
                continue
            v = pd.to_numeric(r.iloc[i], errors="coerce")
            if pd.notna(v):
                rows.append({**base, "fy": fy, "kind": kind, "amount": float(v) * 1000,
                             "source_column": labels[i]})
    return pd.DataFrame(rows)


def line_summary(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (request_fy, exhibit, line): enacted current-year and requested amounts."""
    rows = []
    for (rfy, ex, lid), g in long.groupby(["request_fy", "exhibit", "line_id"]):
        def pick(fy, kinds):
            # first kind present wins, so an enacted column and a total column are never added together
            for k in kinds:
                s = g[(g["fy"] == fy) & (g["kind"] == k)]["amount"]
                if len(s):
                    return float(s.sum())
            return None
        rows.append({"request_fy": rfy, "exhibit": ex, "line_id": lid,
                     "title": g["title"].iloc[0], "organization": g["organization"].iloc[0],
                     "enacted": pick(rfy - 1, ["enacted"]),
                     "request": pick(rfy, ["request"])})
    return pd.DataFrame(rows)


def tag_areas(titles: pd.Series, areas: dict[str, list[str]]) -> pd.Series:
    pats = {a: re.compile("|".join(r"\b" + re.escape(k) for k in kws), re.I) for a, kws in areas.items()}
    return titles.map(lambda t: [a for a, p in pats.items() if p.search(str(t))])


def flag_lines(summary: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Line flags: budget ramp and new start, each judged within one request year."""
    b = cfg["budget"]
    s = summary.copy()
    s["ramp"] = (s["request"].fillna(0) >= b["ramp_min_request"]) & (
        s["request"].fillna(0) >= b["ramp_multiple"] * s["enacted"].fillna(0).clip(lower=1))
    prior = {(r.request_fy, r.exhibit, r.line_id) for r in s.itertuples()}
    first_year = s["request_fy"].min()
    s["new_start"] = [
        r.request_fy > first_year and (r.request_fy - 1, r.exhibit, r.line_id) not in prior
        and (r.request or 0) >= b["new_start_min_request"] and not (r.enacted or 0) > 0
        for r in s.itertuples()]
    return s


def area_signals(flagged: pd.DataFrame, areas: dict[str, list[str]], cfg: dict) -> pd.DataFrame:
    """Area totals per request year and the rising-area flag.

    Rule (version 1, frozen 2026-11-27 before any return was computed): an area
    is rising when its requested total grows at least `area_excess_growth`
    faster than the whole R-1 plus P-1 request does that year, and by at least
    `area_min_increase` dollars. Measuring against the department-wide change
    keeps a year of across-the-board growth from flagging every area.
    """
    b = cfg["budget"]
    md = b.get("signal_month_day", "07-01")
    totals = flagged.groupby("request_fy").agg(E=("enacted", "sum"), R=("request", "sum"))
    dod_growth = (totals["R"] / totals["E"] - 1).to_dict()
    f = flagged.copy()
    f["areas"] = tag_areas(f["title"], areas)
    f = f.explode("areas").dropna(subset=["areas"])
    rows = []
    for (rfy, area), g in f.groupby(["request_fy", "areas"]):
        enacted = g["enacted"].fillna(0).sum()
        request = g["request"].fillna(0).sum()
        growth = (request / enacted - 1) if enacted > 0 else None
        excess = growth - dod_growth.get(rfy, 0.0) if growth is not None else None
        rising = bool(excess is not None and excess >= b["area_excess_growth"]
                      and request - enacted >= b["area_min_increase"])
        rows.append({"request_fy": rfy, "area": area, "lines": len(g), "enacted": enacted, "request": request,
                     "growth": growth, "dod_growth": dod_growth.get(rfy), "excess_growth": excess,
                     "ramps": int(g["ramp"].sum()), "new_starts": int(g["new_start"].sum()),
                     "rising": rising, "signal_date": pd.Timestamp(f"{rfy - 1}-{md}")})
    return pd.DataFrame(rows).sort_values(["request_fy", "area"])


def local_files(folder: Path, fy: int) -> dict[str, Path]:
    """Files saved by hand: data/budget/FY2020/r1.xlsx (or r1_display.xlsx), p1 likewise."""
    d = folder / f"FY{fy}"
    out = {}
    for ex in ("r1", "p1"):
        for name in (f"{ex}.xlsx", f"{ex}_display.xlsx", f"{ex}.xls"):
            if (d / name).exists():
                out[ex] = d / name
                break
    return out


def fetch_all(cfg: dict, http, log=print) -> pd.DataFrame:
    """Read R-1 and P-1 for every request year in budget.first_fy..last_fy.

    Files already saved under data/budget/FYxxxx/ are used first. Otherwise the
    tool tries the Comptroller site once; if the site refuses the request, it
    lists the files to download by hand in a browser instead of retrying.
    """
    b = cfg["budget"]
    folder = Path(cfg["data_dir"]) / "budget"
    folder.mkdir(parents=True, exist_ok=True)
    frames, missing = [], []
    site_refused = False
    for fy in range(int(b["first_fy"]), int(b["last_fy"]) + 1):
        files = local_files(folder, fy)
        if len(files) < 2 and not site_refused:
            for ex, url in known_files(fy).items():
                if ex in files:
                    continue
                try:
                    r = http.session.get(url, timeout=120)
                except Exception:
                    r = None
                if r is not None and r.status_code == 200 and r.content[:2] == b"PK":   # .xlsx is a zip
                    (folder / f"FY{fy}").mkdir(exist_ok=True)
                    dest = folder / f"FY{fy}" / Path(url).name
                    dest.write_bytes(r.content)
                    files[ex] = dest
                elif r is not None and r.status_code == 403:
                    site_refused = True
                    break
        for ex in ("r1", "p1"):
            if ex not in files:
                missing.append((fy, ex))
        for ex, path in sorted(files.items()):
            try:
                df = parse_exhibit(path.read_bytes(), ex, fy)
                frames.append(df)
                log(f"  FY{fy} {ex.upper()}: {df['line_id'].nunique():,} lines from {path.name}")
            except Exception as exc:
                log(f"  FY{fy} {ex.upper()}: could not parse ({exc})")
    if missing:
        log("")
        log("These files are missing. Download each one in your browser from the Comptroller's")
        log("Budget Materials page and save it in the folder shown:")
        for fy, ex in missing:
            log(f"  {known_files(fy)[ex]}  ->  data\\budget\\FY{fy}\\")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

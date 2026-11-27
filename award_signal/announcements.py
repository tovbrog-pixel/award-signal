"""DoD daily contract announcements (war.gov, formerly defense.gov).

DoD posts every contract action of $7.5M or more at 5 p.m. Eastern each
business day. Using the announcement date as the signal date removes the
90-day reporting lag in USAspending, and the announcement text separates new
awards from modifications and single awards from multiple-award contracts.

Output columns match awards.COLUMNS plus: face_value, obligated, modification,
multi_award, small_business, contract_number.
"""

from __future__ import annotations

import html
import re

import pandas as pd

from .net import Http, HttpError

BASE = "https://www.war.gov"
LIST_URL = BASE + "/News/Contracts/?Page={page}"
ARTICLE_URL = BASE + "/News/Contracts/Contract/Article/{id}/{slug}/"

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12}

ARTICLE_RE = re.compile(r"/News/Contracts/Contract/Article/(\d+)/(contracts-for-[a-z]+-\d{1,2}-\d{4})")
AWARD_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has been|have been|is being|are being|has|have)\s+(?:each\s+|jointly\s+)?"
    r"(?:been\s+)?awarded\b", re.I)
MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
OBLIGATED_RE = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s+(?:is|are|was|were|will be|has been|have been)\s+(?:being\s+)?obligated", re.I)
CONTRACT_RE = re.compile(r"\b([A-Z0-9]{6}-?\d{2}-?[A-Z]-?[A-Z0-9]{4})\b")


def slug_date(slug: str) -> pd.Timestamp | None:
    m = re.match(r"contracts-for-([a-z]+)-(\d{1,2})-(\d{4})$", slug)
    if not m or m.group(1)[:3] not in MONTHS:
        return None
    return pd.Timestamp(int(m.group(3)), MONTHS[m.group(1)[:3]], int(m.group(2)))


def parse_listing(text: str) -> list[tuple[str, str, pd.Timestamp]]:
    seen, out = set(), []
    for art_id, slug in ARTICLE_RE.findall(text):
        d = slug_date(slug)
        if d is not None and art_id not in seen:
            seen.add(art_id)
            out.append((art_id, slug, d))
    return out


def _clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def article_paragraphs(page: str) -> list[tuple[bool, str]]:
    """(is_heading, text) for each paragraph in the article body."""
    start = page.find('class="body"')
    if start < 0:
        return []
    end = page.find("</div>", start)
    body = page[start:end if end > 0 else None]
    out = []
    for attrs, inner in re.findall(r"<p([^>]*)>(.*?)</p>", body, flags=re.S | re.I):
        text = _clean(inner)
        if not text:
            continue
        heading = ("center" in attrs.lower() or (text.isupper() and len(text) < 80)) and "$" not in text
        out.append((heading, text))
    return out


def _money(s: str) -> float:
    return float(s.replace(",", ""))


def _company(segment: str) -> str:
    name = segment.split(",")[0]
    name = re.sub(r"^\s*(?:and|the following companies:?)\s+", "", name, flags=re.I)
    return name.replace("*", "").strip(" .;:")


def parse_paragraph(text: str) -> list[dict] | None:
    verb = AWARD_VERB_RE.search(text)
    if not verb or verb.start() == 0:
        return None
    head, tail = text[:verb.start()], text[verb.end():]
    money = MONEY_RE.search(tail)
    if not money:
        return None
    segments = [s for s in head.split(";") if s.strip()]
    multi = len(segments) > 1 or "each awarded" in text[verb.start():verb.end() + 6].lower()
    names = [_company(s) for s in segments]
    names = [n for n in names if n and len(n) > 1]
    if not names:
        return None
    near = tail[: money.start() + 200].lower()
    obligated = OBLIGATED_RE.search(text)
    contract = CONTRACT_RE.search(text)
    return [{
        "recipient_name": n,
        "face_value": _money(money.group(1)),
        "obligated": _money(obligated.group(1)) if obligated else None,
        "modification": "modification" in near,
        "multi_award": multi,
        "small_business": "*" in seg[: seg.find(",") + 3] if "," in seg else "*" in seg,
        "contract_number": contract.group(1) if contract else None,
        "description": text,  # full paragraph (v1 kept the first 300 characters)
    } for n, seg in zip(names, segments if len(segments) == len(names) else [head] * len(names))]


def parse_article(page: str, date: pd.Timestamp, art_id: str) -> list[dict]:
    rows, agency = [], None
    for heading, text in article_paragraphs(page):
        if heading:
            if text.upper() not in ("CONTRACTS", "CONTRACT"):
                agency = text.title()
            continue
        parsed = parse_paragraph(text)
        for r in parsed or []:
            r.update(action_date=date, signal_date=date, sub_agency=agency,
                     award_id=f"{art_id}-{r['contract_number'] or len(rows)}")
            rows.append(r)
    return rows


def cached_listings(http: Http) -> list[tuple[str, str, pd.Timestamp]]:
    """Every article named in listing pages already saved in the cache."""
    seen, out = set(), []
    for f in http.cache_dir.glob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if "/News/Contracts/Contract/Article/" not in text or 'class="body"' in text:
            continue
        for item in parse_listing(text):
            if item[0] not in seen:
                seen.add(item[0])
                out.append(item)
    return out


def fetch_announcements(cfg: dict, http: Http, log=print) -> pd.DataFrame:
    a = cfg["awards"]
    start, end = pd.Timestamp(a["start_date"]), pd.Timestamp(a["end_date"])
    articles, page = [], 1
    while True:
        # listing pages shift by one entry each business day, so cache them per day
        try:
            text = http.text(LIST_URL.format(page=page), cache_tag="#" + pd.Timestamp.today().strftime("%Y-%m-%d"))
        except HttpError as exc:
            if exc.status != 403:
                raise
            # the site refuses the listing today: use the listing pages saved on earlier runs
            articles = [x for x in cached_listings(http) if start <= x[2] <= end]
            log(f"  war.gov refused the listing page ({exc.status}); using {len(articles):,} articles "
                "from listing pages saved earlier")
            break
        listing = parse_listing(text)
        if not listing:
            break
        articles.extend(x for x in listing if start <= x[2] <= end)
        if min(x[2] for x in listing) < start:
            break
        if page % 25 == 0:
            log(f"  listing page {page}: back to {min(x[2] for x in listing).date()}")
        page += 1
    log(f"{len(articles):,} daily announcements between {start.date()} and {end.date()}")
    from concurrent.futures import ThreadPoolExecutor

    def get(item):
        art_id, slug, d = item
        try:
            return item, http.text(ARTICLE_URL.format(id=art_id, slug=slug))
        except Exception as exc:  # keep going; a rerun retries uncached pages
            log(f"  could not fetch {slug}: {exc}")
            return item, None

    rows = []
    ordered = sorted(articles, key=lambda x: x[2])
    with ThreadPoolExecutor(max_workers=int(cfg.get("announcements", {}).get("workers", 4))) as pool:
        for k, ((art_id, slug, d), page_text) in enumerate(pool.map(get, ordered)):
            if page_text:
                rows.extend(parse_article(page_text, d, art_id))
            if (k + 1) % 100 == 0:
                log(f"  {d.date()}: {k + 1:,} announcements read, {len(rows):,} award paragraphs")
    return pd.DataFrame(rows)


def to_awards(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the announcement options and return the standard award columns."""
    o = cfg.get("announcements", {})
    if df.empty:
        return df
    if not o.get("include_modifications", False):
        df = df[~df["modification"]]
    if not o.get("include_multi_award", False):
        df = df[~df["multi_award"]]
    if o.get("amount", "face") == "obligated":
        df = df.assign(amount=df["obligated"])
    else:
        df = df.assign(amount=df["face_value"])
    return df

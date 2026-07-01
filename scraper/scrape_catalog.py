"""Scrape the SHL product catalog -> data/catalog.json.

Scope: Individual Test Solutions only (type=1). Pre-packaged Job Solutions
(type=2) are explicitly out of scope per the assignment.

    python -m scraper.scrape_catalog --out data/catalog.json --enrich

IMPORTANT / defend-this note: SHL's markup can change and the listing is
paginated with ?type=1&start=N (N steps by 12). The row/selector logic below is
written defensively (it looks for links into /view/ and infers the yes-dots and
type keys structurally), but you should open the live page in your browser and
confirm the two selectors flagged with `# VERIFY` still match before trusting a
full run. Scrape politely: a small delay between requests.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; shl-take-home-scraper/1.0)"}

TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
_KEY_RE = re.compile(r"^[ABCDEKPS]$")


def _get(session: requests.Session, url: str, tries: int = 3) -> Optional[str]:
    for attempt in range(tries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _slug_id(url: str) -> str:
    m = re.search(r"/view/([^/]+)/?", url)
    return m.group(1) if m else re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")


def parse_listing_page(html: str) -> List[Dict]:
    """Extract one page of rows. A 'row' is any table row containing a link
    into the catalog detail path /view/."""
    soup = BeautifulSoup(html, "lxml")
    rows: List[Dict] = []

    for tr in soup.select("tr"):
        link = tr.find("a", href=re.compile(r"/view/"))
        if not link:
            continue
        name = link.get_text(strip=True)
        href = urljoin(BASE, link["href"])
        if not name:
            continue

        cells = tr.find_all("td")
        row_html = tr.decode()

        
        yes_dots = tr.select('[class*="-yes"]')
        remote = len(yes_dots) >= 1
        adaptive = len(yes_dots) >= 2

    
        type_keys: List[str] = []
        if cells:
            last_txt = cells[-1].get_text(" ", strip=True)
            for tok in re.split(r"[\s,]+", last_txt):
                if _KEY_RE.match(tok):
                    type_keys.append(tok)
        
        if not type_keys:
            for sp in tr.find_all(["span", "p"]):
                t = sp.get_text(strip=True)
                if _KEY_RE.match(t):
                    type_keys.append(t)

        rows.append(
            {
                "id": _slug_id(href),
                "name": name,
                "url": href,
                "test_type": ", ".join(dict.fromkeys(type_keys)),
                "remote": remote,
                "adaptive": adaptive,
                "description": "",
                "job_levels": "",
                "duration": "",
            }
        )
    return rows


def parse_detail_page(html: str) -> Dict:
    """Pull description / job levels / duration from a product detail page."""
    soup = BeautifulSoup(html, "lxml")
    out = {"description": "", "job_levels": "", "duration": ""}

    def _after_heading(keyword: str) -> str:
        for tag in soup.find_all(["h2", "h3", "h4", "strong", "p"]):
            if keyword.lower() in tag.get_text(strip=True).lower():
                nxt = tag.find_next(["p", "div", "span"])
                if nxt:
                    return nxt.get_text(" ", strip=True)
        return ""

    out["description"] = _after_heading("Description") or ""
    out["job_levels"] = _after_heading("Job levels") or _after_heading("Job level")
    dur = _after_heading("Assessment length") or _after_heading("Completion Time")
    out["duration"] = dur
    # Fallback description: first substantial paragraph.
    if not out["description"]:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) > 60:
                out["description"] = t
                break
    return out


def scrape(type_id: int = 1, enrich: bool = False, page_step: int = 12,
           max_pages: int = 60, delay: float = 1.0) -> List[Dict]:
    session = requests.Session()
    seen: Dict[str, Dict] = {}
    start = 0
    empty_streak = 0

    for _ in range(max_pages):
        url = f"{CATALOG}?type={type_id}&start={start}"
        html = _get(session, url)
        if not html:
            break
        page_rows = parse_listing_page(html)
        new = 0
        for row in page_rows:
            if row["id"] not in seen:
                seen[row["id"]] = row
                new += 1
        print(f"start={start}: {len(page_rows)} rows ({new} new), total={len(seen)}")
        empty_streak = empty_streak + 1 if new == 0 else 0
        if empty_streak >= 2:  # two consecutive pages add nothing -> done
            break
        start += page_step
        time.sleep(delay)

    items = list(seen.values())

    if enrich:
        for i, item in enumerate(items):
            html = _get(session, item["url"])
            if html:
                item.update({k: v for k, v in parse_detail_page(html).items() if v})
            if i % 10 == 0:
                print(f"enriched {i}/{len(items)}")
            time.sleep(delay)

    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/catalog.json")
    ap.add_argument("--type", type=int, default=1, help="1=Individual Test Solutions")
    ap.add_argument("--enrich", action="store_true", help="fetch each detail page")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    items = scrape(type_id=args.type, enrich=args.enrich, delay=args.delay)
    if not items:
        raise SystemExit(
            "No items scraped. Open the catalog in a browser and re-check the "
            "two selectors marked `# VERIFY` in parse_listing_page()."
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} assessments -> {args.out}")


if __name__ == "__main__":
    main()

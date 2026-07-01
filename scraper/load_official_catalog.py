
from __future__ import annotations

import argparse
import json
from typing import List

KEY_MAP = {
    "ability & aptitude": "A",
    "biodata & situational judgment": "B",
    "biodata & situational judgement": "B",  # both spellings seen in the wild
    "competencies": "C",
    "development & 360": "D",
    "assessment exercises": "E",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "personality & behaviour": "P",
    "simulations": "S",
}


def _codes(keys: List[str]) -> str:
    out = []
    for k in keys or []:
        code = KEY_MAP.get(k.strip().lower())
        if code and code not in out:
            out.append(code)
    return ", ".join(out)


def _yn(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "1")


def convert(records: List[dict]) -> List[dict]:
    items, skipped = [], 0
    for r in records:
        
        if r.get("status") and r["status"] != "ok":
            skipped += 1
            continue
        link = r.get("link") or ""
        name = r.get("name") or ""
        if not link or not name:
            skipped += 1
            continue
        items.append(
            {
                "id": str(r.get("entity_id") or link.rstrip("/").split("/")[-1]),
                "name": name.strip(),
                "url": link,
                "test_type": _codes(r.get("keys", [])),
                "description": (r.get("description") or "").strip(),
                "job_levels": ", ".join(r.get("job_levels") or [])
                or (r.get("job_levels_raw") or "").rstrip(", "),
                "remote": _yn(r.get("remote")),
                "adaptive": _yn(r.get("adaptive")),
                "duration": (r.get("duration") or r.get("duration_raw") or "").strip(),
            }
        )
    if skipped:
        print(f"Skipped {skipped} records (non-ok status or missing name/url).")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="infile", help="local path to the catalogue JSON")
    src.add_argument("--url", help="URL to fetch the catalogue JSON from")
    ap.add_argument("--out", default="data/catalog.json")
    args = ap.parse_args()

    if args.url:
        import requests

        resp = requests.get(args.url, timeout=60)
        resp.raise_for_status()
        records = json.loads(resp.text,strict=False)
    else:
        with open(args.infile, encoding="utf-8") as f:
            records = json.loads(f,strict=False)

    if not isinstance(records, list):
        raise SystemExit("Expected a JSON array of assessment objects.")

    items = convert(records)
    if not items:
        raise SystemExit("No usable records after conversion.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # Quick distribution sanity check.
    from collections import Counter

    dist = Counter(
        t.strip() for it in items for t in (it["test_type"].split(",") if it["test_type"] else ["<none>"])
    )
    print(f"Wrote {len(items)} assessments -> {args.out}")
    print("Test-type distribution:", dict(dist))


if __name__ == "__main__":
    main()
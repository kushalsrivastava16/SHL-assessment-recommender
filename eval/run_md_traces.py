
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

from app.agent import handle
from app.schemas import Message

_TURN_CAP = 8


def _slug(url: str) -> str:
    m = re.search(r"/view/([^/>\s]+)", url)
    return m.group(1).lower() if m else url.rstrip("/").split("/")[-1].lower()


def _urls_in_table(block: str) -> List[str]:
    """Pull every catalog URL out of a markdown table block."""
    return re.findall(r"https?://[^\s|<>]+/view/[^\s|<>]+", block)


def parse_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    # User turns: blockquote lines that follow a **User** header.
    user_turns: List[str] = []
    for seg in re.split(r"\*\*User\*\*", text)[1:]:
        # take blockquote lines up to the next bold header / horizontal rule
        stop = re.search(r"\*\*Agent\*\*|\n#|\n---", seg)
        chunk = seg[: stop.start()] if stop else seg
        quoted = " ".join(
            line.strip().lstrip(">").strip()
            for line in chunk.splitlines()
            if line.strip().startswith(">")
        ).strip()
        if quoted:
            user_turns.append(quoted)

    # Gold = URLs from the LAST table in the file (the committed shortlist).
    tables = re.findall(r"(?:^\|.*\n)+", text, re.MULTILINE)
    gold_urls: List[str] = []
    for tbl in tables:
        urls = _urls_in_table(tbl)
        if urls:
            gold_urls = urls  # keep overwriting -> ends on the last table
    gold_slugs = list(dict.fromkeys(_slug(u) for u in gold_urls))

    return {"name": path.stem, "user_turns": user_turns, "gold": gold_slugs}


def recall_at_10(rec_slugs: List[str], gold_slugs: List[str]) -> float:
    if not gold_slugs:
        return float("nan")
    top = set(rec_slugs[:10])
    hit = sum(1 for g in gold_slugs if g in top)
    return hit / len(gold_slugs)


def replay(trace: dict, max_turns: int = _TURN_CAP) -> dict:
    messages: List[Message] = []
    ui = 0
    last_recs: List[str] = []

    while len(messages) < max_turns:
        # Feed the next scripted user turn, or "no preference" if we've run out
        # (mirrors the harness: user says no preference outside its facts).
        if ui < len(trace["user_turns"]):
            messages.append(Message(role="user", content=trace["user_turns"][ui]))
            ui += 1
        else:
            messages.append(Message(role="user", content="No preference."))

        resp = handle(messages)
        messages.append(Message(role="assistant", content=resp.reply))

        if resp.recommendations:
            last_recs = [_slug(r.url) for r in resp.recommendations]

        # Stop once we've consumed all user turns AND have a shortlist.
        if ui >= len(trace["user_turns"]) and last_recs:
            break
        if resp.end_of_conversation:
            break

    return {
        "recall@10": recall_at_10(last_recs, trace["gold"]),
        "n_recs": len(last_recs),
        "n_gold": len(trace["gold"]),
        "turns": len(messages),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--max-turns", type=int, default=_TURN_CAP)
    args = ap.parse_args()

    files = sorted(Path(args.traces).glob("*.md"))
    if not files:
        raise SystemExit(f"No .md traces found in {args.traces}")

    recalls = []
    for fp in files:
        trace = parse_md(fp)
        r = replay(trace, args.max_turns)
        recalls.append(r["recall@10"])
        print(f"{trace['name']}: recall@10={r['recall@10']:.3f} "
              f"(matched of {r['n_gold']} gold; agent returned {r['n_recs']}; "
              f"{r['turns']} msgs)")

    valid = [x for x in recalls if x == x]
    if valid:
        print(f"\nMean Recall@10 over {len(valid)} traces: {sum(valid)/len(valid):.3f}")


if __name__ == "__main__":
    main()
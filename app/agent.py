
from __future__ import annotations

import json
import re
from typing import List, Optional

from app import prompts
from app.catalog import Assessment, load_catalog
from app.config import settings
from app.llm import complete_json
from app.retrieval import get_retriever
from app.schemas import ChatResponse, Message, Recommendation

_INJECTION = re.compile(
    r"(ignore (all |the )?(previous|prior|above) (instructions|prompts))"
    r"|(disregard (your|the) (instructions|system))"
    r"|(reveal|show|print).{0,20}(system prompt|your prompt|instructions)"
    r"|(you are now )|(\bjailbreak\b)|(developer mode)",
    re.IGNORECASE,
)

_REFUSAL = (
    "I can only help you choose SHL assessments. Tell me about the role you're "
    "hiring for — responsibilities, level, and the skills or behaviors that "
    "matter — and I'll suggest assessments from the SHL catalog."
)

_TURN_CAP = 8 


def _history_text(messages: List[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _last_user(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _assistant_turns(messages: List[Message]) -> int:
    return sum(1 for m in messages if m.role == "assistant")


def _to_recs(items: List[Assessment]) -> List[Recommendation]:
    return [
        Recommendation(name=a.name, url=a.url, test_type=a.test_type or "")
        for a in items[: settings.max_recommendations]
    ]


def _fallback_query(messages: List[Message]) -> str:
    """If the controller didn't give us a search query, stitch the user's own
    words together."""
    return " ".join(m.content for m in messages if m.role == "user").strip()


def _facet_queries(decision: dict, messages: List[Message]) -> List[str]:
    """Decompose the need into one query per facet.

    A hiring need is usually multi-part ("graduate analyst: numerical + finance
    + situational judgement + personality"). A single blended query embeds the
    average and buries individual facets, so gold items for the weaker facets
    fall out of the top 10. Retrieving per facet and interleaving guarantees
    every facet contributes candidates. Each facet string is also literal-term
    rich, which helps the BM25 half match named report/variant products.
    """
    facets: List[str] = []
    sq = decision.get("search_query")
    if sq:
        facets.append(str(sq))

    req = decision.get("requirements", {})
    if isinstance(req, dict):
        if req.get("role"):
            facets.append(str(req["role"]))
        for key in ("skills", "competencies", "test_types_wanted"):
            v = req.get(key)
            if isinstance(v, list):
                facets.extend(str(x) for x in v if x)
            elif v:
                facets.append(str(v))

    facets.append(_fallback_query(messages))

    seen, out = set(), []
    for f in facets:
        f = f.strip()
        key = f.lower()
        if f and key not in seen:
            out.append(f)
            seen.add(key)
    return out[:6] or [_fallback_query(messages)]


def _interleave_retrieve(facet_queries: List[str], limit: int) -> List["Assessment"]:
    """Retrieve per facet, then round-robin interleave so each facet's top hits
    land early in the merged candidate list."""
    retriever = get_retriever()
    per_facet: List[List[Assessment]] = []
    for fq in facet_queries:
        hits = retriever.search(fq, 15)
        per_facet.append([a for a, _score in hits])

    merged: List[Assessment] = []
    seen: set = set()
    depth = max((len(h) for h in per_facet), default=0)
    for rank in range(depth):
        for facet_hits in per_facet:
            if rank < len(facet_hits):
                a = facet_hits[rank]
                if a.id not in seen:
                    merged.append(a)
                    seen.add(a.id)
            if len(merged) >= limit:
                return merged
    return merged

def _recommend(facet_queries: List[str], requirements: str, at_cap: bool) -> ChatResponse:
    merged = _interleave_retrieve(facet_queries, settings.retrieve_k)
    if not merged:
        return ChatResponse(
            reply="I couldn't find matching assessments yet. Could you tell me "
            "the role and the key skills involved?",
            recommendations=[],
            end_of_conversation=False,
        )

    candidates = [(i, a) for i, a in enumerate(merged)]
    id_to_asmt = {i: a for i, a in candidates}

    sel = complete_json(
        prompts.SELECTOR_SYSTEM,
        prompts.selector_user(requirements, candidates),
    )

    chosen: List[Assessment] = []
    reply: Optional[str] = None
    if sel and isinstance(sel.get("chosen_ids"), list):
        for cid in sel["chosen_ids"]:
            try:
                a = id_to_asmt.get(int(cid))
            except (TypeError, ValueError):
                a = None
            if a is not None:
                chosen.append(a)
        reply = sel.get("reply") if isinstance(sel.get("reply"), str) else None

    if not chosen:
        chosen = [a for _i, a in candidates[:8]]
        reply = reply or "Here are the assessments that best match what you described."

    if len(chosen) < settings.max_recommendations:
        chosen_ids = {a.id for a in chosen}
        for _i, a in candidates:
            if len(chosen) >= settings.max_recommendations:
                break
            if a.id not in chosen_ids:
                chosen.append(a)
                chosen_ids.add(a.id)

    reply = reply or "Here are the assessments that best match what you described."
    return ChatResponse(
        reply=reply,
        recommendations=_to_recs(chosen),
        end_of_conversation=at_cap,
    )


def _compare(targets: List[str]) -> ChatResponse:
    catalog = load_catalog()
    found = [catalog.find_by_name(t) for t in (targets or [])]
    found = [f for f in found if f is not None]
    if len(found) < 2:
        return ChatResponse(
            reply="I can compare two SHL assessments if you name both. Which two "
            "did you mean?",
            recommendations=[],
            end_of_conversation=False,
        )
    a, b = found[0], found[1]
    out = complete_json(prompts.COMPARE_SYSTEM, prompts.compare_user(a, b))
    reply = (out or {}).get("reply") if isinstance(out, dict) else None
    if not reply:
        # Deterministic grounded fallback from catalog fields only.
        reply = (
            f"{a.name} (type {a.test_type or 'n/a'}) measures: "
            f"{(a.description or 'not specified')[:180]}. "
            f"{b.name} (type {b.test_type or 'n/a'}) measures: "
            f"{(b.description or 'not specified')[:180]}."
        )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

def handle(messages: List[Message]) -> ChatResponse:
    last = _last_user(messages)
    at_cap = len(messages) + 1 >= _TURN_CAP  # our reply would be the final turn

    # Backstop 1: obvious injection -> refuse, no LLM needed.
    if _INJECTION.search(last):
        return ChatResponse(reply=_REFUSAL, recommendations=[], end_of_conversation=False)

    clarified = _assistant_turns(messages)
    turns_left = max(1, (_TURN_CAP - len(messages) + 1) // 2)

    decision = complete_json(
        prompts.CONTROLLER_SYSTEM.replace("{max_clarify}", str(settings.max_clarify_turns)),
        prompts.controller_user(_history_text(messages), clarified, turns_left),
    )

    if not decision or "action" not in decision:
        # Vague + first contact -> clarify; otherwise best-effort recommend.
        if clarified == 0 and len(last.split()) < 6:
            return ChatResponse(
                reply="Happy to help. What role are you hiring for, and what "
                "skills or seniority matter most?",
                recommendations=[],
                end_of_conversation=False,
            )
        return _recommend([_fallback_query(messages)], _fallback_query(messages), at_cap)

    action = str(decision.get("action", "")).lower()
    req = decision.get("requirements", {})
    req_str = json.dumps(req, ensure_ascii=False) if isinstance(req, dict) else str(req)

    force_commit = clarified >= settings.max_clarify_turns or at_cap
    if action == "clarify" and force_commit:
        action = "recommend"

    if action == "refuse":
        reply = decision.get("reply") or _REFUSAL
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    if action == "clarify":
        reply = decision.get("reply") or (
            "Could you tell me a bit more about the role and the skills involved?"
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    if action == "compare":
        return _compare(decision.get("compare_targets", []))

    if action in ("recommend", "refine"):
        return _recommend(_facet_queries(decision, messages), req_str, at_cap)

    return _recommend(_facet_queries(decision, messages), req_str, at_cap)
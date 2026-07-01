from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from app.config import settings


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Optional[dict]:
    """Models sometimes wrap JSON in prose or ```json fences. Be forgiving."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # First balanced-looking object.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _endpoint_and_payload(system: str, user: str):
    p = settings.llm_provider
    key = settings.llm_api_key
    if not key:
        raise LLMError("LLM_API_KEY is not set")

    if p == "groq":
        return (
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {
                "model": settings.llm_model,
                "temperature": 0.1,
                "max_tokens": settings.llm_max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            "openai",
        )
    if p == "openai":
        return (
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {
                "model": settings.llm_model,
                "temperature": 0.1,
                "max_tokens": settings.llm_max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            "openai",
        )
    if p == "anthropic":
        return (
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": settings.llm_model,
                "max_tokens": settings.llm_max_tokens,
                "temperature": 0.1,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            "anthropic",
        )
    if p == "gemini":
        model = settings.llm_model
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            {"Content-Type": "application/json"},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": settings.llm_max_tokens,
                    "responseMimeType": "application/json",
                },
            },
            "gemini",
        )
    raise LLMError(f"Unknown LLM_PROVIDER: {p}")


def _parse_text(kind: str, data: dict) -> str:
    if kind == "openai":
        return data["choices"][0]["message"]["content"]
    if kind == "anthropic":
        return "".join(b.get("text", "") for b in data.get("content", []))
    if kind == "gemini":
        cands = data.get("candidates", [])
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)
    return ""


def complete_json(system: str, user: str) -> Optional[dict]:
    
    url, headers, payload, kind = _endpoint_and_payload(system, user)
    for attempt in range(2):
        try:
            with httpx.Client(timeout=settings.llm_timeout_s) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                text = _parse_text(kind, r.json())
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        user = user + "\n\nReturn ONLY a single valid JSON object. No prose."
    return None

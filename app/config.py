
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:

    llm_provider: str = _get("LLM_PROVIDER", "groq")
    llm_model: str = _get("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_api_key: str = _get("LLM_API_KEY", "")
    llm_timeout_s: float = float(_get("LLM_TIMEOUT_S", "18"))
    llm_max_tokens: int = int(_get("LLM_MAX_TOKENS", "1024"))

    embed_model: str = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    catalog_path: str = _get("CATALOG_PATH", "data/catalog.json")
    # How many candidates vector search hands to the selector.
    retrieve_k: int = int(_get("RETRIEVE_K", "30"))
    
    max_recommendations: int = 10

    max_clarify_turns: int = int(_get("MAX_CLARIFY_TURNS", "1"))


settings = Settings()

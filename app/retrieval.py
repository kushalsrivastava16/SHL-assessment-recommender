from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from app.catalog import Assessment, Catalog, load_catalog
from app.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> List[str]:
    return _TOKEN.findall(s.lower())


class BM25:
    """Standard Okapi BM25 with a postings index (only touches docs that
    contain each query term)."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.postings: dict = defaultdict(list)  # term -> [(doc_idx, tf), ...]
        for i, d in enumerate(docs_tokens):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query_tokens: List[str]) -> List[float]:
        s = [0.0] * self.N
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                s[i] += idf * (tf * (self.k1 + 1)) / denom
        return s


def _rrf(rankings: List[List[int]], rrf_k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion: fused[d] = sum_i 1/(rrf_k + rank_i(d))."""
    fused: dict = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (rrf_k + rank)
    return sorted(fused, key=lambda i: -fused[i])


class Retriever:
    # Size of each single-retriever pool fed into the fusion step.
    POOL = 60

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embed_model)
        texts = [a.embed_text() for a in catalog.items]
        emb = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64
        )
        self.matrix = emb.astype(np.float32)  # (N, d), L2-normalized

        # Lexical index. Weight the name more than the description by repeating
        # name tokens, so an exact product-name match ranks strongly.
        docs_tokens = [
            _tok(a.name) * 3 + _tok(a.test_type) + _tok(a.job_levels) + _tok(a.description)
            for a in catalog.items
        ]
        self.bm25 = BM25(docs_tokens)

    def search(self, query: str, k: int) -> List[Tuple[Assessment, float]]:
        if not query.strip():
            return []

        # Dense ranking.
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        dense_scores = (self.matrix @ q[0]).astype(float)
        dense_rank = list(np.argsort(-dense_scores)[: self.POOL])

        # Lexical ranking (drop zero-score docs so RRF isn't fed noise).
        lex_scores = self.bm25.scores(_tok(query))
        lex_rank = [
            int(i)
            for i in np.argsort(-np.asarray(lex_scores))[: self.POOL]
            if lex_scores[int(i)] > 0
        ]

        fused = _rrf([[int(i) for i in dense_rank], lex_rank])
        if not fused:  # degenerate fallback: pure dense
            fused = [int(i) for i in dense_rank]

        k = min(k, len(fused))
        # Pseudo-score = fusion rank position (higher = better); the selector
        # only needs ordering, not calibrated scores.
        return [(self.catalog.items[idx], 1.0 / (1 + pos)) for pos, idx in enumerate(fused[:k])]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_catalog())
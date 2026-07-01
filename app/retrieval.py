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
    """Okapi BM25 with a postings index (touches only docs containing a term)."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.postings: dict = defaultdict(list)
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
    fused: dict = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (rrf_k + rank)
    return sorted(fused, key=lambda i: -fused[i])


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class Retriever:
    POOL = 60

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        from fastembed import TextEmbedding

        self._embedder = TextEmbedding(model_name=settings.embed_model)
        texts = [a.embed_text() for a in catalog.items]
        emb = np.array(list(self._embedder.embed(texts)), dtype=np.float32)
        self.matrix = _normalize(emb)  # (N, d), L2-normalized

        docs_tokens = [
            _tok(a.name) * 3 + _tok(a.test_type) + _tok(a.job_levels) + _tok(a.description)
            for a in catalog.items
        ]
        self.bm25 = BM25(docs_tokens)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.array(list(self._embedder.embed([query]))[0], dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def search(self, query: str, k: int) -> List[Tuple[Assessment, float]]:
        if not query.strip():
            return []

        q = self._embed_query(query)
        dense_scores = (self.matrix @ q).astype(float)
        dense_rank = [int(i) for i in np.argsort(-dense_scores)[: self.POOL]]

        lex_scores = self.bm25.scores(_tok(query))
        lex_rank = [
            int(i)
            for i in np.argsort(-np.asarray(lex_scores))[: self.POOL]
            if lex_scores[int(i)] > 0
        ]

        fused = _rrf([dense_rank, lex_rank]) or dense_rank
        k = min(k, len(fused))
        return [(self.catalog.items[idx], 1.0 / (1 + pos)) for pos, idx in enumerate(fused[:k])]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_catalog())

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings


@dataclass
class Assessment:
    id: str
    name: str
    url: str
    test_type: str = ""            
    description: str = ""
    job_levels: str = ""
    remote: bool = False
    adaptive: bool = False
    duration: str = ""

    def embed_text(self) -> str:
        """Text used to build the retrieval embedding. Concatenating the
        fields a hiring manager actually reasons about (name, what it
        measures, who it's for, type) gives the query something to match on
        beyond the literal title."""
        parts = [
            self.name,
            f"Test type: {self.test_type}" if self.test_type else "",
            f"Job levels: {self.job_levels}" if self.job_levels else "",
            self.description,
        ]
        return "\n".join(p for p in parts if p)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class Catalog:
    def __init__(self, items: List[Assessment]):
        self.items = items
        self._by_id: Dict[str, Assessment] = {a.id: a for a in items}
        self._by_norm_name: Dict[str, Assessment] = {_norm(a.name): a for a in items}

    def __len__(self) -> int:
        return len(self.items)

    def get(self, id_: str) -> Optional[Assessment]:
        return self._by_id.get(id_)

    def valid_ids(self, ids: List[str]) -> List[Assessment]:
        """Drop anything the model invents; keep catalog order-stable."""
        seen, out = set(), []
        for i in ids:
            a = self._by_id.get(i)
            if a and a.id not in seen:
                out.append(a)
                seen.add(a.id)
        return out

    def find_by_name(self, query: str) -> Optional[Assessment]:
        """Fuzzy-ish match for compare queries ("OPQ", "GSA"). Exact norm
        first, then substring both ways."""
        q = _norm(query)
        if q in self._by_norm_name:
            return self._by_norm_name[q]
        for norm_name, a in self._by_norm_name.items():
            if q and (q in norm_name or norm_name in q):
                return a
        return None


@lru_cache(maxsize=1)
def load_catalog(path: Optional[str] = None) -> Catalog:
    p = Path(path or settings.catalog_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Catalog not found at {p}. Run `python -m scraper.scrape_catalog` "
            f"first (or point CATALOG_PATH at data/catalog.sample.json to smoke-test)."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = [Assessment(**r) for r in raw]
    if not items:
        raise ValueError(f"Catalog at {p} is empty.")
    return Catalog(items)

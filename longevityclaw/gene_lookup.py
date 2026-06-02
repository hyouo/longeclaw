"""
Gene annotation via MyGene.info with persistent SQLite cache.
Lazy singleton — the biothings_client is only instantiated on first query,
not at import time, so it doesn't slow down startup.

Cache: data/mygene_cache.sqlite (pre-populated, 211 MB)
Config via .env:
  LONGEVITYCLAW_MYGENE_CACHE  — path to SQLite cache (default: data/mygene_cache.sqlite)
  LONGEVITYCLAW_MYGENE_TTL    — cache lifetime in days; -1 = never expire (default)
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "mygene_cache.sqlite"
DEFAULT_FIELDS = "symbol,name,summary,alias,type_of_gene,go,pathway.reactome"

_instance: "GeneLookup | None" = None


def get_gene_client() -> "GeneLookup":
    global _instance
    if _instance is None:
        _instance = GeneLookup()
    return _instance


class GeneLookup:
    def __init__(self):
        self._mg = None
        self._memory_cache: dict[str, dict | None] = {}

    def _get_client(self):
        if self._mg is not None:
            return self._mg

        from biothings_client import get_client

        mg = get_client("gene")

        cache_path = os.environ.get("LONGEVITYCLAW_MYGENE_CACHE", str(DEFAULT_CACHE))

        mg._default_cache_file = cache_path
        mg.set_caching(cache_db=cache_path, verbose=False)

        # hishel's POST cache ignores request body, making querymany unreliable.
        # We use individual query() calls (GET) instead, which cache correctly.
        # _force_cache is NOT set — cache misses fall through to the network.

        self._mg = mg
        return mg

    def lookup(self, symbol: str) -> dict:
        upper = symbol.upper()
        if upper in self._memory_cache:
            cached = self._memory_cache[upper]
            return cached if cached else {"symbol": symbol, "found": False}

        mg = self._get_client()
        try:
            result = mg.query(symbol, scopes="symbol,alias", species="human", fields=DEFAULT_FIELDS)
        except Exception as e:
            logger.warning(f"MyGene query failed for '{symbol}': {e}")
            return {"symbol": symbol, "found": False, "error": str(e)}

        hits = result.get("hits", [])
        if not hits or "notfound" in hits[0]:
            self._memory_cache[upper] = None
            return {"symbol": symbol, "found": False}

        hit = hits[0]
        if not self._is_valid_match(symbol, hit):
            self._memory_cache[upper] = None
            return {"symbol": symbol, "found": False}

        parsed = self._parse_hit(hit)
        self._memory_cache[upper] = parsed
        return parsed

    def lookup_batch(self, symbols: list[str]) -> dict[str, dict]:
        # Individual query() calls instead of querymany — hishel's POST cache
        # ignores request body, so querymany returns wrong cached results.
        return {sym: self.lookup(sym) for sym in symbols}

    @staticmethod
    def _is_valid_match(query: str, hit: dict) -> bool:
        returned = hit.get("symbol", "").upper()
        aliases = hit.get("alias", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases_upper = [a.upper() for a in aliases]
        return query.upper() == returned or query.upper() in aliases_upper

    @staticmethod
    def _parse_hit(hit: dict) -> dict:
        result = {
            "found": True,
            "symbol": hit.get("symbol", ""),
            "name": hit.get("name", ""),
            "summary": hit.get("summary", ""),
            "type": hit.get("type_of_gene", ""),
            "aliases": hit.get("alias", []),
        }
        if isinstance(result["aliases"], str):
            result["aliases"] = [result["aliases"]]

        go = hit.get("go", {})
        if go:
            result["go_terms"] = {}
            for cat, label in [("MF", "molecular_function"), ("BP", "biological_process"), ("CC", "cellular_component")]:
                terms = go.get(cat, [])
                if isinstance(terms, dict):
                    terms = [terms]
                seen: set[str] = set()
                unique = []
                for t in terms:
                    if not isinstance(t, dict) or str(t.get("qualifier", "")).startswith("NOT"):
                        continue
                    term_name = t.get("term", "")
                    if term_name and term_name not in seen:
                        seen.add(term_name)
                        unique.append(term_name)
                result["go_terms"][label] = unique[:10]

        pathways = hit.get("pathway", {})
        reactome = pathways.get("reactome", []) if isinstance(pathways, dict) else []
        if isinstance(reactome, dict):
            reactome = [reactome]
        if reactome:
            result["reactome_pathways"] = [
                {"id": p.get("id", ""), "name": p.get("name", "")}
                for p in reactome
                if isinstance(p, dict)
            ]

        return result

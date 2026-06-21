from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from arion_agent.semantic_search.config import (
    BM25_WEIGHT,
    DEFAULT_WORKSPACE,
    DEPRIORITIZE_PATH_SUBSTRINGS,
    FINAL_TOP_K,
    MIN_HYBRID_SCORE,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
    resolve_index_dir,
)
from arion_agent.semantic_search.embedder import get_embedder
from arion_agent.semantic_search.ignore import path_matches_glob
from arion_agent.semantic_search.store import ChunkStore, quote_literal
from arion_agent.semantic_search.translate import prepare_search_text


@dataclass(frozen=True, slots=True)
class SearchResult:
    path: str
    start_line: int
    end_line: int
    kind: str
    score: float
    vector_score: float
    bm25_score: float
    snippet: str


# ── Token helpers ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "how", "what", "when", "where", "why", "does", "do", "did",
    "work", "works", "working", "use", "used", "using",
})


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _query_tokens(query: str) -> set[str]:
    tokens = set(_tokenize(query)) - STOPWORDS
    return tokens if tokens else set(_tokenize(query))


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


# ── Query-time filters (pushed to LanceDB WHERE) ─────────────────────

def _build_where(
    target_directories: list[str] | None,
    path_glob: str | None,
) -> str | None:
    """Build a LanceDB WHERE clause from scope filters.

    Pushes down ``target_directories`` as ``path LIKE 'dir/%'`` chains
    and simple ``path_glob`` patterns (suffix / prefix / extension) as
    ``path LIKE`` expressions.  Complex globs that cannot be expressed
    in SQL LIKE are left as post-filters (handled in _post_filter).
    """
    clauses: list[str] = []

    if target_directories:
        dir_clauses = [
            f"LOWER(path) LIKE LOWER({quote_literal(d.strip('./ ').rstrip('/') + '/%')})"
            for d in target_directories
            if d and d.strip("./ ")
        ]
        if dir_clauses:
            clauses.append("(" + " OR ".join(dir_clauses) + ")")

    if path_glob:
        like = _glob_to_sql_like(path_glob)
        if like is not None:
            clauses.append(f"LOWER(path) LIKE LOWER({quote_literal(like)})")

    if not clauses:
        return None
    return " AND ".join(clauses)


def _glob_to_sql_like(pattern: str) -> str | None:
    """Translate a simple glob into an SQL LIKE pattern.

    Returns ``None`` for complex patterns that cannot be expressed as LIKE.
    """
    p = pattern.strip()
    if not p:
        return None

    # prefix/**/*.ext → prefix/%.ext  (most common: src/**/*.py)
    m = re.match(r"^(.+)/\*\*/\*\.(\w+)$", p)
    if m:
        return f"{m.group(1)}/%.{m.group(2)}"

    # dir/** → dir/%
    if p.endswith("/**"):
        return p[:-3] + "/%"

    # **/*.ext → %.ext
    if p.startswith("**/"):
        rest = p[3:]
        # only simple: **/*.py → %.py; reject **/* or **/foo/bar
        if "/" not in rest and rest.startswith("*."):
            return f"%{rest[1:]}"
        return None

    # *.ext → %.ext
    if p.startswith("*."):
        return f"%{p[1:]}"

    # src/*.py → src/%.py  (single-level wildcard, no ** in prefix)
    if "/*." in p and "**" not in p:
        prefix, suffix = p.split("/*.", 1)
        if "*" not in prefix:
            return f"{prefix}/%.{suffix}"
        return None

    # No wildcards → exact match
    if "*" not in p and "?" not in p:
        return p

    return None  # too complex for LIKE


# ── Post-filter for complex globs ────────────────────────────────────

def _post_filter(row, target_directories: list[str] | None, path_glob: str | None) -> bool:
    """Post-filter a result row against scope constraints.

    Only checks constraints that were NOT pushed to the WHERE clause.
    """
    path = row["path"]

    # target_directories is always pushed to WHERE — nothing to check here
    if path_glob:
        # Only check if the glob couldn't be pushed down
        like = _glob_to_sql_like(path_glob)
        if like is None and not path_matches_glob(path, path_glob):
            return False

    return True


# ── Score helpers ────────────────────────────────────────────────────

def _literal_token_hit(query_tokens: set[str], text: str, path: str) -> bool:
    hay = f"{path}\n{text}".lower()
    return any(len(t) >= 3 and t in hay for t in query_tokens)


def _symbol_boost(query_tokens: set[str], text: str) -> float:
    boost = 0.0
    for token in query_tokens:
        if len(token) < 4:
            continue
        if f"def {token}" in text:
            boost += 0.18
        elif f"function {token}" in text:
            boost += 0.14
        elif f"class {token}" in text:
            boost += 0.14
    return min(boost, 0.22)


def _is_deprioritized_path(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    return any(sub in norm for sub in DEPRIORITIZE_PATH_SUBSTRINGS)


def _path_boost(query_tokens: set[str], path: str) -> float:
    path_tokens = set(_tokenize(path.replace("/", " ").replace("_", " ").replace(".", " ")))
    overlap = len(query_tokens & path_tokens)
    if overlap == 0:
        return 0.0
    return min(0.12, 0.04 * overlap)


# ── Hybrid search entrypoint ─────────────────────────────────────────

def hybrid_search(
    query: str,
    *,
    workspace: Path | None = None,
    index_dir: Path | None = None,
    store: ChunkStore | None = None,
    target_directories: list[str] | None = None,
    path_glob: str | None = None,
    num_results: int = FINAL_TOP_K,
    min_score: float = MIN_HYBRID_SCORE,
) -> list[SearchResult]:
    if store is None:
        ws = (workspace or DEFAULT_WORKSPACE).resolve()
        resolved_index = resolve_index_dir(ws, index_dir)
        store = ChunkStore(resolved_index)

    if not store.has_table():
        return []

    search_query = prepare_search_text(query)
    query_tokens = _query_tokens(search_query)
    embedder = get_embedder()
    query_vec = embedder.embed_query(search_query)

    # Build and push down WHERE clause for efficient pre-filtering
    where = _build_where(target_directories, path_glob)
    vector_hits = store.vector_search(query_vec, VECTOR_TOP_K, where=where)

    # Apply any remaining post-filters (complex globs not expressible as LIKE)
    if path_glob and _glob_to_sql_like(path_glob) is None:
        vector_hits = [
            row for row in vector_hits
            if _post_filter(row, target_directories, path_glob)
        ]

    if not vector_hits:
        return []

    # ── BM25 re-rank on vector candidates ────────────────────────────
    corpus_tokens = [
        _tokenize(row.get("search_text") or row["text"]) for row in vector_hits
    ]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw = bm25.get_scores(_tokenize(search_query))
    bm25_norm = _normalize_scores(list(bm25_raw))

    vector_raw = [float(row.get("_distance", 0.0)) for row in vector_hits]
    vector_sim = [1.0 / (1.0 + max(0.0, d)) for d in vector_raw]
    vector_norm = _normalize_scores(vector_sim)

    # ── Hybrid fusion ────────────────────────────────────────────────
    merged: list[SearchResult] = []
    for i, row in enumerate(vector_hits):
        if not _literal_token_hit(
            query_tokens,
            row.get("search_text") or row["text"],
            row["path"],
        ) and bm25_norm[i] < 0.35:
            continue

        hybrid = (
            VECTOR_WEIGHT * vector_norm[i]
            + BM25_WEIGHT * bm25_norm[i]
            + _path_boost(query_tokens, row["path"])
            + _symbol_boost(query_tokens, row["text"])
        )

        norm_path = row["path"].replace("\\", "/").lower()
        if "/non-coding-agent-semantic-search/" in norm_path and bm25_norm[i] < 0.55:
            hybrid *= 0.4

        if _is_deprioritized_path(row["path"]) and bm25_norm[i] < 0.35:
            hybrid *= 0.5

        if hybrid < min_score:
            continue

        text = row["text"]
        snippet = text if len(text) <= 900 else text[:900] + "\n..."
        merged.append(
            SearchResult(
                path=row["path"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                kind=row["kind"],
                score=hybrid,
                vector_score=vector_norm[i],
                bm25_score=bm25_norm[i],
                snippet=snippet,
            )
        )

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:num_results]

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
    INCREMENTAL_BATCH_FILES,
    MIN_HYBRID_SCORE,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
    resolve_index_dir,
)
from arion_agent.semantic_search.embedder import get_embedder
from arion_agent.semantic_search.store import ChunkStore
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "how", "what", "when", "where", "why", "does", "do", "did",
    "work", "works", "working", "use", "used", "using",
})


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


def _path_matches(path: str, prefixes: list[str] | None) -> bool:
    if not prefixes:
        return True
    norm_path = path.replace("\\", "/").lower()
    for prefix in prefixes:
        p = prefix.replace("\\", "/").strip().lower().strip("./")
        if not p:
            continue
        if norm_path == p or norm_path.startswith(p + "/"):
            return True
    return False


def _is_deprioritized_path(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    return any(sub in norm for sub in DEPRIORITIZE_PATH_SUBSTRINGS)


def _path_boost(query_tokens: set[str], path: str) -> float:
    path_tokens = set(_tokenize(path.replace("/", " ").replace("_", " ").replace(".", " ")))
    overlap = len(query_tokens & path_tokens)
    if overlap == 0:
        return 0.0
    return min(0.12, 0.04 * overlap)


def hybrid_search(
    query: str,
    *,
    workspace: Path | None = None,
    index_dir: Path | None = None,
    store: ChunkStore | None = None,
    target_directories: list[str] | None = None,
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
    vector_hits = store.vector_search(query_vec, VECTOR_TOP_K)

    scoped = [
        row for row in vector_hits if _path_matches(row["path"], target_directories)
    ]
    if target_directories and not scoped:
        return []

    candidates = scoped if scoped else vector_hits
    if not candidates:
        return []

    corpus_tokens = [
        _tokenize(row.get("search_text") or row["text"]) for row in candidates
    ]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw = bm25.get_scores(_tokenize(search_query))
    bm25_norm = _normalize_scores(list(bm25_raw))

    vector_raw = [float(row.get("_distance", 0.0)) for row in candidates]
    vector_sim = [1.0 / (1.0 + max(0.0, d)) for d in vector_raw]
    vector_norm = _normalize_scores(vector_sim)

    merged: list[SearchResult] = []
    for i, row in enumerate(candidates):
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

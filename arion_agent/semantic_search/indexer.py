from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from arion_agent.semantic_search.chunker import chunk_file
from arion_agent.semantic_search.config import (
    INDEX_CHUNK_WORKERS,
    INCREMENTAL_BATCH_FILES,
    TEXT_EXTENSIONS,
    Chunk,
    resolve_index_dir,
)
from arion_agent.semantic_search.embedder import get_embedder
from arion_agent.semantic_search.ignore import iter_indexable_files, load_ignore_patterns
from arion_agent.semantic_search.store import ChunkStore
from arion_agent.semantic_search.translate import prepare_search_text


@dataclass(frozen=True, slots=True)
class IndexStats:
    files_seen: int
    files_indexed: int
    files_skipped_unchanged: int
    files_renamed: int
    files_removed: int
    chunks_embedded: int
    chunks_reused: int
    chunks: int
    elapsed_sec: float


@dataclass(frozen=True, slots=True)
class BatchIndexStats:
    files_indexed: int
    chunks_embedded: int
    chunks_reused: int
    elapsed_sec: float


@dataclass(frozen=True, slots=True)
class BatchIndexResult:
    stats: BatchIndexStats
    chunks: list[Chunk]
    vectors: list[list[float]]
    manifest_entries: dict[str, str]


@dataclass(frozen=True, slots=True)
class SyncPlan:
    renames: list[tuple[str, str]] = field(default_factory=list)
    removals: list[str] = field(default_factory=list)
    to_index: list[Path] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def file_content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_manifest(
    workspace: Path,
    patterns: list[str] | None = None,
) -> dict[str, str]:
    workspace = workspace.resolve()
    if patterns is None:
        patterns = load_ignore_patterns(workspace)
    files = iter_indexable_files(workspace, patterns, TEXT_EXTENSIONS)
    return {
        path.relative_to(workspace).as_posix(): file_content_hash(path)
        for path in files
    }


def detect_renames(
    old_manifest: dict[str, str],
    new_manifest: dict[str, str],
) -> list[tuple[str, str]]:
    removed = set(old_manifest) - set(new_manifest)
    added = set(new_manifest) - set(old_manifest)
    renames: list[tuple[str, str]] = []

    for new_path in list(added):
        new_hash = new_manifest[new_path]
        matches = [old for old in removed if old_manifest[old] == new_hash]
        if len(matches) == 1:
            old_path = matches[0]
            renames.append((old_path, new_path))
            removed.remove(old_path)
            added.remove(new_path)

    return renames


def plan_sync(
    workspace: Path,
    old_manifest: dict[str, str],
    patterns: list[str] | None = None,
) -> SyncPlan:
    workspace = workspace.resolve()
    new_manifest = scan_manifest(workspace, patterns)
    renames = detect_renames(old_manifest, new_manifest)

    renamed_old = {old for old, _ in renames}
    renamed_new = {new for _, new in renames}

    removals = [
        path for path in old_manifest
        if path not in new_manifest and path not in renamed_old
    ]

    unchanged: list[str] = []
    to_index: list[Path] = []

    for rel, digest in new_manifest.items():
        if rel in renamed_new:
            continue
        if rel in old_manifest and old_manifest[rel] == digest:
            unchanged.append(rel)
            continue
        to_index.append(workspace / rel)

    return SyncPlan(
        renames=renames,
        removals=removals,
        to_index=sorted(to_index),
        unchanged=unchanged,
    )


def apply_renames(
    store: ChunkStore,
    renames: list[tuple[str, str]],
) -> int:
    manifest = store.load_manifest()
    updates: dict[str, str] = {}
    removes: list[str] = []
    moved_count = 0

    for old_path, new_path in renames:
        moved = store.rename_path(old_path, new_path)
        if moved == 0:
            continue
        moved_count += 1
        digest = manifest.pop(old_path, None)
        if digest is not None:
            updates[new_path] = digest
            removes.append(old_path)

    if removes:
        store.remove_manifest_entries(removes)
    if updates:
        store.update_manifest_entries(updates)

    return moved_count


def apply_removals(store: ChunkStore, removals: list[str]) -> None:
    for path in removals:
        store.delete_by_path(path)
    if removals:
        store.remove_manifest_entries(removals)


def _chunk_one(path: Path, workspace: Path) -> list[Chunk]:
    return chunk_file(path, workspace)


def _prepare_chunks(raw_chunks: list[Chunk]) -> list[Chunk]:
    prepared: list[Chunk] = []
    for chunk in raw_chunks:
        search_text = prepare_search_text(chunk.text)
        if search_text == chunk.search_text:
            prepared.append(chunk)
        else:
            prepared.append(
                Chunk(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    search_text=search_text,
                    kind=chunk.kind,
                    content_hash=chunk.content_hash,
                )
            )
    return prepared


def _vector_cache_from_store(store: ChunkStore) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for row in store.all_rows():
        cache_key = row.get("search_text_hash") or row["content_hash"]
        cache[cache_key] = row["vector"]
    return cache


def index_file_batch(
    paths: list[Path],
    workspace: Path,
    store: ChunkStore | None = None,
    vector_cache: dict[str, list[float]] | None = None,
    *,
    persist: bool = True,
) -> BatchIndexResult:
    started = time.perf_counter()
    if not paths:
        empty = BatchIndexStats(0, 0, 0, 0.0)
        return BatchIndexResult(empty, [], [], {})

    workspace = workspace.resolve()
    if vector_cache is None and store is not None:
        vector_cache = _vector_cache_from_store(store)
    if vector_cache is None:
        vector_cache = {}

    raw_chunks: list[Chunk] = []
    with ThreadPoolExecutor(max_workers=INDEX_CHUNK_WORKERS) as pool:
        for chunks in pool.map(lambda p: _chunk_one(p, workspace), paths):
            raw_chunks.extend(chunks)

    prepared = _prepare_chunks(raw_chunks)

    by_path: dict[str, list[Chunk]] = {}
    for chunk in prepared:
        by_path.setdefault(chunk.path, []).append(chunk)

    embedder = get_embedder()
    files_indexed = 0
    chunks_embedded = 0
    chunks_reused = 0
    manifest_entries: dict[str, str] = {}
    batch_chunks: list[Chunk] = []
    batch_vectors: list[list[float]] = []

    for path in paths:
        rel = path.relative_to(workspace).as_posix()
        file_chunks = by_path.get(rel, [])
        if not file_chunks:
            if persist and store is not None:
                store.delete_by_path(rel)
                store.remove_manifest_entries([rel])
            continue

        need_embed: list[Chunk] = []
        need_embed_indices: list[int] = []
        vectors: list[list[float] | None] = [None] * len(file_chunks)

        for i, chunk in enumerate(file_chunks):
            cached = vector_cache.get(chunk.search_text_hash)
            if cached is not None:
                vectors[i] = cached
                chunks_reused += 1
            else:
                need_embed.append(chunk)
                need_embed_indices.append(i)

        if need_embed:
            new_vectors = embedder.embed_documents([c.search_text for c in need_embed])
            for chunk, vector, idx in zip(
                need_embed, new_vectors, need_embed_indices, strict=True
            ):
                vectors[idx] = vector
                vector_cache[chunk.search_text_hash] = vector
                chunks_embedded += 1

        resolved = [v for v in vectors if v is not None]
        batch_chunks.extend(file_chunks)
        batch_vectors.extend(resolved)
        manifest_entries[rel] = file_content_hash(path)
        files_indexed += 1

        if persist and store is not None:
            store.upsert_file(file_chunks, resolved)
            store.update_manifest_entries({rel: manifest_entries[rel]})

    stats = BatchIndexStats(
        files_indexed=files_indexed,
        chunks_embedded=chunks_embedded,
        chunks_reused=chunks_reused,
        elapsed_sec=time.perf_counter() - started,
    )
    return BatchIndexResult(stats, batch_chunks, batch_vectors, manifest_entries)


def index_workspace(
    workspace: Path,
    index_dir: Path | None = None,
    *,
    force: bool = False,
    patterns: list[str] | None = None,
) -> IndexStats:
    started = time.perf_counter()
    workspace = workspace.resolve()
    index_dir = resolve_index_dir(workspace, index_dir)
    if patterns is None:
        patterns = load_ignore_patterns(workspace)

    store = ChunkStore(index_dir)
    old_manifest = {} if force else store.load_manifest()
    plan = plan_sync(workspace, old_manifest, patterns)

    if (
        not force
        and not plan.to_index
        and not plan.removals
        and not plan.renames
        and store.has_table()
    ):
        return IndexStats(
            files_seen=len(plan.unchanged),
            files_indexed=0,
            files_skipped_unchanged=len(plan.unchanged),
            files_renamed=0,
            files_removed=0,
            chunks_embedded=0,
            chunks_reused=store.chunk_count(),
            chunks=store.chunk_count(),
            elapsed_sec=time.perf_counter() - started,
        )

    apply_renames(store, plan.renames)
    apply_removals(store, plan.removals)

    vector_cache = {} if force else _vector_cache_from_store(store)
    all_chunks: list[Chunk] = []
    all_vectors: list[list[float]] = []

    if not force:
        unchanged_rows = [
            row for row in store.all_rows()
            if row["path"] in plan.unchanged
        ]
        for row in unchanged_rows:
            all_chunks.append(
                Chunk(
                    path=row["path"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=row["text"],
                    search_text=row.get("search_text") or row["text"],
                    kind=row["kind"],
                    content_hash=row["content_hash"],
                )
            )
            all_vectors.append(row["vector"])

    if force:
        plan = plan_sync(workspace, {}, patterns)

    total_embedded = 0
    total_reused = 0
    paths = plan.to_index

    for i in range(0, len(paths), INCREMENTAL_BATCH_FILES):
        batch = paths[i : i + INCREMENTAL_BATCH_FILES]
        result = index_file_batch(
            batch,
            workspace,
            store=store if not force else None,
            vector_cache=vector_cache,
            persist=not force,
        )
        total_embedded += result.stats.chunks_embedded
        total_reused += result.stats.chunks_reused
        if force:
            all_chunks.extend(result.chunks)
            all_vectors.extend(result.vectors)

    if force:
        store.replace_all(all_chunks, all_vectors)
        store.save_manifest(scan_manifest(workspace, patterns))
    elif plan.to_index:
        store.save_manifest(scan_manifest(workspace, patterns))

    new_manifest = scan_manifest(workspace, patterns)
    files_seen = len(new_manifest)
    return IndexStats(
        files_seen=files_seen,
        files_indexed=len(plan.to_index),
        files_skipped_unchanged=len(plan.unchanged),
        files_renamed=len(plan.renames),
        files_removed=len(plan.removals),
        chunks_embedded=total_embedded,
        chunks_reused=total_reused,
        chunks=store.chunk_count(),
        elapsed_sec=time.perf_counter() - started,
    )

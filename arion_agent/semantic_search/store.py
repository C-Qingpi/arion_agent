from __future__ import annotations

import json
import threading
from pathlib import Path

import lancedb
import pyarrow as pa

from arion_agent.semantic_search.config import (
    IVF_NUM_PARTITIONS,
    IVF_NUM_SUB_VECTORS,
    Chunk,
)


TABLE_NAME = "chunks"
INDEX_NAME = "vector_idx"
VECTOR_COLUMN = "vector"


class ChunkStore:
    """Persistent LanceDB-backed chunk store with IVF_PQ index.

    All mutation operations (upsert, delete, rename, replace_all) preserve
    the ANN index so that vector search remains fast at any data size.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(index_dir / "lance"))
        self.manifest_path = index_dir / "manifest.json"
        self._write_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._index_exists = False

    # ── Manifest management ──────────────────────────────────────────

    def load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def save_manifest(self, manifest: dict[str, str]) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update_manifest_entries(self, entries: dict[str, str]) -> None:
        manifest = self.load_manifest()
        manifest.update(entries)
        self.save_manifest(manifest)

    def remove_manifest_entries(self, paths: list[str]) -> None:
        manifest = self.load_manifest()
        for path in paths:
            manifest.pop(path, None)
        self.save_manifest(manifest)

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _row_dict(chunk: Chunk, vector: list[float]) -> dict:
        return {
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "kind": chunk.kind,
            "content_hash": chunk.content_hash,
            "text": chunk.text,
            "search_text": chunk.search_text,
            "search_text_hash": chunk.search_text_hash,
            VECTOR_COLUMN: vector,
        }

    def _ensure_index(self) -> None:
        """Create IVF_PQ index on the vector column if it does not exist."""
        if self._index_exists:
            return
        with self._index_lock:
            if self._index_exists:
                return
            try:
                table = self.db.open_table(TABLE_NAME)
            except Exception:
                return
            # LanceDB 0.33 list_indices() is session-local — a recognized
            # index in this session means it was created (or verified) here.
            existing = [idx.name for idx in table.list_indices()]
            if INDEX_NAME in existing:
                self._index_exists = True
                return
            # On-disk index files from a prior session are invisible to
            # list_indices() due to a LanceDB metadata quirk. We rebuild
            # with replace=True so the current version can use them.

            row_count = table.count_rows()
            if row_count < IVF_NUM_SUB_VECTORS * 8:
                return  # too few rows for PQ training — skip until we have enough

            try:
                table.create_index(
                    metric="cosine",
                    num_partitions=min(IVF_NUM_PARTITIONS, row_count // 4),
                    num_sub_vectors=IVF_NUM_SUB_VECTORS,
                    index_type="IVF_PQ",
                    name=INDEX_NAME,
                    replace=True,  # overwrite any orphaned index files (LanceDB version skew)
                )
                self._index_exists = True
            except Exception:
                pass  # non-critical; search still works via exhaustive scan

    def _ensure_table(self) -> object | None:
        """Return the table handle, or None if the table does not exist."""
        try:
            return self.db.open_table(TABLE_NAME)
        except Exception:
            return None

    # ── Table lifecycle ──────────────────────────────────────────────

    def clear(self) -> None:
        with self._write_lock:
            for name in self.db.table_names():
                if name == TABLE_NAME:
                    self.db.drop_table(TABLE_NAME)
            self.save_manifest({})
            self._index_exists = False

    def replace_all(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace entire store contents and (re)build the vector index."""
        with self._write_lock:
            if TABLE_NAME in self.db.table_names():
                self.db.drop_table(TABLE_NAME)

            rows = [
                self._row_dict(chunk, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            if rows:
                self.db.create_table(TABLE_NAME, pa.Table.from_pylist(rows))
                self._index_exists = False
                self._ensure_index()

    # ── Incremental mutations (index-preserving) ─────────────────────

    def delete_by_path(self, path: str) -> None:
        """Delete all chunks for a given file path. Preserves the index."""
        with self._write_lock:
            table = self._ensure_table()
            if table is None:
                return
            table.delete(f"path = {quote_literal(path)}")

    def upsert_file(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert all chunks for a single file path. Preserves the index."""
        if not chunks:
            return
        with self._write_lock:
            path = chunks[0].path
            new_rows = [
                self._row_dict(chunk, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            new_table = pa.Table.from_pylist(new_rows)

            existing = self._ensure_table()
            if existing is None:
                # First write — create table, build index if enough rows
                self.db.create_table(TABLE_NAME, new_table)
                self._ensure_index()
                return

            # Delete old chunks for this path, then insert fresh.
            # This avoids ambiguous merge inserts when multiple chunks
            # share the same path key (the norm for any non-trivial file).
            existing.delete(f"path = {quote_literal(path)}")
            existing.add(new_table)
            self._ensure_index()

    def rename_path(self, old_path: str, new_path: str) -> int:
        """Rename path in all chunks. Preserves the index."""
        with self._write_lock:
            table = self._ensure_table()
            if table is None:
                return 0
            result = table.update(
                where=f"path = {quote_literal(old_path)}",
                values={"path": new_path},
            )
            n = getattr(result, "rows_updated", 0)
            return n or 0

    # ── Read operations ──────────────────────────────────────────────

    def has_table(self) -> bool:
        return TABLE_NAME in self.db.table_names()

    def table(self):
        return self.db.open_table(TABLE_NAME)

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        where: str | None = None,
    ) -> list[dict]:
        """ANN vector search, optionally pre-filtered with a LanceDB WHERE clause.

        Parameters
        ----------
        query_vector : list[float]
            Query embedding vector.
        limit : int
            Max results to return.
        where : str | None
            Optional SQL-like filter (e.g. ``"path LIKE 'src/%'"``).
            Applied BEFORE the ANN search for efficient pre-filtering.
        """
        try:
            t = self.db.open_table(TABLE_NAME)
        except Exception:
            return []

        q = t.search(query_vector).limit(limit)
        if where:
            q = q.where(where)
        return q.to_list()

    def chunk_count(self) -> int:
        try:
            t = self.db.open_table(TABLE_NAME)
            return t.count_rows()
        except Exception:
            return 0

    def all_rows(self) -> list[dict]:
        """Load all rows (expensive — avoid for large stores)."""
        try:
            t = self.db.open_table(TABLE_NAME)
            return t.to_arrow().to_pylist()
        except Exception:
            return []

    def indexed_paths(self) -> set[str]:
        return {row["path"] for row in self.all_rows()}


def quote_literal(value: str) -> str:
    """Quote a string for use as a SQL string literal in LanceDB's dialect."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"

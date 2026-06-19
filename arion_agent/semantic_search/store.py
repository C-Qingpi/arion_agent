from __future__ import annotations

import json
import threading
from pathlib import Path

import lancedb
import pyarrow as pa

from arion_agent.semantic_search.config import Chunk


TABLE_NAME = "chunks"


class ChunkStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(index_dir / "lance"))
        self.manifest_path = index_dir / "manifest.json"
        self._lock = threading.RLock()

    def load_manifest(self) -> dict[str, str]:
        with self._lock:
            if not self.manifest_path.exists():
                return {}
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: dict[str, str]) -> None:
        with self._lock:
            self.manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def update_manifest_entries(self, entries: dict[str, str]) -> None:
        with self._lock:
            manifest = self.load_manifest()
            manifest.update(entries)
            self.save_manifest(manifest)

    def remove_manifest_entries(self, paths: list[str]) -> None:
        with self._lock:
            manifest = self.load_manifest()
            for path in paths:
                manifest.pop(path, None)
            self.save_manifest(manifest)

    def _row_dict(self, chunk: Chunk, vector: list[float]) -> dict:
        return {
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "kind": chunk.kind,
            "content_hash": chunk.content_hash,
            "text": chunk.text,
            "search_text": chunk.search_text,
            "search_text_hash": chunk.search_text_hash,
            "vector": vector,
        }

    def replace_all(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._lock:
            rows = [
                self._row_dict(chunk, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            table = pa.Table.from_pylist(rows)
            if TABLE_NAME in self.db.table_names():
                self.db.drop_table(TABLE_NAME)
            if rows:
                self.db.create_table(TABLE_NAME, table)

    def delete_by_path(self, path: str) -> None:
        with self._lock:
            if not self.has_table():
                return
            kept = [
                row for row in self.all_rows_unlocked()
                if row["path"] != path
            ]
            self.db.drop_table(TABLE_NAME)
            if kept:
                self.db.create_table(TABLE_NAME, pa.Table.from_pylist(kept))

    def upsert_file(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._lock:
            if not chunks:
                return
            path = chunks[0].path
            new_rows = [
                self._row_dict(chunk, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            kept = [
                row for row in self.all_rows_unlocked()
                if row["path"] != path
            ]
            kept.extend(new_rows)
            merged = pa.Table.from_pylist(kept)
            if TABLE_NAME in self.db.table_names():
                self.db.drop_table(TABLE_NAME)
            self.db.create_table(TABLE_NAME, merged)

    def rename_path(self, old_path: str, new_path: str) -> int:
        with self._lock:
            if not self.has_table():
                return 0
            rows = [dict(row) for row in self.all_rows_unlocked()]
            updated = 0
            for row in rows:
                if row["path"] == old_path:
                    row["path"] = new_path
                    updated += 1
            if updated == 0:
                return 0
            merged = pa.Table.from_pylist(rows)
            self.db.drop_table(TABLE_NAME)
            self.db.create_table(TABLE_NAME, merged)
            return updated

    def has_table(self) -> bool:
        with self._lock:
            return TABLE_NAME in self.db.table_names()

    def table(self):
        return self.db.open_table(TABLE_NAME)

    def vector_search(self, query_vector: list[float], limit: int) -> list[dict]:
        with self._lock:
            if not self.has_table():
                return []
            return (
                self.table()
                .search(query_vector)
                .limit(limit)
                .to_list()
            )

    def all_rows(self) -> list[dict]:
        with self._lock:
            return self.all_rows_unlocked()

    def all_rows_unlocked(self) -> list[dict]:
        if not self.has_table():
            return []
        return self.table().to_arrow().to_pylist()

    def indexed_paths(self) -> set[str]:
        with self._lock:
            return {row["path"] for row in self.all_rows_unlocked()}

    def chunk_count(self) -> int:
        with self._lock:
            return len(self.all_rows_unlocked())

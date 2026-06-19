from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchConfig:
    index_dir: Path | None = None
    batch_size: int = 8
    extra_ignore: list[str] | None = None
    warmup_embedder: bool = True
    enable_watcher: bool = True
    min_score: float = 0.32
    num_results: int = 10

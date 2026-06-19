from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = PACKAGE_DIR.parent

# Prototype-local index (CLI default when --index-dir omitted uses workspace/.arion/index)
DEFAULT_INDEX_DIR = PACKAGE_DIR / ".index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64
INDEX_CHUNK_WORKERS = 4

# Background indexer (agent middleware)
INCREMENTAL_BATCH_FILES = 8
WATCHER_DEBOUNCE_SEC = 0.5
EMBEDDER_WARMUP = True
INDEX_MAX_DEPTH = 12

# Hybrid retrieval
VECTOR_TOP_K = 60
FINAL_TOP_K = 25
MIN_HYBRID_SCORE = 0.32
VECTOR_WEIGHT = 0.62
BM25_WEIGHT = 0.38

# Deprioritize build artifacts unless BM25 is strong
DEPRIORITIZE_PATH_SUBSTRINGS = (
    "/dist/",
    "/build/",
    "package-lock.json",
    ".min.js",
    ".min.css",
    "/.index/",
    "/.arion/",
)

# Chunk sizing: general docs + code (not Cursor-identical)
WHOLE_FILE_MAX_LINES = 180
WHOLE_FILE_MAX_CHARS = 12_000
TARGET_CHUNK_CHARS = 2_800
MAX_CHUNK_CHARS = 4_500
MIN_CHUNK_CHARS = 120
CHUNK_OVERLAP_LINES = 4

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".md", ".markdown", ".txt", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ts", ".tsx",
    ".js", ".jsx", ".css", ".html", ".sql", ".sh", ".ps1", ".bat",
    ".rs", ".go", ".java", ".kt", ".xml", ".csv",
}


CJK_TRANSLATE_THRESHOLD = 0.08
MT_MODEL = "Helsinki-NLP/opus-mt-zh-en"

@dataclass(frozen=True, slots=True)
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    search_text: str
    kind: str
    content_hash: str

    @property
    def search_text_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.search_text.encode("utf-8")).hexdigest()


def resolve_index_dir(workspace: Path, index_dir: Path | None = None) -> Path:
    if index_dir is not None:
        return index_dir.resolve()
    return (workspace.resolve() / ".arion" / "index").resolve()

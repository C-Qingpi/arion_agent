from __future__ import annotations

from fastembed import TextEmbedding

from arion_agent.semantic_search.config import EMBED_BATCH_SIZE, EMBED_MODEL

_embedder: "Embedder | None" = None


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            vectors.extend(list(self._model.embed(batch)))
            done = min(i + EMBED_BATCH_SIZE, len(texts))
            if done % 512 == 0 or done == len(texts):
                print(f"  embedded {done}/{len(texts)}", flush=True)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return list(self._model.embed([query]))[0]


def embedder_loaded() -> bool:
    return _embedder is not None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

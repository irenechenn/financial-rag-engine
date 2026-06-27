from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.interfaces import Chunk, EmbeddingProvider, SearchResult


class FaissVectorStore:
    def __init__(self, index: faiss.Index | None = None, chunks: list[Chunk] | None = None) -> None:
        self.index = index
        self.chunks = chunks or []

    @classmethod
    async def from_chunks(
        cls,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        batch_size: int = 64,
    ) -> "FaissVectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(await embedding_provider.embed_texts([chunk.text for chunk in batch]))

        matrix = _as_float32_matrix(vectors)
        index = faiss.IndexFlatIP(matrix.shape[1])
        faiss.normalize_L2(matrix)
        index.add(matrix)
        return cls(index=index, chunks=chunks)

    def save(self, directory: str | Path) -> None:
        if self.index is None:
            raise ValueError("No FAISS index has been built.")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "index.faiss"))
        with (target / "chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        source = Path(directory)
        index = faiss.read_index(str(source / "index.faiss"))
        with (source / "chunks.pkl").open("rb") as file:
            chunks = pickle.load(file)
        return cls(index=index, chunks=chunks)

    async def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        candidate_multiplier: int = 10,
    ) -> list[SearchResult]:
        if self.index is None:
            raise ValueError("No FAISS index has been built.")

        filters = filters or {}
        query_vector = _as_float32_matrix([await embedding_provider.embed_query(query)])
        faiss.normalize_L2(query_vector)

        candidate_k = len(self.chunks) if filters else min(len(self.chunks), max(top_k * candidate_multiplier, top_k))
        scores, indices = self.index.search(query_vector, candidate_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if _matches_filters(chunk.metadata, filters):
                results.append(SearchResult(text=chunk.text, metadata=chunk.metadata, score=float(score)))
            if len(results) >= top_k:
                break

        return results


def _as_float32_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")
    return matrix


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(value is None or metadata.get(key) == value for key, value in filters.items())

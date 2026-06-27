from __future__ import annotations

import os
from typing import Literal

import aiohttp
from dotenv import load_dotenv

from src.interfaces import EmbeddingProvider


VoyageInputType = Literal["document", "query"]


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Voyage AI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.voyageai.com/v1/embeddings",
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-finance-2")
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is required for Voyage embeddings.")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="document")

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self._embed([query], input_type="query")
        return vectors[0]

    async def _embed(self, texts: list[str], input_type: VoyageInputType) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                response_payload = await response.json()
                if response.status >= 400:
                    message = response_payload.get("error", response_payload)
                    raise RuntimeError(f"Voyage embeddings request failed: {message}")

        return [item["embedding"] for item in response_payload["data"]]

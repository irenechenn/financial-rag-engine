from __future__ import annotations

import math

import pytest

from src.interfaces import Chunk, EmbeddingProvider
from src.tools import TranscriptSearchTool, compare_financial_metrics_tool, normalize_ticker
from src.vector_store import FaissVectorStore


class KeywordEmbeddingProvider(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("services" in normalized),
            float("margin" in normalized),
            float("cost" in normalized or "costs" in normalized),
        ]


@pytest.mark.asyncio
async def test_transcript_search_tool_returns_filtered_observations() -> None:
    chunks = [
        Chunk(
            text="Apple services revenue reached a record level.",
            metadata={"ticker": "AAPL", "year": 2023, "quarter": 1.0},
        ),
        Chunk(
            text="Tesla gross margin declined because of higher costs.",
            metadata={"ticker": "TSLA", "year": 2025, "quarter": 1.0},
        ),
        Chunk(
            text="Apple services engagement remained strong.",
            metadata={"ticker": "AAPL", "year": 2024, "quarter": 1.0},
        ),
    ]
    provider = KeywordEmbeddingProvider()
    store = await FaissVectorStore.from_chunks(chunks, provider, batch_size=2)
    tool = TranscriptSearchTool(vector_store=store, embedding_provider=provider, top_k=2)

    observation = await tool(
        query="services revenue",
        ticker="AAPL",
        year=2023,
        quarter=1,
    )

    assert observation.tool_name == "search_transcript_tool"
    assert observation.query == "services revenue"
    assert observation.filters == {"ticker": "AAPL", "year": 2023, "quarter": 1.0}
    assert len(observation.results) == 1
    assert observation.results[0]["metadata"]["ticker"] == "AAPL"
    assert observation.results[0]["metadata"]["year"] == 2023
    assert "services revenue" in observation.results[0]["text"].lower()


@pytest.mark.asyncio
async def test_transcript_search_tool_builds_query_when_model_omits_query() -> None:
    chunks = [
        Chunk(
            text="Meta advertising demand improved as engagement increased.",
            metadata={"ticker": "META", "year": 2024, "quarter": 1.0},
        ),
        Chunk(
            text="Tesla gross margin declined because of higher costs.",
            metadata={"ticker": "TSLA", "year": 2025, "quarter": 1.0},
        ),
    ]
    provider = KeywordEmbeddingProvider()
    store = await FaissVectorStore.from_chunks(chunks, provider, batch_size=2)
    tool = TranscriptSearchTool(vector_store=store, embedding_provider=provider, top_k=2)

    observation = await tool(ticker="META", year=2024, topic="advertising demand")

    assert observation.tool_name == "search_transcript_tool"
    assert observation.query == "advertising demand META 2024"
    assert observation.filters == {"ticker": "META", "year": 2024, "quarter": None}


def test_compare_financial_metrics_tool_reports_changes() -> None:
    observation = compare_financial_metrics_tool(
        metric_name="gross_margin",
        values=[
            {"label": "2024", "value": 18.0},
            {"label": "2025", "value": 16.0},
        ],
    )

    assert observation.tool_name == "compare_financial_metrics_tool"
    assert observation.output is not None
    assert observation.output["metric_name"] == "gross_margin"
    assert observation.output["absolute_change"] == -2.0
    assert math.isclose(observation.output["relative_change_pct"], -11.1111111111)
    assert observation.output["direction"] == "decrease"


def test_compare_financial_metrics_tool_requires_two_values() -> None:
    observation = compare_financial_metrics_tool(metric_name="gross_margin", values=[{"label": "2025", "value": 16.0}])

    assert observation.output is not None
    assert observation.output["error"] == "compare_financial_metrics_tool requires at least two values."


def test_compare_financial_metrics_tool_returns_error_for_non_numeric_values() -> None:
    observation = compare_financial_metrics_tool(
        metric_name="gross_margin",
        values=[
            {"label": "2024", "value": 18.0},
            {"label": "2025", "value": "management said margins were under pressure"},
        ],
    )

    assert observation.output is not None
    assert observation.output["error"] == "compare_financial_metrics_tool requires values with numeric label/value pairs."
    assert observation.output["invalid_value"]["label"] == "2025"


def test_normalize_ticker_accepts_company_aliases() -> None:
    assert normalize_ticker("Apple") == "AAPL"
    assert normalize_ticker("tesla inc.") == "TSLA"
    assert normalize_ticker("msft") == "MSFT"
    assert normalize_ticker(None) is None

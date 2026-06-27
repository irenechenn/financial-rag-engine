from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent_types import ToolObservation
from src.interfaces import EmbeddingProvider
from src.vector_store import FaissVectorStore


TICKER_ALIASES = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "tesla inc.": "TSLA",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
}


@dataclass(frozen=True)
class TranscriptSearchTool:
    vector_store: FaissVectorStore
    embedding_provider: EmbeddingProvider
    top_k: int = 3

    async def __call__(
        self,
        query: str | None = None,
        ticker: str | None = None,
        company: str | None = None,
        year: int | None = None,
        quarter: int | float | None = None,
        **extra_args: Any,
    ) -> ToolObservation:
        normalized_ticker = normalize_ticker(ticker or company)
        normalized_query = normalize_search_query(
            query=query,
            ticker=normalized_ticker,
            company=company,
            year=year,
            quarter=quarter,
            extra_args=extra_args,
        )
        filters = {
            "ticker": normalized_ticker,
            "year": year,
            "quarter": float(quarter) if quarter is not None else None,
        }
        results = await self.vector_store.search(
            query=normalized_query,
            embedding_provider=self.embedding_provider,
            top_k=self.top_k,
            filters=filters,
        )
        return ToolObservation(
            tool_name="search_transcript_tool",
            query=normalized_query,
            filters=filters,
            results=[
                {
                    "text": result.text,
                    "metadata": result.metadata,
                    "score": result.score,
                }
                for result in results
            ],
        )


def load_transcript_search_tool(
    index_path: str | Path,
    embedding_provider: EmbeddingProvider,
    top_k: int = 3,
) -> TranscriptSearchTool:
    return TranscriptSearchTool(
        vector_store=FaissVectorStore.load(index_path),
        embedding_provider=embedding_provider,
        top_k=top_k,
    )


def normalize_ticker(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    alias = TICKER_ALIASES.get(normalized.lower())
    if alias:
        return alias

    return normalized.upper()


def normalize_search_query(
    query: str | None,
    ticker: str | None,
    company: str | None,
    year: int | None,
    quarter: int | float | None,
    extra_args: dict[str, Any],
) -> str:
    if isinstance(query, str) and query.strip():
        return query.strip()

    parts: list[str] = []
    for key in ("topic", "subject", "metric", "metric_name", "keyword", "question"):
        value = extra_args.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if company:
        parts.append(company)
    if ticker:
        parts.append(ticker)
    if year is not None:
        parts.append(str(year))
    if quarter is not None:
        parts.append(f"Q{quarter}")

    if not parts:
        parts.append("earnings call financial performance")
    return " ".join(parts)


def compare_financial_metrics_tool(metric_name: str, values: list[dict[str, Any]]) -> ToolObservation:
    numeric_values: list[dict[str, float | str]] = []
    for item in values:
        try:
            numeric_values.append(
                {
                    "label": str(item["label"]),
                    "value": float(item["value"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ToolObservation(
                tool_name="compare_financial_metrics_tool",
                output={
                    "error": "compare_financial_metrics_tool requires values with numeric label/value pairs.",
                    "invalid_value": item,
                    "details": str(exc),
                },
            )

    if len(numeric_values) < 2:
        return ToolObservation(
            tool_name="compare_financial_metrics_tool",
            output={
                "error": "compare_financial_metrics_tool requires at least two values.",
                "values": numeric_values,
            },
        )

    first = numeric_values[0]
    last = numeric_values[-1]
    absolute_change = last["value"] - first["value"]
    relative_change_pct = None if first["value"] == 0 else (absolute_change / abs(first["value"])) * 100

    if absolute_change > 0:
        direction = "increase"
    elif absolute_change < 0:
        direction = "decrease"
    else:
        direction = "flat"

    return ToolObservation(
        tool_name="compare_financial_metrics_tool",
        output={
            "metric_name": metric_name,
            "values": numeric_values,
            "baseline": first,
            "comparison": last,
            "absolute_change": absolute_change,
            "relative_change_pct": relative_change_pct,
            "direction": direction,
        },
    )

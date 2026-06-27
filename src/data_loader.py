from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.interfaces import Chunk


TOKEN_PATTERN = re.compile(r"\S+")


def load_transcripts(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        for key in ("data", "records", "transcripts"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        raise ValueError("JSON object must include a list under data, records, or transcripts.")

    if not isinstance(payload, list):
        raise ValueError("Transcript dataset must be a JSON list or an object containing a list.")

    return payload


def naive_token_chunks(
    records: Iterable[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for record_idx, record in enumerate(records):
        transcript = str(record.get("transcript", "")).lstrip("\ufeff").strip()
        if not transcript:
            continue

        tokens = TOKEN_PATTERN.findall(transcript)
        for chunk_idx, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + chunk_size]
            if not window:
                continue

            metadata = {
                "ticker": record.get("ticker"),
                "year": record.get("year"),
                "quarter": record.get("quarter"),
                "record_index": record_idx,
                "chunk_index": chunk_idx,
                "chunk_strategy": "naive",
                "token_start": start,
                "token_end": start + len(window),
            }
            chunks.append(Chunk(text=" ".join(window), metadata=metadata))

            if start + chunk_size >= len(tokens):
                break

    return chunks


def load_and_chunk(
    path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    return naive_token_chunks(load_transcripts(path), chunk_size=chunk_size, overlap=overlap)

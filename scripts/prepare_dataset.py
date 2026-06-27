from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET = "glopardo/sp500-earnings-transcripts"
DEFAULT_OUTPUT = Path("data/mini_sp500_transcripts.json")
DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "META", "AMZN")


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    transcript = record.get("transcript") or record.get("content") or record.get("text") or ""
    transcript = str(transcript).lstrip("\ufeff")
    ticker = record.get("ticker") or record.get("symbol")
    year = record.get("year") or record.get("fiscal_year")
    quarter = record.get("quarter") or record.get("fiscal_quarter")

    return {
        "ticker": ticker,
        "year": int(year) if year is not None else None,
        "quarter": quarter,
        "transcript": transcript,
        **{
            key: value
            for key, value in record.items()
            if key not in {"transcript", "content", "text", "ticker", "symbol", "year", "fiscal_year", "quarter", "fiscal_quarter"}
        },
    }


def prepare_dataset(
    dataset_name: str,
    output_path: Path,
    tickers: set[str],
    start_year: int | None,
    end_year: int | None,
    limit: int | None,
) -> int:
    dataset = load_dataset(dataset_name, split="train")
    records: list[dict[str, Any]] = []

    for raw_record in dataset:
        record = normalize_record(dict(raw_record))
        ticker = str(record.get("ticker") or "").upper()
        year = record.get("year")

        if tickers and ticker not in tickers:
            continue
        if start_year is not None and (year is None or year < start_year):
            continue
        if end_year is not None and (year is None or year > end_year):
            continue
        if not record.get("transcript"):
            continue

        record["ticker"] = ticker
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local mini transcript dataset from Hugging Face.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare_dataset(
        dataset_name=args.dataset,
        output_path=args.output,
        tickers={ticker.upper() for ticker in args.tickers},
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()

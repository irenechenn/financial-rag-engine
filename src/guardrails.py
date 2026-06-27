from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.agent_types import AgentRunResult, ToolObservation


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?(?:\s?(?:m|mm|bn|b|t|million|billion|trillion))?(?=$|[^\w])",
    re.IGNORECASE,
)

UNIT_MULTIPLIERS = {
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    checked_numbers: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    unsupported_contexts: dict[str, str] = field(default_factory=dict)


def check_numeric_faithfulness(result: AgentRunResult) -> GuardrailResult:
    answer_numbers = extract_numbers(result.final_answer)
    if not answer_numbers:
        return GuardrailResult(passed=True)

    observation_text = collect_observation_text(result)
    unsupported = [number for number in answer_numbers if not number_supported_by_observations(number, observation_text)]
    issues = [f"Number '{number}' was not found in tool observations." for number in unsupported]

    return GuardrailResult(
        passed=not unsupported,
        checked_numbers=answer_numbers,
        unsupported_numbers=unsupported,
        issues=issues,
        unsupported_contexts={number: number_context(result.final_answer, number) for number in unsupported},
    )


def extract_numbers(text: str) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for match in NUMBER_PATTERN.finditer(text):
        number = normalize_number_text(match.group(0))
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def collect_observation_text(result: AgentRunResult) -> str:
    parts: list[str] = []
    for step in result.steps:
        parts.extend(observation_text_parts(step.observation))
    return "\n".join(parts)


def observation_text_parts(observation: ToolObservation) -> list[str]:
    parts: list[str] = []
    for item in observation.results:
        parts.append(str(item.get("text", "")))
        parts.append(str(item.get("metadata", "")))
    if observation.output is not None:
        parts.append(str(observation.output))
    return parts


def number_supported_by_observations(number: str, observation_text: str) -> bool:
    normalized_number = normalize_number_for_match(number)
    observation_numbers = extract_numbers(observation_text)
    normalized_observation_numbers = {normalize_number_for_match(value) for value in observation_numbers}
    if normalized_number in normalized_observation_numbers:
        return True
    return any(numbers_are_tolerant_match(number, observed) for observed in observation_numbers)


def normalize_number_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_number_for_match(value: str) -> str:
    normalized = normalize_number_text(value).lower()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" percent", "%")
    normalized = normalized.replace(" mm", " million")
    normalized = normalized.replace(" m", " million")
    normalized = normalized.replace(" bn", " billion")
    normalized = normalized.replace(" b", " billion")
    normalized = normalized.replace(" t", " trillion")
    return normalized


def numbers_are_tolerant_match(answer_number: str, observation_number: str) -> bool:
    answer = parse_number(answer_number)
    observed = parse_number(observation_number)
    if answer is None or observed is None:
        return False

    if answer["kind"] == "money" and observed["kind"] == "money":
        return values_close(
            answer["absolute_value"], observed["absolute_value"], relative_tolerance=0.03
        ) or values_close(answer["value"], observed["value"], relative_tolerance=0.03)
    if answer["kind"] == "percent":
        if observed["kind"] == "percent":
            return abs(answer["value"] - observed["value"]) <= 0.05
        return observed["kind"] == "plain" and abs(answer["value"] - observed["value"]) <= 1.0
    if answer["kind"] == observed["kind"] == "plain":
        return abs(answer["value"] - observed["value"]) <= 0.01
    return False


def parse_number(value: str) -> dict[str, float | str] | None:
    normalized = normalize_number_text(value).lower().replace(",", "")
    number_match = re.search(
        r"(?P<number>-?\d+(?:\.\d+)?)(?:\s?(?P<unit>m|mm|bn|b|t|million|billion|trillion))?",
        normalized,
    )
    if number_match is None:
        return None

    number = float(number_match.group("number"))
    unit = number_match.group("unit")
    if "%" in normalized:
        return {"kind": "percent", "value": number, "absolute_value": number}
    if "$" in normalized or unit in UNIT_MULTIPLIERS:
        multiplier = UNIT_MULTIPLIERS.get(unit or "", 1.0)
        return {"kind": "money", "value": number, "absolute_value": number * multiplier}
    return {"kind": "plain", "value": number, "absolute_value": number}


def values_close(left: float | str, right: float | str, relative_tolerance: float) -> bool:
    left_value = float(left)
    right_value = float(right)
    scale = max(abs(left_value), abs(right_value), 1.0)
    return abs(left_value - right_value) / scale <= relative_tolerance


def number_context(text: str, number: str, window: int = 80) -> str:
    index = text.find(number)
    if index == -1:
        return ""
    start = max(index - window, 0)
    end = min(index + len(number) + window, len(text))
    return " ".join(text[start:end].split())

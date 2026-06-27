from __future__ import annotations

import pytest

from src.providers import ollama_prov
from src.providers.ollama_prov import OllamaLLMProvider


class FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def post(self, url: str, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        return self.response


@pytest.mark.asyncio
async def test_ollama_provider_sends_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = FakeSession(FakeResponse(200, {"message": {"content": "  final text  "}}))
    monkeypatch.setattr(ollama_prov.aiohttp, "ClientSession", lambda: fake_session)

    provider = OllamaLLMProvider(
        model="qwen2.5:7b-instruct",
        base_url="http://localhost:11434/api/chat",
        temperature=0.2,
        num_ctx=4096,
    )

    text = await provider.generate(system_prompt="system", user_prompt="user")

    assert text == "final text"
    assert fake_session.calls == [
        {
            "url": "http://localhost:11434/api/chat",
            "json": {
                "model": "qwen2.5:7b-instruct",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 4096},
            },
        }
    ]


@pytest.mark.asyncio
async def test_ollama_provider_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = FakeSession(FakeResponse(500, {"error": "model unavailable"}))
    monkeypatch.setattr(ollama_prov.aiohttp, "ClientSession", lambda: fake_session)
    provider = OllamaLLMProvider(model="llama3.1:8b")

    with pytest.raises(RuntimeError, match="model unavailable"):
        await provider.generate(system_prompt="system", user_prompt="user")

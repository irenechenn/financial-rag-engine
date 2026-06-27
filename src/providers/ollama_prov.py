from __future__ import annotations

import os
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.interfaces import LLMProvider


class OllamaLLMProvider(LLMProvider):
    """AI Intuition / Why This Exists

    The 5080 path needs the same agentic RAG logic to run against a local model
    instead of Claude, so provider selection belongs behind the LLMProvider
    interface. This keeps ReAct planning, tool calls, guardrails, and profiling
    unchanged while letting the expensive/slow generation backend swap from a
    cloud API to a GPU-hosted Ollama server.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
        self.temperature = temperature if temperature is not None else float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
        env_num_ctx = os.getenv("OLLAMA_NUM_CTX")
        self.num_ctx = num_ctx if num_ctx is not None else int(env_num_ctx) if env_num_ctx else 8192

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as response:
                response_payload = await response.json()
                if response.status >= 400:
                    message = response_payload.get("error", response_payload)
                    raise RuntimeError(f"Ollama generation request failed: {message}")

        message = response_payload.get("message", {})
        text = message.get("content") or response_payload.get("response", "")
        return text.strip()

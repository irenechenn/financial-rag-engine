from __future__ import annotations

import os
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.interfaces import LLMProvider


class ClaudeLLMProvider(LLMProvider):
    """LLM provider backed by Anthropic Claude's Messages API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1/messages",
        max_tokens: int = 1024,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude generation.")

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                response_payload = await response.json()
                if response.status >= 400:
                    message = response_payload.get("error", response_payload)
                    raise RuntimeError(f"Claude generation request failed: {message}")

        text_blocks = [
            block.get("text", "")
            for block in response_payload.get("content", [])
            if block.get("type") == "text"
        ]
        return "\n".join(text_blocks).strip()

from __future__ import annotations

from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.interfaces import EmbeddingProvider, LLMProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        load_dotenv()
        self.model = model
        self.client = AsyncOpenAI()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        self.model = model
        self.client = AsyncOpenAI()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

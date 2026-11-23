"""The abstract interface every provider adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm.types import ChatMessage, LLMResponse, ToolSpec


class LLMError(Exception):
    """Base class for provider-neutral LLM failures."""


class LLMClient(ABC):
    """Minimal contract. Attacker, judge and toy bot depend only on this."""

    model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Run one completion. Per-call args override the client's configured defaults.

        Implementations own retries/backoff and map provider errors to `LLMError`.
        `json_mode` asks for a single JSON object in `text` (callers still validate it).
        """

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources. Default: nothing to do."""

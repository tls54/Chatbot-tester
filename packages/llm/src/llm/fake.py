"""Deterministic scripted client for tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from llm.base import LLMClient, LLMError
from llm.types import ChatMessage, LLMResponse, ToolSpec, Usage

Reply = str | LLMResponse | Callable[[list[ChatMessage]], str | LLMResponse]


class FakeLLM(LLMClient):
    """Returns scripted replies in order. Records every call in `calls`."""

    def __init__(self, replies: Sequence[Reply] = (), model: str = "fake") -> None:
        self.model = model
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def queue(self, reply: Reply) -> None:
        self._replies.append(reply)

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
        self.calls.append(
            {"messages": messages, "tools": tools, "temperature": temperature,
             "max_tokens": max_tokens, "seed": seed, "json_mode": json_mode}
        )
        if not self._replies:
            raise LLMError("FakeLLM has no scripted replies left")
        reply = self._replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, str):
            return LLMResponse(text=reply, model=self.model, usage=Usage())
        return reply

"""Groq adapter (Groq's API is OpenAI-shaped). Requires the `groq` extra."""

from __future__ import annotations

import json
import os
from typing import Any

from llm.base import LLMClient, LLMError
from llm.types import ChatMessage, LLMConfig, LLMResponse, ToolCall, ToolSpec, Usage


def _to_wire_message(m: ChatMessage) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


def _to_wire_tool(t: ToolSpec) -> dict[str, Any]:
    return {"type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Models sometimes emit malformed arguments. Keep the evidence rather than crash.
        return {"_unparsed_arguments": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self.model = config.model
        self._config = config
        if client is not None:  # injected (tests)
            self._client = client
            return
        try:
            from groq import AsyncGroq
        except ImportError as e:
            raise LLMError("Groq support needs the extra: uv sync --extra groq") from e
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set (put it in .env)")
        # The SDK already retries 429/5xx with backoff.
        self._client = AsyncGroq(
            api_key=api_key, max_retries=config.max_retries, timeout=config.timeout_s
        )

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire_message(m) for m in messages],
        }
        temperature = temperature if temperature is not None else self._config.temperature
        max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = [_to_wire_tool(t) for t in tools]
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self._config.options)

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:  # SDK error types are provider-specific; normalise
            raise LLMError(f"Groq request failed: {e}") from e

        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=msg.content or "",
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name,
                         arguments=_parse_args(tc.function.arguments))
                for tc in (msg.tool_calls or [])
            ],
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=getattr(resp, "model", self.model),
            raw=resp,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            await close()

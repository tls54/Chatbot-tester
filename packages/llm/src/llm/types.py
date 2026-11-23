"""Provider-neutral message, tool and response types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolSpec(BaseModel):
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant messages
    tool_call_id: str | None = None  # tool-result messages


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    raw: Any = Field(default=None, exclude=True)  # provider payload, not serialised

    def as_message(self) -> ChatMessage:
        """The assistant message to append to history."""
        return ChatMessage(role="assistant", content=self.text or None, tool_calls=self.tool_calls)


class LLMConfig(BaseModel):
    """Config for building a client. Provider and model are config, never code."""

    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 3
    timeout_s: float = 60.0
    options: dict[str, Any] = Field(default_factory=dict)  # provider-specific extras

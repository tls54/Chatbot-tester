import json
from types import SimpleNamespace as NS

import pytest
from llm import (
    ChatMessage,
    FakeLLM,
    LLMConfig,
    LLMError,
    ToolCall,
    ToolSpec,
    create_client,
    register,
)
from llm.groq_client import GroqClient


async def test_fake_llm_scripts_and_records():
    llm = FakeLLM(["hi", lambda msgs: f"echo:{msgs[-1].content}"])
    msgs = [ChatMessage(role="user", content="yo")]
    assert (await llm.complete(msgs)).text == "hi"
    assert (await llm.complete(msgs, json_mode=True)).text == "echo:yo"
    assert llm.calls[1]["json_mode"] is True
    with pytest.raises(LLMError):
        await llm.complete(msgs)


def test_registry_unknown_provider():
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        create_client(LLMConfig(provider="nope", model="x"))


def test_registry_custom_provider():
    register("fake", lambda cfg: FakeLLM(["ok"], model=cfg.model))
    assert create_client(LLMConfig(provider="fake", model="m")).model == "m"


class _StubGroq:
    """Mimics AsyncGroq().chat.completions.create"""

    def __init__(self, message, fail=False):
        self.calls = []
        self._message, self._fail = message, fail
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if self._fail:
            raise RuntimeError("boom")
        return NS(model="stub-model", choices=[NS(message=self._message)],
                  usage=NS(prompt_tokens=7, completion_tokens=3))


async def test_groq_maps_request_and_tool_calls():
    msg = NS(content=None, tool_calls=[
        NS(id="c1", function=NS(name="refund", arguments='{"amount": 5}')),
        NS(id="c2", function=NS(name="refund", arguments="not json")),
    ])
    stub = _StubGroq(msg)
    client = GroqClient(LLMConfig(provider="groq", model="m", temperature=0.2), client=stub)

    history = [
        ChatMessage(role="user", content="refund me"),
        ChatMessage(role="assistant", tool_calls=[ToolCall(id="x", name="refund", arguments={"a": 1})]),
        ChatMessage(role="tool", content="done", tool_call_id="x"),
    ]
    resp = await client.complete(
        history, tools=[ToolSpec(name="refund", description="d")], seed=1, json_mode=True
    )

    sent = stub.calls[0]
    assert sent["model"] == "m" and sent["temperature"] == 0.2 and sent["seed"] == 1
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["tools"][0]["function"]["name"] == "refund"
    assert json.loads(sent["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert sent["messages"][2]["tool_call_id"] == "x"

    assert resp.tool_calls[0].arguments == {"amount": 5}
    assert resp.tool_calls[1].arguments == {"_unparsed_arguments": "not json"}
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.total_tokens) == (7, 3, 10)
    assert resp.model == "stub-model"


async def test_groq_normalises_errors():
    client = GroqClient(LLMConfig(provider="groq", model="m"),
                        client=_StubGroq(NS(content="", tool_calls=None), fail=True))
    with pytest.raises(LLMError, match="boom"):
        await client.complete([ChatMessage(role="user", content="x")])

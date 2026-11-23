"""Opt-in smoke test against the real API: uv run pytest -m live"""

import os

import pytest
from dotenv import load_dotenv
from llm import ChatMessage, ToolSpec, config_from_env, create_client

load_dotenv()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


async def test_live_plain_completion():
    client = create_client(config_from_env())
    resp = await client.complete(
        [ChatMessage(role="user", content="Reply with exactly the word: pong")],
        temperature=0, max_tokens=20,
    )
    assert "pong" in resp.text.lower()
    assert resp.usage.total_tokens > 0
    await client.aclose()


async def test_live_tool_call():
    client = create_client(config_from_env())
    tool = ToolSpec(
        name="get_weather", description="Get the weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]},
    )
    resp = await client.complete(
        [ChatMessage(role="user", content="What's the weather in Paris? Use the tool.")],
        tools=[tool], temperature=0,
    )
    assert resp.tool_calls and resp.tool_calls[0].name == "get_weather"
    await client.aclose()


async def test_live_json_mode():
    import json

    client = create_client(config_from_env())
    resp = await client.complete(
        [ChatMessage(role="user", content='Return JSON like {"ok": true}')], json_mode=True
    )
    assert isinstance(json.loads(resp.text), dict)
    await client.aclose()

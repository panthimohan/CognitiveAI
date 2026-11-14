import asyncio
import json

import httpx
import pytest

from backend.models.factory import create_provider, register_provider
from backend.models.fake_provider import FakeProvider
from backend.models.ollama_provider import OllamaProvider, strip_reasoning
from backend.models.providers import ModelRequest
from config.settings import get_settings


def run(coro):
    return asyncio.run(coro)


def provider_with(handler, model="llama3.2"):
    return OllamaProvider(
        model=model,
        base_url="http://test:11434",
        transport=httpx.MockTransport(handler)
    )


def chat_reply(content):
    return httpx.Response(200, json={"message": {"content": content}})


# ---------- OllamaProvider.generate ----------

def test_generate_success_and_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return chat_reply("Paris")

    provider = provider_with(handler)
    response = run(provider.generate(ModelRequest(
        prompt="Capital of France?",
        system_prompt="Be brief.",
        temperature=0.2,
        max_tokens=50
    )))

    assert response.success is True
    assert response.content == "Paris"
    assert response.provider == "local"
    assert response.model == "llama3.2"

    assert seen["url"] == "http://test:11434/api/chat"
    assert seen["body"]["model"] == "llama3.2"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"] == {
        "temperature": 0.2, "num_predict": 50
    }
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
    ]


def test_generate_without_system_prompt():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return chat_reply("ok")

    run(provider_with(handler).generate(ModelRequest(prompt="hi")))

    assert seen["body"]["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_generate_strips_deepseek_thinking():
    provider = provider_with(
        lambda r: chat_reply("<think>hmm, let me see</think>The answer is 4.")
    )

    response = run(provider.generate(ModelRequest(prompt="2+2?")))

    assert response.content == "The answer is 4."


def test_generate_empty_reply_is_failure():
    provider = provider_with(lambda r: chat_reply("   "))

    response = run(provider.generate(ModelRequest(prompt="hi")))

    assert response.success is False
    assert "empty" in response.error.lower()


def test_generate_reports_server_not_running():
    def handler(request):
        raise httpx.ConnectError("refused")

    response = run(provider_with(handler).generate(
        ModelRequest(prompt="hi")
    ))

    assert response.success is False
    assert "ollama serve" in response.error


def test_generate_reports_missing_model():
    provider = provider_with(
        lambda r: httpx.Response(404, json={"error": "not found"}),
        model="qwen2.5-coder:7b"
    )

    response = run(provider.generate(ModelRequest(prompt="hi")))

    assert response.success is False
    assert "ollama pull qwen2.5-coder:7b" in response.error


def test_generate_reports_other_http_errors():
    provider = provider_with(lambda r: httpx.Response(500))

    response = run(provider.generate(ModelRequest(prompt="hi")))

    assert response.success is False
    assert "500" in response.error


def test_strip_reasoning_keeps_text_without_tags():
    assert strip_reasoning("plain answer") == "plain answer"


# ---------- OllamaProvider.is_available ----------

def tags_handler(*names):
    return lambda request: httpx.Response(
        200, json={"models": [{"name": n} for n in names]}
    )


def test_available_when_model_installed():
    provider = provider_with(tags_handler("llama3.2:latest"))

    assert provider.is_available() is True


def test_available_matches_exact_tag():
    provider = provider_with(
        tags_handler("qwen2.5-coder:7b"), model="qwen2.5-coder:7b"
    )

    assert provider.is_available() is True


def test_unavailable_when_model_missing():
    provider = provider_with(
        tags_handler("llama3.2:latest"), model="deepseek-r1:7b"
    )

    assert provider.is_available() is False


def test_unavailable_when_server_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert provider_with(handler).is_available() is False


# ---------- FakeProvider ----------

def test_fake_provider_follows_script_and_records_requests():
    fake = FakeProvider(responses=["first", "second"])

    first = run(fake.generate(ModelRequest(prompt="a")))
    second = run(fake.generate(ModelRequest(prompt="b")))
    third = run(fake.generate(ModelRequest(prompt="c")))

    assert [first.content, second.content, third.content] == [
        "first", "second", "second"
    ]
    assert [r.prompt for r in fake.requests] == ["a", "b", "c"]


# ---------- factory ----------

def test_factory_creates_local_provider_with_model():
    provider = create_provider("local", "qwen2.5-coder:7b")

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5-coder:7b"
    assert create_provider("ollama").get_provider_name() == "local"


def test_factory_creates_openai_provider():
    provider = create_provider("openai", "gpt-4.1-mini")

    assert provider.get_provider_name() == "openai"
    assert provider.model == "gpt-4.1-mini"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="not available"):
        create_provider("anthropic")


def test_factory_accepts_registered_provider():
    fake = FakeProvider(name="fake")
    register_provider("fake", lambda model: fake)

    assert create_provider("fake") is fake


# ---------- settings ----------

def test_default_local_model_per_task_type():
    settings = get_settings()

    assert settings.local_model_for("coding") == "qwen2.5-coder:7b"
    assert settings.local_model_for("reasoning") == "deepseek-r1:7b"
    assert settings.local_model_for("conversation") == "llama3.2"
    assert settings.local_model_for("research") == "llama3.2"
    assert settings.local_model_for("anything-else") == "llama3.2"


def test_settings_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_CODING", "my-coder:1b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://other:1234/")

    settings = get_settings()

    assert settings.local_model_for("coding") == "my-coder:1b"
    assert settings.ollama_base_url == "http://other:1234"

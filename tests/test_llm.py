from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from conftest import FakeLLM
from mdsearch.llm import CachedLLM, LLMError, OpenAICompatLLM, SQLiteCache, complete_json, extract_json
from mdsearch.llm.cache import make_key


# --------------------------------------------------------------------------- cache


def test_cache_hits_and_persists(tmp_path):
    inner = FakeLLM(lambda p, s, sc: f"reply:{p}")
    llm = CachedLLM(inner, SQLiteCache(tmp_path / "c.sqlite"))
    assert llm.complete("hi") == "reply:hi"
    assert llm.complete("hi") == "reply:hi"
    assert len(inner.calls) == 1 and llm.hits == 1 and llm.misses == 1

    # a new wrapper over the same file reuses the stored answer
    llm2 = CachedLLM(FakeLLM(lambda p, s, sc: "different"), SQLiteCache(tmp_path / "c.sqlite"))
    assert llm2.complete("hi") == "reply:hi"


def test_cache_key_depends_on_prompt_system_schema():
    k = lambda **kw: make_key("p", "m", kw.get("system"), kw.get("prompt", "x"), kw.get("schema"), 0.0)
    assert len({k(), k(prompt="y"), k(system="s"), k(schema={"a": 1})}) == 4


def test_empty_replies_are_not_cached(tmp_path):
    inner = FakeLLM(lambda p, s, sc: "  ")
    llm = CachedLLM(inner, SQLiteCache(tmp_path / "c.sqlite"))
    llm.complete("hi")
    llm.complete("hi")
    assert len(inner.calls) == 2


def test_complete_many_preserves_order(tmp_path):
    llm = CachedLLM(FakeLLM(lambda p, s, sc: p.upper()), SQLiteCache(tmp_path / "c.sqlite"))
    assert llm.complete_many(["a", "b", "c"]) == ["A", "B", "C"]


# ----------------------------------------------------------------------- structured


class Out(BaseModel):
    x: int


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"x": 1}\n```') == {"x": 1}
    assert extract_json('Sure! {"x": 2} hope that helps') == {"x": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_json_retries_then_succeeds():
    replies = iter(["not json", '{"x": "oops"}', '{"x": 3}'])
    llm = FakeLLM(lambda p, s, sc: next(replies))
    assert complete_json(llm, "q", Out, retries=2).x == 3
    assert "previous reply was not valid" in llm.calls[1][0]


def test_complete_json_gives_up():
    llm = FakeLLM(lambda p, s, sc: "garbage")
    with pytest.raises(LLMError):
        complete_json(llm, "q", Out, retries=1)
    assert len(llm.calls) == 2


# ------------------------------------------------------------------------ openai_compat


def _client(handler):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=handler)))


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_openai_compat_native_schema():
    seen = []

    def handler(**kw):
        seen.append(kw)
        return _resp('{"x": 1}')

    llm = OpenAICompatLLM(base_url="http://x", model="m", client=_client(handler))
    assert llm.complete("p", schema={"type": "object"}) == '{"x": 1}'
    assert "response_format" in seen[0] and llm.native_schema is True


def test_openai_compat_falls_back_when_response_format_rejected():
    seen = []
    req = httpx.Request("POST", "http://x")

    def handler(**kw):
        seen.append(kw)
        if "response_format" in kw:
            raise openai.BadRequestError(
                "unsupported", response=httpx.Response(400, request=req), body=None
            )
        return _resp('{"x": 2}')

    llm = OpenAICompatLLM(base_url="http://x", model="m", client=_client(handler))
    assert llm.complete("p", system="sys", schema={"type": "object"}) == '{"x": 2}'
    assert llm.native_schema is False
    assert "JSON schema" in seen[-1]["messages"][0]["content"]
    assert seen[-1]["messages"][0]["content"].startswith("sys")

    # once known to be unsupported, later calls skip the failing attempt
    n = len(seen)
    llm.complete("p2", schema={"type": "object"})
    assert len(seen) == n + 1


def test_openai_compat_none_content_becomes_empty_string():
    llm = OpenAICompatLLM(base_url="http://x", model="m", client=_client(lambda **kw: _resp(None)))
    assert llm.complete("p") == ""


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(LLMError, match="SOME_KEY"):
        OpenAICompatLLM(base_url="http://x", model="m", api_key_env="SOME_KEY")

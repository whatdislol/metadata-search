from __future__ import annotations

import pytest

from mdsearch.llm import CachedLLM, LLMError, get_llm, register_kind

CFG = """
default: a
cache_path: {cache}
providers:
  a:
    kind: openai_compat
    base_url: http://a/v1
    model: model-a
    api_key_env: TEST_KEY_A
  b:
    kind: openai_compat
    base_url: http://b/v1
    model: model-b
    api_key_env: TEST_KEY_A
  c:
    kind: custom
    model: custom-model
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "llm.yaml"
    p.write_text(CFG.format(cache=tmp_path / "cache.sqlite"))
    monkeypatch.setenv("TEST_KEY_A", "secret")
    monkeypatch.delenv("MDSEARCH_LLM_PROVIDER", raising=False)
    return p


def test_default_provider_and_cache_wrapping(cfg):
    llm = get_llm(config_path=cfg)
    assert isinstance(llm, CachedLLM) and llm.provider == "a" and llm.model == "model-a"
    assert get_llm(config_path=cfg, cache=False).model == "model-a"


def test_explicit_name_and_env_override(cfg, monkeypatch):
    assert get_llm("b", config_path=cfg).model == "model-b"
    monkeypatch.setenv("MDSEARCH_LLM_PROVIDER", "b")
    assert get_llm("a", config_path=cfg).model == "model-b"


def test_unknown_provider(cfg):
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        get_llm("nope", config_path=cfg)


def test_register_custom_kind(cfg):
    class Custom:
        def __init__(self, provider, model):
            self.provider, self.model = provider, model

        def complete(self, prompt, *, system=None, schema=None):
            return "custom"

        def complete_many(self, prompts, *, system=None, schema=None):
            return ["custom"] * len(prompts)

    register_kind("custom", Custom)
    llm = get_llm("c", config_path=cfg, cache=False)
    assert llm.complete("x") == "custom" and llm.provider == "c"

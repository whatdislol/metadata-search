"""The only LLM surface the rest of the code depends on."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised for configuration problems or when an LLM reply cannot be used."""


@runtime_checkable
class LLM(Protocol):
    """Minimal text-in/text-out interface.

    `schema` is an optional JSON schema. When given, the reply must be a JSON
    document matching it; how that is enforced (native structured output or a
    prompted fallback) is the implementation's business, not the caller's.
    """

    provider: str  # name from configs/llm.yaml, recorded in experiment outputs
    model: str

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None
    ) -> str: ...

    def complete_many(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> list[str]: ...

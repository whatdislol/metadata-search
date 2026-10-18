"""Client for any OpenAI-compatible chat endpoint (vLLM, OpenAI, Ollama, OpenRouter, ...)."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import BadRequestError, OpenAI

from .base import LLMError

log = logging.getLogger(__name__)

_JSON_ONLY = (
    "Reply with ONLY a single JSON object that matches this JSON schema. "
    "No prose and no code fences.\n{schema}"
)


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        provider: str = "openai_compat",
        max_workers: int = 8,
        timeout: float = 300.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        client: Any = None,  # injectable for tests
    ) -> None:
        if client is None:
            key = os.environ.get(api_key_env or "")
            if not key:
                raise LLMError(
                    f"API key environment variable {api_key_env!r} is not set (see .env.example)"
                )
            client = OpenAI(
                base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries
            )
        self.client = client
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        # None = not yet known, True/False once we have seen how the endpoint behaves.
        self.native_schema: bool | None = None

    # ------------------------------------------------------------------ public

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        if schema is None:
            return self._chat(self._messages(prompt, system))

        if self.native_schema is not False:
            try:
                out = self._chat(
                    self._messages(prompt, system),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema},
                    },
                )
                self.native_schema = True
                return out
            except BadRequestError as e:
                log.warning(
                    "Endpoint rejected response_format (%s); falling back to prompted JSON", e
                )
                self.native_schema = False

        prefix = f"{system}\n\n" if system else ""
        sys_msg = prefix + _JSON_ONLY.format(schema=json.dumps(schema))
        return self._chat(self._messages(prompt, sys_msg))

    def complete_many(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> list[str]:
        # The server batches concurrent requests, so threads give a real speed-up.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self.complete(p, system=system, schema=schema), prompts))

    # ----------------------------------------------------------------- private

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, **kwargs
        )
        # Reasoning models can return reasoning separately; we only use the final content.
        return r.choices[0].message.content or ""

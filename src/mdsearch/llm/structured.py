"""Structured (JSON) output on top of the plain LLM interface."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from .base import LLM, LLMError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse a JSON document from a model reply, tolerating code fences and stray prose."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def parse_reply(text: str, model_cls: type[T]) -> T:
    return model_cls.model_validate(extract_json(text))


def complete_json(
    llm: LLM,
    prompt: str,
    model_cls: type[T],
    *,
    system: str | None = None,
    retries: int = 2,
) -> T:
    """Ask for JSON matching `model_cls`, validate it, and retry with the error fed back."""
    schema = model_cls.model_json_schema()
    attempt_prompt = prompt
    last_error: Exception | None = None
    for _ in range(retries + 1):
        text = llm.complete(attempt_prompt, system=system, schema=schema)
        try:
            return parse_reply(text, model_cls)
        except ValueError as e:  # JSONDecodeError and pydantic ValidationError both subclass it
            last_error = e
            attempt_prompt = (
                f"{prompt}\n\nYour previous reply was not valid ({str(e)[:300]}). "
                "Reply with ONLY the JSON object."
            )
    raise LLMError(f"No valid JSON after {retries + 1} attempts: {last_error}")

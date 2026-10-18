"""Metadata encoded as plain text appended to a base representation (project variant 1)."""

from __future__ import annotations

from ..data.schema import Product
from .base import DocRepresentation, truncate


def _fmt(v: float) -> str:
    return f"{v:g}"


class TemplatedMetadata:
    def __init__(
        self,
        base: DocRepresentation,
        fields: tuple[str, ...] = ("brand", "price", "rating", "min_age"),
        max_chars: int = 2200,
    ) -> None:
        self.base = base
        self.fields = fields
        self.max_chars = max_chars
        self.name = f"templated[{base.name}]"

    def render(self, p: Product) -> str:
        parts: list[str] = []
        for f in self.fields:
            v = getattr(p, f, None)
            if v is None:
                continue
            parts.append(f"{f}: {_fmt(v) if isinstance(v, (int, float)) else v}")
        meta = " | ".join(parts)
        # metadata goes first so truncation never cuts it off
        return truncate(f"{meta} | {self.base.render(p)}" if meta else self.base.render(p), self.max_chars)

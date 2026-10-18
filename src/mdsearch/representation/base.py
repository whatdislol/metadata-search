"""How a product (text + metadata) becomes the string a model sees.

This is the main research variable for the project: new ways of encoding metadata
are new implementations of this interface, nothing else in the pipeline changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..data.schema import Product


@runtime_checkable
class DocRepresentation(Protocol):
    name: str

    def render(self, p: Product) -> str: ...


def truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars].rsplit(" ", 1)[0]

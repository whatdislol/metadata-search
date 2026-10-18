from __future__ import annotations

from .base import DocRepresentation
from .templated import TemplatedMetadata
from .text import TitleDescription, TitleReviews


def get_representation(name: str, max_chars: int = 2000) -> DocRepresentation:
    """'title_description' | 'title_reviews' | 'templated:<base>' (metadata as text)."""
    if name == "title_description":
        return TitleDescription(max_chars)
    if name == "title_reviews":
        return TitleReviews(max_chars)
    if name.startswith("templated:"):
        return TemplatedMetadata(get_representation(name.split(":", 1)[1], max_chars))
    raise ValueError(f"Unknown representation {name!r}")


__all__ = [
    "DocRepresentation",
    "TemplatedMetadata",
    "TitleDescription",
    "TitleReviews",
    "get_representation",
]

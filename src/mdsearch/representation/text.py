"""Text-only representations (no metadata): the baselines."""

from __future__ import annotations

from ..data.schema import Product
from .base import truncate


class TitleDescription:
    name = "title_description"

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def render(self, p: Product) -> str:
        return truncate(f"{p.title}. {p.description}".strip(), self.max_chars)


class TitleReviews:
    """Title plus review text; QAM matches the semantic part of a query against reviews."""

    name = "title_reviews"

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def render(self, p: Product) -> str:
        body = " ".join(p.reviews) if p.reviews else p.description
        return truncate(f"{p.title}. {body}".strip(), self.max_chars)

"""Structured form of a search query: metadata constraints plus the leftover semantic text."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Filters(BaseModel):
    """Hard metadata constraints. None means "not constrained"."""

    brand: str | None = Field(None, description="Brand or manufacturer name as written")
    price_min: float | None = Field(None, description="Minimum price, number only")
    price_max: float | None = Field(None, description="Maximum price, number only")
    rating_min: float | None = Field(None, description="Minimum average star rating (0-5)")
    age_min: float | None = Field(None, description="Youngest age (years) of the intended child")
    age_max: float | None = Field(None, description="Oldest age (years) of the intended child")

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def is_empty(self) -> bool:
        return not self.active()


class ParsedQuery(BaseModel):
    semantic_text: str = Field(
        description="What the product is and the qualities wanted, with filter terms removed"
    )
    filters: Filters = Field(default_factory=Filters)

    @field_validator("filters", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> Any:
        return {} if v is None else v

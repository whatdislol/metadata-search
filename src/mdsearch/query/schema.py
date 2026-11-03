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


# Roles a generated tag can play. The tag TEXT is free-form; only this role label is from a fixed list.
TAG_KINDS = (
    "product_type", "brand", "color", "material", "size", "audience", "style", "feature",
    "price", "quality", "occasion", "other",
)
# Roles that describe an attribute of a product (not what the product is); the default set a hard
# filter is applied to when "attribute only" tags are requested.
ATTRIBUTE_KINDS = ("brand", "color", "material", "size", "audience", "price", "feature")


class Tag(BaseModel):
    """A free-form requirement the LLM wrote itself; matched to products by text, not by field."""

    text: str = Field(description="Short phrase describing one attribute products should have")
    kind: str = Field("", description="Role of the tag, one of TAG_KINDS (e.g. 'price', 'audience').")

    @field_validator("text", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return " ".join(str(v).split()) if v is not None else ""

    @field_validator("kind", mode="before")
    @classmethod
    def _clean_kind(cls, v: Any) -> Any:
        return "_".join(str(v).lower().split()) if v is not None else ""


class ParsedQuery(BaseModel):
    semantic_text: str = Field(
        description="What the product is and the qualities wanted, with filter terms removed"
    )
    filters: Filters = Field(default_factory=Filters)
    tags: list[Tag] = Field(default_factory=list)  # filled by TagParser; empty for fixed-tag QAM
    # LLM-written expansion of the query (synonyms, product type, typical features); only when requested
    expansion: str = ""

    @field_validator("filters", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> Any:
        return {} if v is None else v

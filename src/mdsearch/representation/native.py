"""The corpus's own product metadata written into the product text: `Metadata: category=...; brand=...`."""

from __future__ import annotations

from typing import Sequence

from ..data.schema import Product
from .base import DocRepresentation
from .tagged import place_metadata


def field_key(spec: str) -> str:
    """Name written in the text for a field spec: the first alias, lowercase, spaces as underscores."""
    return spec.split("|")[0].strip().lower().replace(" ", "_")


class NativeMetadata:
    """Adds `Metadata: key=value; ...` built from the product's own category path and attribute tables.

    fields: attribute names to write; "Brand|Manufacturer" means the first one found (case-insensitive) in
        Product.extra["attrs"], then ["info"]; the text uses the first alias as the key.
    category_levels: how many of the LAST levels of the category path to write (0 = none).
    max_value_chars: longer attribute values are cut.
    position: "before" (after the title, survives the encoder's length limit) or "after" (end of the text).
    Products with nothing to write are rendered as the base does."""

    def __init__(
        self,
        base: DocRepresentation,
        fields: Sequence[str] = (),
        category_levels: int = 0,
        max_value_chars: int = 80,
        position: str = "before",
    ) -> None:
        if position not in ("before", "after"):
            raise ValueError("position must be 'before' or 'after'")
        if not fields and category_levels <= 0:
            raise ValueError("NativeMetadata needs fields and/or category_levels")
        self.base, self.fields, self.position = base, [f for f in fields if f.strip()], position
        self.category_levels, self.max_value_chars = max(int(category_levels), 0), int(max_value_chars)
        self.name = f"native[{base.name}]"

    def items(self, p: Product) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.category_levels and p.category:
            levels = [c.strip() for c in p.category.split(" > ") if c.strip()]
            if levels:
                out.append(("category", " > ".join(levels[-self.category_levels:])))
        tables = [
            {k.lower(): v for k, v in (p.extra.get(name) or {}).items()} for name in ("attrs", "info")
        ]
        seen: set[str] = set()
        for spec in self.fields:
            for alias in spec.split("|"):
                value = next((t[alias.strip().lower()] for t in tables if t.get(alias.strip().lower())), None)
                if value:
                    value = " ".join(str(value).split())[: self.max_value_chars]
                    if value.lower() not in seen:  # the same value under two names (Brand = Manufacturer) is written once
                        seen.add(value.lower())
                        out.append((field_key(spec), value))
                    break
        return out

    def meta_text(self, p: Product) -> str:
        items = self.items(p)
        return "Metadata: " + "; ".join(f"{k}={v}" for k, v in items) if items else ""

    def render(self, p: Product) -> str:
        return place_metadata(self.base.render(p), p, self.meta_text(p), self.position)

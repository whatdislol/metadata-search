"""Product tags written into the product text (document-side metadata): `Metadata: brand=X; color=Y`."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..data.schema import Product
from ..query.schema import Tag
from .base import DocRepresentation


class TaggedMetadata:
    """Adds `Metadata: kind=value; kind=value` to a base representation.

    position "before" (default) puts it right after the title, so it survives the encoder's length limit;
    "after" appends it to the end of the text. Products without tags are rendered as the base does, so a
    tag file that covers only part of the catalog still works. `kinds` limits which tag roles are written."""

    def __init__(
        self,
        base: DocRepresentation,
        tags: Mapping[str, Sequence[Tag]],
        position: str = "before",
        kinds: Sequence[str] | None = None,
    ) -> None:
        if position not in ("before", "after"):
            raise ValueError("position must be 'before' or 'after'")
        self.base, self.tags, self.position = base, tags, position
        self.kinds = frozenset(kinds) if kinds is not None else None
        self.name = f"tagged[{base.name}]"

    def meta_text(self, p: Product) -> str:
        ts = [t for t in self.tags.get(p.id, ()) if t.text and (self.kinds is None or t.kind in self.kinds)]
        return "Metadata: " + "; ".join(f"{t.kind or 'tag'}={t.text}" for t in ts) if ts else ""

    def render(self, p: Product) -> str:
        return place_metadata(self.base.render(p), p, self.meta_text(p), self.position)


def place_metadata(base: str, p: Product, meta: str, position: str) -> str:
    """Puts a `Metadata: ...` line into the product text: right after the title ("before") or at the end ("after")."""
    if not meta:
        return base
    if position == "after":
        return f"{base} {meta}"
    head = f"{p.title}."
    if base.startswith(head):
        return f"{head} {meta}. {base[len(head):].strip()}".strip()
    return f"{meta}. {base}"

"""Dataset-independent product record. Every loader (toys, WANDS, ESCI) produces these."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Product:
    id: str
    title: str
    description: str = ""
    brand: str | None = None
    price: float | None = None
    rating: float | None = None  # average star rating, 0-5
    n_reviews: int | None = None
    min_age: float | None = None  # recommended minimum age in years
    category: str | None = None
    reviews: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Product":
        return cls(**d)


def save_products(path: str | Path, products: Iterable[Product]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_products(path: str | Path) -> list[Product]:
    with open(path, encoding="utf-8") as f:
        return [Product.from_dict(json.loads(line)) for line in f if line.strip()]

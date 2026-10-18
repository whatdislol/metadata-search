"""A product collection with fast array views for vectorised filtering and indexing."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .data.schema import Product
from .filtering.constraints import normalize_brand


class Catalog:
    def __init__(self, products: Iterable[Product]) -> None:
        self.products: list[Product] = list(products)
        self.ids: list[str] = [p.id for p in self.products]
        self._index = {pid: i for i, pid in enumerate(self.ids)}
        if len(self._index) != len(self.ids):
            raise ValueError("Duplicate product ids in catalog")

        def arr(attr: str) -> np.ndarray:
            return np.array(
                [np.nan if getattr(p, attr) is None else getattr(p, attr) for p in self.products],
                dtype=float,
            )

        self.price = arr("price")
        self.rating = arr("rating")
        self.min_age = arr("min_age")
        self.brand_norm: list[str] = [normalize_brand(p.brand) for p in self.products]

    def __len__(self) -> int:
        return len(self.products)

    def index(self, product_id: str) -> int:
        return self._index[product_id]

    def get(self, product_id: str) -> Product:
        return self.products[self._index[product_id]]

    def texts(self, representation) -> list[str]:
        """Render every product with a DocRepresentation."""
        return [representation.render(p) for p in self.products]

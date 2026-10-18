"""Step 2 of QAM: hard-filter the catalog on parsed metadata constraints (vectorised)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..query.schema import Filters
from .constraints import age_upper, normalize_brand

if TYPE_CHECKING:  # catalog imports this package, so import it for typing only
    from ..catalog import Catalog


@dataclass
class FilterStep:
    name: str
    value: Any
    removed: int  # products removed by this step, given the steps before it


@dataclass
class FilterReport:
    total: int
    remaining: int
    steps: list[FilterStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "remaining": self.remaining,
            "steps": [{"name": s.name, "value": s.value, "removed": s.removed} for s in self.steps],
        }


class MetadataFilter:
    def __init__(
        self,
        catalog: Catalog,
        *,
        missing: str = "exclude",
        price_tol: float = 0.0,
        rating_tol: float = 0.0,
    ) -> None:
        if missing not in ("exclude", "keep"):
            raise ValueError("missing must be 'exclude' or 'keep'")
        self.catalog = catalog
        self.missing = missing
        self.price_tol = price_tol
        self.rating_tol = rating_tol

    def apply(self, f: Filters) -> tuple[np.ndarray, FilterReport]:
        c = self.catalog
        n = len(c)
        mask = np.ones(n, dtype=bool)
        steps: list[FilterStep] = []

        def step(name: str, value: Any, known: np.ndarray, ok: np.ndarray) -> None:
            nonlocal mask
            keep = (ok & known) if self.missing == "exclude" else (ok | ~known)
            before = int(mask.sum())
            mask = mask & keep
            steps.append(FilterStep(name, value, before - int(mask.sum())))

        if f.brand is not None:
            qb = normalize_brand(f.brand)
            known = np.array([bool(b) for b in c.brand_norm])
            ok = np.array([bool(qb) and bool(b) and (qb in b or b in qb) for b in c.brand_norm])
            step("brand", f.brand, known, ok)

        price_known = ~np.isnan(c.price)
        with np.errstate(invalid="ignore"):
            if f.price_min is not None:
                step(
                    "price_min", f.price_min, price_known,
                    c.price >= f.price_min * (1 - self.price_tol),
                )
            if f.price_max is not None:
                step(
                    "price_max", f.price_max, price_known,
                    c.price <= f.price_max * (1 + self.price_tol),
                )
            if f.rating_min is not None:
                step(
                    "rating_min", f.rating_min, ~np.isnan(c.rating),
                    c.rating >= f.rating_min * (1 - self.rating_tol),
                )
            hi = age_upper(f)
            if hi is not None:
                step("age", hi, ~np.isnan(c.min_age), c.min_age <= hi)

        return mask, FilterReport(total=n, remaining=int(mask.sum()), steps=steps)

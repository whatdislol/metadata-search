from __future__ import annotations

from typing import Protocol, Sequence


class Scorer(Protocol):
    def score(self, query: str, docs: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderScorer:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/msmarco-MiniLM-L12-en-de-v1",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RuntimeError("Install the ML extras: pip install -e '.[ml]'") from e
            self._model = CrossEncoder(
                self.name, max_length=self.max_length, device=self.device
            )
        return self._model

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        model = self._load()
        out = model.predict([(query, d) for d in docs], batch_size=self.batch_size)
        return [float(x) for x in out]

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


def _check_device(device: str | None) -> None:
    """Fails loudly when the config asks for a GPU that PyTorch cannot see (instead of running on the CPU)."""
    import torch

    if device and str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"encoders.device is {device!r} but PyTorch {torch.__version__} cannot use a GPU "
            "(torch.cuda.is_available() is False). Install the CUDA build of PyTorch in this environment."
        )


def _report_device(name: str, model: Any) -> None:
    dev = getattr(model, "device", None) or getattr(getattr(model, "model", None), "device", None)
    print(f"{name} runs on {dev}", flush=True)


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        """kind is 'query' or 'document' (some models use different prefixes)."""
        ...


class SentenceTransformerEncoder:
    """Lazy wrapper so importing the package does not require torch."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length  # tokens kept per text; lower = much faster on CPU
        # embeddings depend on the truncation length, so it is part of the on-disk cache key
        self.cache_key = f"{model_name}@{max_seq_length}" if max_seq_length else model_name
        self._model: Any = None  # SentenceTransformer, loaded lazily
        self._prefixes = (
            {"query": "search_query: ", "document": "search_document: "}
            if "nomic" in model_name
            else {"query": "", "document": ""}
        )

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("Install the ML extras: pip install -e '.[ml]'") from e
            _check_device(self.device)
            self._model = SentenceTransformer(
                self.name, trust_remote_code=True, device=self.device
            )
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
            _report_device(self.name, self._model)
        return self._model

    def encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        model = self._load()
        prefix = self._prefixes[kind]
        batch: list[str] = [prefix + str(t) for t in texts]
        show_bar: bool = len(batch) > 64
        return np.asarray(
            model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_bar,
            ),
            dtype=np.float32,
        )


def _fingerprint(ids: Sequence[str], texts: Sequence[str], encoder_name: str) -> str:
    h = hashlib.sha256(encoder_name.encode())
    for i, t in zip(ids, texts):
        h.update(i.encode())
        h.update(b"\0")
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()


class DenseRetriever:
    def __init__(self, ids: Sequence[str], embeddings: np.ndarray, encoder: Encoder) -> None:
        self.ids = list(ids)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.maximum(norms, 1e-12)
        self.encoder = encoder

    @classmethod
    def build(
        cls,
        ids: Sequence[str],
        texts: Sequence[str],
        encoder: Encoder,
        cache_path: str | Path | None = None,
    ) -> "DenseRetriever":
        """Embed the documents, reusing an on-disk copy if it matches ids, texts and encoder."""
        fp = _fingerprint(ids, texts, getattr(encoder, "cache_key", getattr(encoder, "name", type(encoder).__name__)))
        if cache_path is not None and Path(cache_path).exists():
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["fp"]) == fp:
                    return cls(ids, z["emb"], encoder)
        emb = np.asarray(encoder.encode(list(texts), kind="document"), dtype=np.float32)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, emb=emb, fp=np.array(fp))
        return cls(ids, emb, encoder)

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray:
        q = np.asarray(self.encoder.encode([query], kind="query"), dtype=np.float32)[0]
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return self.embeddings @ q

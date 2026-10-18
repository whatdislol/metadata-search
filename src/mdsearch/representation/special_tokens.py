"""STEP 5 (not implemented): metadata as special tokens / value-conditioned embeddings.

Planned variants for the project description, each as a DocRepresentation or a model wrapper:
  - categorical fields -> added special tokens, e.g. "[RATING_4_5]" (needs tokenizer + model
    embedding resize, hence fine-tuning with transformers/PyTorch, run on a GPU node);
  - numeric fields -> bucketed tokens or a value-conditioned prefix token
    (cf. Value-Aware Numerical Representations, arXiv 2601.09706);
  - an LLM-written document that states the metadata in prose (uses mdsearch.llm).
"""

from __future__ import annotations

from ..data.schema import Product


class SpecialTokenRepresentation:
    name = "special_tokens"

    def render(self, p: Product) -> str:
        raise NotImplementedError("Step 5: implement together with a tokenizer/model that knows the tokens")

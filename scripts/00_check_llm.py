"""Step 1: check the LLM endpoint works, including structured output and parallelism.

    python scripts/00_check_llm.py [--provider uofg]
"""

from __future__ import annotations

import argparse
import time

from mdsearch.llm import complete_json, get_llm
from mdsearch.query.schema import ParsedQuery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="provider name from configs/llm.yaml")
    args = ap.parse_args()

    llm = get_llm(args.provider, cache=False)
    print(f"provider={llm.provider} model={llm.model}")

    t = time.time()
    print("1) plain completion (first call may be slow if the model is sleeping)...")
    out = llm.complete("Reply with the single word: pong")
    print(f"   -> {out.strip()!r}  ({time.time() - t:.1f}s)")

    print("2) structured output...")
    parsed = complete_json(llm, "a LEGO set for a 6 year old, under 40 pounds", ParsedQuery)
    print(f"   -> {parsed.model_dump()}")
    native = getattr(llm, "native_schema", None)
    print(f"   native response_format supported: {native} (False = prompted-JSON fallback in use)")

    print("3) 8 parallel requests...")
    t = time.time()
    outs = llm.complete_many([f"Reply with the number {i}." for i in range(8)])
    print(f"   -> {[o.strip()[:12] for o in outs]}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()

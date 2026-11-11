"""Step 6: let an LLM write metadata tags for every product (document-side metadata for the `_meta` methods).

    python scripts/06_tag_products.py --config trec_product

Reads the products written by script 01 (dataset.processed) and appends {"id", "tags"} lines to
representation.metadata.file. Products that already have tags are skipped, so an interrupted run resumes
and a bigger run only pays for the missing products; --limit N tags N random untagged products per call.
Settings (config section product_tagging): max_tags (8), max_chars of description shown to the LLM (1500),
batch (200). Replies are cached by the LLM cache. Afterwards, methods with a `_meta` suffix
(rerank_meta, hybrid_meta, ...) read the tags; add them to eval.methods and run script 04.
"""

from __future__ import annotations

import argparse
from collections import Counter

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.schema import load_products
from mdsearch.llm import get_llm
from mdsearch.product_tags import ProductTagger, load_product_tags, tag_catalog
from mdsearch.runlog import record_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="trec_product")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--limit", type=int, default=None, help="tag only N random untagged products in this call")
    ap.add_argument("--show", type=int, default=8, help="how many tagged products to print at the end")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    meta = (cfg.get("representation") or {}).get("metadata") or {}
    if not meta.get("file"):
        raise SystemExit("set representation.metadata.file in the config (where the tags are stored)")
    out = resolve_path(meta["file"])
    pt = cfg.get("product_tagging") or {}
    processed = resolve_path(cfg["dataset"]["processed"])
    if not processed.exists():
        raise SystemExit(f"{processed} not found. Run scripts/01_prepare.py --config {args.config} first.")

    with record_run("06_tag_products", config=args.config, args=vars(args)) as run:
        run.set_filtering({}, print_it=False)
        products = load_products(processed)
        llm = get_llm(args.provider)
        tagger = ProductTagger(llm, max_tags=pt.get("max_tags", 8), max_chars=pt.get("max_chars", 1500))
        print(f"{len(products)} products in {processed.name}; tags go to {out}")
        stats = tag_catalog(products, tagger, out, limit=args.limit, batch=pt.get("batch", 200))
        tags = load_product_tags(out)
        have = sum(p.id in tags for p in products)
        print(f"\ncoverage: {have} of {len(products)} products have tags ({have / len(products):.1%})   "
              f"unusable replies this call: {stats['failures']}")
        roles = Counter(t.kind or "(none)" for ts in tags.values() for t in ts)
        sizes = [len(ts) for ts in tags.values()]
        print(f"tags per product: mean {sum(sizes) / max(len(sizes), 1):.1f}, none for {sizes.count(0)}")
        print("roles: " + ", ".join(f"{k} {n}" for k, n in roles.most_common()))
        for p in products:
            if args.show > 0 and p.id in tags and tags[p.id]:
                args.show -= 1
                print(f"  {p.id}: {p.title[:70]!r} -> " + "; ".join(f"{t.kind}={t.text}" for t in tags[p.id]))


if __name__ == "__main__":
    main()

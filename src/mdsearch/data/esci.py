"""STEP 5 (not implemented): ESCI / TREC Product Search loader.

Same target interface as data/wands.py. ESCI products carry brand and colour; the TREC 2023
collection (via ir_datasets, `pip install -e '.[bench]'`) adds a metadata-enriched version.
Labels: Exact=1.0, Substitute=0.1, Complement=0.01, Irrelevant=0 (the KDD Cup gains).
Data: https://github.com/amazon-science/esci-data (download manually).
"""

from __future__ import annotations


def load_esci(*args, **kwargs):
    raise NotImplementedError("Step 5")


def prepare(cfg, *, limit=None, source=None):
    """Hook used by scripts/01_prepare.py. To implement it, read the raw ESCI files, build
    `queries` and `qrels` dicts and a record iterator, then call data.evalset.build_qrels_dataset
    (see data/trec_product.py for a complete example) and save the products and the EvalSet."""
    raise NotImplementedError("ESCI loader not written yet; see data/trec_product.py for the pattern")

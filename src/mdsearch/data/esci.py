"""STEP 5 (not implemented): ESCI / TREC Product Search loader.

Same target interface as data/wands.py. ESCI products carry brand and colour; the TREC 2023
collection (via ir_datasets, `pip install -e '.[bench]'`) adds a metadata-enriched version.
Labels: Exact=1.0, Substitute=0.1, Complement=0.01, Irrelevant=0 (the KDD Cup gains).
Data: https://github.com/amazon-science/esci-data (download manually).
"""

from __future__ import annotations


def load_esci(*args, **kwargs):
    raise NotImplementedError("Step 5")

"""Build the per-head label spaces from the CIM-10 referential.

Writes ``data/valid_labels_all_{dp,das,mdp}.json`` — the label lists referenced by each
head in ``configs/config.yml``. Mirrors ``notebooks/valid_labels.ipynb``.

This script only needs pandas (no Spark), so it also runs in the portable venv::

    python -m scripts.build_labels
"""

from __future__ import annotations

import json

from ._common import REFERENTIAL, load_cim10_referential

DATA_DIR = REFERENTIAL.parent


def main() -> None:
    dp_codes, das_codes, mdp_codes = load_cim10_referential(REFERENTIAL)
    # eds.doc_classifier reads label files as JSON (or .txt), never as pickles.
    for name, codes in (("dp", dp_codes), ("das", das_codes), ("mdp", mdp_codes)):
        path = DATA_DIR / f"valid_labels_all_{name}.json"
        path.write_text(json.dumps(codes, ensure_ascii=False, indent=0))
    print(f"dp={len(dp_codes)} das={len(das_codes)} mdp={len(mdp_codes)}")


if __name__ == "__main__":
    main()

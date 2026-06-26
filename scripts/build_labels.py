"""Build the per-head label spaces from the CIM-10 referential.

Writes ``data/valid_labels_all_{dp,das,mdp}.pkl`` — the label lists referenced by each
head in ``configs/config.yml``. Mirrors ``notebooks/valid_labels.ipynb``.

This script only needs pandas (no Spark), so it also runs in the portable venv::

    python -m scripts.build_labels
"""

from __future__ import annotations

import pandas as pd

from ._common import REFERENTIAL, load_cim10_referential

DATA_DIR = REFERENTIAL.parent


def main() -> None:
    dp_codes, das_codes, mdp_codes = load_cim10_referential(REFERENTIAL)
    pd.to_pickle(dp_codes, DATA_DIR / "valid_labels_all_dp.pkl")
    pd.to_pickle(das_codes, DATA_DIR / "valid_labels_all_das.pkl")
    pd.to_pickle(mdp_codes, DATA_DIR / "valid_labels_all_mdp.pkl")
    print(f"dp={len(dp_codes)} das={len(das_codes)} mdp={len(mdp_codes)}")


if __name__ == "__main__":
    main()

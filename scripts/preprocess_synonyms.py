"""Build the synthetic synonym parquets.

Produces two independent synthetic streams, each supervising a single head of
``eds.doc_classifier`` (see ``configs/config.yml``):

- ``syn`` — one synonym per document, labelled with ``dp`` (single-label).
- ``syn_das`` — ``k`` synonyms concatenated per document (``k`` in
  ``[DAS_K_MIN, DAS_K_MAX]``), labelled with the list of their codes in ``das``
  (multi-label). Teaches the ``das`` head to flag several diagnoses at once.

The two streams are kept separate on purpose: a document must never carry both
``dp`` and ``das`` gold (the text would be ambiguous), and ``eds.doc_classifier``
requires a batch to supervise a fixed set of heads. Codes are sampled uniformly
and independently — no co-occurrence realism yet; the low ``loss_scale`` (0.1)
cushions the resulting prose gap.

Mirrors ``notebooks/preprocess_synonyms.ipynb``. AP-HP only: runs on the Spark cluster.
See ``docs/preprocessing.md``.

Usage::

    python -m scripts.preprocess_synonyms
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import REFERENTIAL, get_spark, load_cim10_referential, save_to_parquet

USER = "ldedieu"
SCRATCH = Path(f"/data/scratch/{USER}/CU2/encoder_baseline")
HDFS_BASE = "CU2/encoder_baseline"

# The synonym source pickle shipped in the repo (data/).
SYNONYMS_PICKLE = Path(__file__).resolve().parent.parent / "data" / "synonyms.pkl"

# --- DAS synthetic-document generation ---
RANDOM_SEED = 42
N_DAS_DOCS = 150_000  # number of multi-synonym documents to generate
DAS_K_MIN, DAS_K_MAX = 1, 8  # inclusive range for the number of codes per document
DAS_SEP = "\n"  # separator joining the synonyms of one document


def build_dp_frame(syn: pd.DataFrame, dp_codes: list[str]) -> pd.DataFrame:
    """One synonym per document, labelled with its ``dp`` code (single-label)."""
    dp_set = set(dp_codes)
    dp = syn[syn["code"].isin(dp_set)]
    return dp[["libelle", "code"]].rename(
        columns={"libelle": "note_text", "code": "dp"}
    )


def build_das_frame(syn: pd.DataFrame, das_codes: list[str]) -> pd.DataFrame:
    """``k`` synonyms per document, labelled with their code list (multi-label).

    For each of ``N_DAS_DOCS`` documents, sample ``k`` (uniform in
    ``[DAS_K_MIN, DAS_K_MAX]``) distinct codes, pick one random synonym per code,
    and join the texts. Returns a frame with ``note_text`` (str) and ``das`` (list).
    """
    das_set = set(das_codes)
    das_syn = syn[syn["code"].isin(das_set)]
    syn_by_code = {
        code: group["libelle"].to_numpy()
        for code, group in das_syn.groupby("code", sort=False)
    }
    codes = np.array(list(syn_by_code))

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for _ in range(N_DAS_DOCS):
        k = int(rng.integers(DAS_K_MIN, DAS_K_MAX + 1))
        chosen = rng.choice(codes, size=k, replace=False)
        texts = [str(rng.choice(syn_by_code[code])) for code in chosen]
        rows.append({"note_text": DAS_SEP.join(texts), "das": [str(c) for c in chosen]})
    return pd.DataFrame(rows)


def main() -> None:
    dp_codes, das_codes, _ = load_cim10_referential(REFERENTIAL)
    syn = pd.read_pickle(SYNONYMS_PICKLE)

    spark = get_spark()

    dp_frame = build_dp_frame(syn, dp_codes)
    save_to_parquet(
        spark.createDataFrame(dp_frame),
        hdfs_path=f"{HDFS_BASE}/syn",
        local_path=SCRATCH / "syn",
        columns=["note_text", "dp"],
        num_partitions=121,
    )

    das_frame = build_das_frame(syn, das_codes)
    save_to_parquet(
        spark.createDataFrame(das_frame),
        hdfs_path=f"{HDFS_BASE}/syn_das",
        local_path=SCRATCH / "syn_das",
        columns=["note_text", "das"],
        num_partitions=121,
    )


if __name__ == "__main__":
    main()

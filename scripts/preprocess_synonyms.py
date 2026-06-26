"""Build the synthetic synonym parquet (dp only).

Mirrors ``notebooks/preprocess_synonyms.ipynb``. AP-HP only: runs on the Spark cluster.
See ``docs/preprocessing.md``.

Usage::

    python -m scripts.preprocess_synonyms
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._common import REFERENTIAL, get_spark, load_cim10_referential, save_to_parquet

USER = "ldedieu"
SCRATCH = Path(f"/data/scratch/{USER}/CU2/encoder_baseline")
HDFS_BASE = "CU2/encoder_baseline"

# The synonym source pickle lives outside this repo (built by the binary_model project).
SYNONYMS_PICKLE = Path("../binary_model/data/V2_avril_2026/phase1_df_pos.pkl")


def main() -> None:
    dp_codes, _, _ = load_cim10_referential(REFERENTIAL)
    dp_set = set(dp_codes)

    syn = pd.read_pickle(SYNONYMS_PICKLE)
    syn = syn[syn["code"].isin(dp_set)]

    spark = get_spark()
    df = (
        spark.createDataFrame(syn)
        .withColumnRenamed("libelle", "note_text")
        .withColumnRenamed("code", "dp")
    )

    save_to_parquet(
        df,
        hdfs_path=f"{HDFS_BASE}/syn",
        local_path=SCRATCH / "syn",
        columns=["note_text", "dp"],
        num_partitions=121,
    )


if __name__ == "__main__":
    main()

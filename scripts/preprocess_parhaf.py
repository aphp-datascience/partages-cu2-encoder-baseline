"""Build the PARHAF validation parquet (dp only) from the Hugging Face dataset.

Mirrors ``notebooks/preprocess_PARHAF.ipynb``. AP-HP only: runs on the Spark cluster
and needs the ``HF_TOKEN`` env var. See ``docs/preprocessing.md``.

Usage::

    export HF_TOKEN=hf_xxx
    python -m scripts.preprocess_parhaf
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from pyspark.sql import functions as F

from ._common import get_spark, save_to_parquet

USER = "ldedieu"
SCRATCH = Path(f"/data/scratch/{USER}/CU2/encoder_baseline")
HDFS_BASE = "CU2/encoder_baseline"
CODING_POOL = "CU 2 - ICD-10 coding"


def main() -> None:
    ds = load_dataset("HealthDataHub/PARHAF", token=os.environ["HF_TOKEN"])
    coding_patients = [p for p in ds["train"] if p["pool"] == CODING_POOL]
    print(f"{len(coding_patients)} coding patients")

    spark = get_spark()
    df = spark.createDataFrame(coding_patients).select(
        F.explode("documents.text").alias("note_text"),
        F.col("structured_abstract.primary_diagnosis.code").getItem(0).alias("dp"),
    )
    df = df.filter((F.col("dp").isNotNull()) & (F.col("dp") != ""))

    save_to_parquet(
        df,
        hdfs_path=f"{HDFS_BASE}/PARHAF",
        local_path=SCRATCH / "PARHAF",
        columns=["note_text", "dp"],
        num_partitions=1,
    )


if __name__ == "__main__":
    main()

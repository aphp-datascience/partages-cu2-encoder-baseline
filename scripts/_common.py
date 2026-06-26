"""Shared helpers for the CU2 preprocessing scripts.

These scripts build the parquet datasets and label artifacts consumed by
``configs/config.yml``. They run on the AP-HP Spark / YARN / HDFS cluster and need
``pyspark`` + ``edstoolbox`` installed separately (see ``docs/preprocessing.md``).

Run them from the repo root as modules, e.g.::

    python -m scripts.preprocess_mistral --split train
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# The CIM-10 referential shipped in the repo (data/).
REFERENTIAL = (
    Path(__file__).resolve().parent.parent / "data" / "Referentiel_CIM-10-20250108.csv"
)


def get_spark(app_name: str = "cu2") -> SparkSession:
    """Open a YARN-backed Spark session (AP-HP cluster)."""
    return (
        SparkSession.builder.enableHiveSupport()
        .appName(app_name)
        .master("yarn")
        .config("spark.submit.deployMode", "client")
        .getOrCreate()
    )


def load_cim10_referential(
    path: str | Path = REFERENTIAL,
) -> tuple[list[str], list[str], list[str]]:
    """Load valid ICD-10 (CIM-10) codes from the AP-HP referential CSV.

    Returns ``(dp_codes, das_codes, mdp_codes)``, each in referential order:

    - ``dp_codes``: usable as principal diagnosis  (``code MCO/HAD == 0``)
    - ``das_codes``: usable as associated diagnosis (``code MCO/HAD in {0, 1, 2}``)
    - ``mdp_codes``: the ``Z...`` codes among ``dp_codes``
    """
    codes = pd.read_csv(path, sep=";", header=1, dtype=str)
    codes = codes[["code", "libellé long", "code MCO/HAD"]].rename(
        columns={"libellé long": "definition", "code MCO/HAD": "mco_had"}
    )
    codes["code"] = codes["code"].astype(str).str.strip()
    codes["mco_had"] = pd.to_numeric(codes["mco_had"], errors="coerce")

    dp_codes = codes.loc[codes["mco_had"].isin([0]), "code"].tolist()
    das_codes = codes.loc[codes["mco_had"].isin([0, 1, 2]), "code"].tolist()
    mdp_codes = [c for c in dp_codes if c.startswith("Z")]
    return dp_codes, das_codes, mdp_codes


def save_to_parquet(
    df: DataFrame,
    hdfs_path: str,
    local_path: str | Path,
    columns: Sequence[str],
    num_partitions: int,
) -> None:
    """Add a ``note_id``, write ``columns`` to HDFS parquet, then fetch to scratch.

    Mirrors the notebooks: write to HDFS, then ``hdfs dfs -copyToLocal`` into the local
    directory the training config reads from.
    """
    df = df.withColumn("note_id", F.monotonically_increasing_id())
    (
        df.select("note_id", *columns)
        .repartition(num_partitions)
        .write.mode("overwrite")
        .parquet(hdfs_path)
    )

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hdfs", "dfs", "-copyToLocal", "-f", f"{hdfs_path}/*", f"{local_path}/"],
        check=True,
    )

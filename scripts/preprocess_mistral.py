"""Build the MISTRAL real-data parquet datasets (train / test split).

Mirrors ``notebooks/prepocess_MISTRAL.ipynb`` and ``prepocess_MISTRAL_test.ipynb``.
AP-HP only: runs on the Spark / YARN / HDFS cluster. See ``docs/preprocessing.md``.

Usage::

    python -m scripts.preprocess_mistral --split train
    python -m scripts.preprocess_mistral --split test
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from ._common import REFERENTIAL, get_spark, load_cim10_referential, save_to_parquet

USER = "ldedieu"
SCRATCH = Path(f"/data/scratch/{USER}/CU2/encoder_baseline")
HDFS_BASE = "CU2/encoder_baseline"

SPLITS = {
    "train": {
        "csv": SCRATCH / "MISTRAL" / "train_format2.csv",
        "name": "MISTRAL/v1_das_mdp",
        "num_partitions": 9,
    },
    "test": {
        "csv": SCRATCH / "MISTRAL" / "test_format2.csv",
        "name": "MISTRAL/test_v1_das",
        "num_partitions": 1,
    },
}


def extract_all_diagnostics(cell_value: object) -> pd.Series:
    """Parse one note's annotation dict into its dp / dr / das codes."""
    res: dict[str, object] = {"dp": None, "dr": None, "das": []}
    if pd.isna(cell_value):
        return pd.Series(res)
    if isinstance(cell_value, str):
        try:
            cell_value = ast.literal_eval(cell_value)
        except (ValueError, SyntaxError):
            return pd.Series(res)
    if isinstance(cell_value, dict):
        for key, details in cell_value.items():
            if isinstance(details, dict):
                pos = details.get("position")
                if pos == "DP":
                    res["dp"] = key
                elif pos == "DR":
                    res["dr"] = key
                elif pos == "DAS":
                    res["das"].append(key)  # type: ignore[union-attr]
    return pd.Series(res)


def build_pandas(csv_path: Path) -> pd.DataFrame:
    """Clean one MISTRAL CSV into a (clinical_note, dp, das, mdp) DataFrame."""
    df = pd.read_csv(csv_path, on_bad_lines="skip").dropna()
    df = df[["clinical_note", "annot_diagnostics_v2"]]

    df[["dp", "dr", "das"]] = df["annot_diagnostics_v2"].apply(extract_all_diagnostics)
    df["das"] = df["das"].apply(lambda x: list(set(x)) if isinstance(x, list) else x)
    df = df.drop("annot_diagnostics_v2", axis=1)

    # mdp defaults to Z769; when a DR exists, dp <- dr and mdp <- the original dp.
    df["mdp"] = "Z769"
    mask = df["dr"].notna()
    df.loc[mask, "mdp"] = df.loc[mask, "dp"]
    df.loc[mask, "dp"] = df.loc[mask, "dr"]
    df = df.drop(columns=["dr"])

    df = df[df["dp"].apply(lambda x: isinstance(x, str))]

    # Keep only codes valid for their position in the CIM-10 referential.
    dp_codes, das_codes, _ = load_cim10_referential(REFERENTIAL)
    dp_set, das_set = set(dp_codes), set(das_codes)
    df = df[df["dp"].isin(dp_set)]
    df = df[df["mdp"].isin(dp_set)]
    df["das"] = df["das"].apply(
        lambda x: [c for c in x if c in das_set] if isinstance(x, list) else x
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), default="train")
    args = parser.parse_args()
    cfg = SPLITS[args.split]

    pdf = build_pandas(cfg["csv"])

    spark = get_spark()
    df = spark.createDataFrame(pdf).withColumnRenamed("clinical_note", "note_text")
    df = df.filter((F.col("dp").isNotNull()) & (F.col("dp") != ""))

    save_to_parquet(
        df,
        hdfs_path=f"{HDFS_BASE}/{cfg['name']}",
        local_path=SCRATCH / cfg["name"],
        columns=["note_text", "dp", "das", "mdp"],
        num_partitions=cfg["num_partitions"],
    )


if __name__ == "__main__":
    main()

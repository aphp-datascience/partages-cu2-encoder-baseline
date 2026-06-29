# Data preprocessing

The notebooks in [`notebooks/`](../notebooks/) build the parquet datasets and label artifacts
that the training config consumes. **They are AP-HP-specific**: they run on the AP-HP
Spark / YARN / HDFS cluster and read internal data sources. External partners do not need to run
them — instead, produce parquet files that follow the schema in [data.md](data.md) and the
label `.pkl` artifacts described there.

This page documents what each notebook does and the order to run them in.

## AP-HP environment

The notebooks need **PySpark** and the AP-HP-internal **`edstoolbox`** (to launch a
Spark-enabled Jupyter kernel). These were removed from `pyproject.toml` so that external users
can install the project from public PyPI. On the AP-HP cluster, install them into the venv
separately:

```bash
# AP-HP only — requires access to the AP-HP GitLab package index
uv pip install "pyspark==2.4.3.post5" \
    --index-url https://gitlab.eds.aphp.fr/api/v4/projects/2378/packages/pypi/simple
uv pip install "git+https://gitlab.eds.aphp.fr/entrep-t-de-donn-es-de-sant/eds-tools/datasciencetools/eds-toolbox.git"
```

Each Spark notebook opens a YARN session (`master("yarn")`), writes its output as parquet to
HDFS, then copies it to local scratch with `hdfs dfs -copyToLocal`.

## Run order

`build_dataset.ipynb` unions the per-source outputs, so it must run last. The label artifacts
(`valid_labels.ipynb`) are independent of the data and can be built any time.

```
valid_labels.ipynb ─┐
prepocess_MISTRAL.ipynb ──────┐
prepocess_MISTRAL_test.ipynb ─┤
preprocess_PARHAF.ipynb ──────┼──→ build_dataset.ipynb
preprocess_synonyms.ipynb ────┘
```

## Python scripts (`scripts/`)

The notebooks have been factored into clean, runnable Python scripts under `scripts/`, sharing
the Spark session, CIM-10 referential loading and parquet-saving logic in `scripts/_common.py`.
Run them from the repo root as modules:

| Command                                          | Mirrors notebook                | Output                                   |
|--------------------------------------------------|---------------------------------|------------------------------------------|
| `python -m scripts.build_labels`                 | `valid_labels.ipynb`            | `data/valid_labels_all_{dp,das,mdp}.pkl` |
| `python -m scripts.preprocess_mistral --split train` | `prepocess_MISTRAL.ipynb`   | MISTRAL train parquet                    |
| `python -m scripts.preprocess_mistral --split test`  | `prepocess_MISTRAL_test.ipynb` | MISTRAL test parquet                    |
| `python -m scripts.preprocess_parhaf`            | `preprocess_PARHAF.ipynb`       | PARHAF parquet (`HF_TOKEN` required)     |
| `python -m scripts.preprocess_synonyms`          | `preprocess_synonyms.ipynb`     | synthetic parquet                        |

Absolute paths (user, scratch dirs) are constants at the top of each script —
edit them for your environment. `build_labels` only needs pandas (no Spark) and reproduces the
committed label artifacts exactly. The `build_dataset.ipynb` notebook has not been scripted yet
(it is work-in-progress).

## Notebooks

### `valid_labels.ipynb`
Builds the per-head label space from `data/Referentiel_CIM-10-20250108.csv`:
- `valid_labels_all_dp.pkl` — codes with `code MCO/HAD == 0`
- `valid_labels_all_das.pkl` — codes with `code MCO/HAD ∈ {0, 1, 2}`
- `valid_labels_all_mdp.pkl` — the `Z…` codes among the valid `dp` codes

### `prepocess_MISTRAL.ipynb`
Builds the real training set from the MISTRAL CSV:
1. Reads the CSV, keeps `clinical_note` + the diagnosis annotation column.
2. Extracts `dp` / `dr` / `das` from the per-note annotation dict.
3. Applies the `mdp` rule: `mdp = 'Z769'` by default; if a `dr` exists, swap so `dp ← dr` and
   `mdp ← original dp`.
4. Filters codes against the referential (`dp`/`mdp` must be valid DP codes; `das` kept to valid
   DAS codes).
5. Writes parquet (`note_id`, `note_text`, `dp`, `das`, `mdp`).

### `prepocess_MISTRAL_test.ipynb`
Same logic as above for the MISTRAL **test** split (used as the validation set in the current
config — `vars.val_path`).

### `preprocess_PARHAF.ipynb`
Builds an alternative validation set from the `HealthDataHub/PARHAF` Hugging Face dataset:
keeps the *CU 2 - ICD-10 coding* pool, explodes each patient's documents into rows, and takes
the primary diagnosis code as `dp`. Writes parquet (`note_id`, `note_text`, `dp`).

> **Authentication:** the dataset requires a Hugging Face token. The notebook reads it from the
> `HF_TOKEN` environment variable — set it before launching Jupyter:
> ```bash
> export HF_TOKEN=hf_xxx
> ```
> Never commit a token to the repo.

### `preprocess_synonyms.ipynb`
Builds the synthetic dataset: takes the valid `dp` codes from the referential, loads the
synonym/augmentation pickle (`data/synonyms.pkl`), keeps only rows whose code is a valid `dp`,
and writes parquet (`note_id`, `note_text`, `dp`). This feeds the down-weighted
`doc_classifier_syn` head.

### `build_dataset.ipynb`
Unions the per-source datasets, reconciles the label sets across train / val / synthetic,
computes label frequencies, and builds the `label2id` / `id2label` mappings and the
`valid_labels_dp.pkl` / `label_freq_dict_dp.pkl` artifacts.

> ⚠️ This notebook is the most in-flux of the set (it carries work-in-progress cells for the
> multi-head `das` / `mdp` extension and several commented-out branches). Read it carefully
> before relying on a specific output path.

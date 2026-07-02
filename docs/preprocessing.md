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
committed label artifacts exactly. `build_dataset.ipynb` (dataset-driven label spaces and class
weights) has not been scripted — it stays a notebook.

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
Builds the synthetic datasets from the synonym/augmentation pickle (`data/synonyms.pkl`). Writes
**two** parquets, each supervising a single head of the down-weighted `doc_classifier_syn`
component:

- `syn` (`note_id`, `note_text`, `dp`) — one synonym per document, kept only if its code is a
  valid `dp`; trains the single-label `dp` head.
- `syn_das` (`note_id`, `note_text`, `das`) — `k` synonyms (uniform in `1..8`, valid `das`
  codes) concatenated per document, labelled with their code list; trains the multi-label `das`
  head to flag several diagnoses at once. Codes are sampled uniformly, no co-occurrence realism.

The two streams are kept separate so no document carries both `dp` and `das` gold (which would
make the text ambiguous) and so each training batch supervises a fixed set of heads.

### `build_dataset.ipynb`
Derives the per-head label artifacts from the **datasets** (whereas `valid_labels.ipynb` derives
them from the referential):

- **Label spaces** `valid_labels_{dp,das,mdp}.pkl` — the union of codes seen across the datasets,
  so every observed code is scoreable. `dp` = train ∪ synthetic ∪ test; `das` / `mdp` = train ∪
  test (only MISTRAL carries `das` / `mdp`).
- **Class-weight frequencies** `label_freq_dict_{dp,das,mdp}.pkl` — document frequency per code,
  counted on the **real train** set (MISTRAL) only; synthetic and validation data do not
  contribute.
- The `label2id` / `id2label` mappings (`label2id.pkl`, `id2label.pkl`).

It also prints label-coverage diagnostics (which codes are unique to a single dataset). It writes
no parquet — `prepocess_MISTRAL*` and `preprocess_synonyms` already materialise the datasets the
config reads. Run it last, after the per-source notebooks.

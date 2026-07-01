# Data format & labels

This page describes the inputs the training config consumes: the parquet datasets, the label
artifacts, and the CIM-10 referential used to validate codes.

## Datasets

Training reads three parquet datasets, all in **OMOP** layout (read with EDS-NLP's
`converter: omop`). Paths are set in the `vars` block of `configs/config.yml`.

| Role        | `vars` key   | Source                | Labels present     | Notes                              |
|-------------|--------------|-----------------------|--------------------|------------------------------------|
| Train       | `train_path` | MISTRAL generated  | `dp`, `das`, `mdp` | Primary supervision signal         |
| Synthetic   | `synth_path` | Synonym augmentation  | `dp` only          | Down-weighted (`loss_scales` 0.1)  |
| Validation  | `val_path`   | MISTRAL test split    | `dp`, `das`, `mdp` | PARHAF is an alternative `dp`-only val set |

### Availability

None of these datasets are AP-HP-private — they are usable by any PARTAGES partner:

- **MISTRAL** (train + test split) is shared on a **Google Drive accessible to all PARTAGES
  project members**. It is model-generated data, not real patient records.
- **Synthetic** synonyms are **shipped in this repo** (`data/synonyms.pkl`, pushed to GitHub) and
  derived from public referentials, so every partner can use them. `scripts.preprocess_synonyms`
  turns that pickle into the training parquet (`note_text` = the synonym label, `dp` = the code).
  The pipeline that *builds* `synonyms.pkl` itself lives in a separate repo
  (`icd-synonym-qualification`, source `phase3_df_pos`) that is not yet public.
- **PARHAF** (optional `dp`-only validation set) is the public `HealthDataHub/PARHAF` dataset on
  the Hugging Face Hub. It is **gated**: request access and set `HF_TOKEN` (see
  [preprocessing.md](preprocessing.md)).

If you want to train on **your own clinical notes** instead, see
[*Using your own data*](training.md#using-your-own-data) in the training guide.

### Parquet schema

Each row is one clinical note:

| Column      | Type            | Description                                                        |
|-------------|-----------------|-------------------------------------------------------------------|
| `note_id`   | int / string    | Unique note identifier                                            |
| `note_text` | string          | Raw clinical note text                                           |
| `dp`        | string          | *Diagnostic principal* — a single ICD-10 code                    |
| `das`       | list[string]    | *Diagnostics associés* — zero or more ICD-10 codes |
| `mdp`       | string          | *Mode de prise en charge* — a single ICD-10 code (`Z769` by default) |

The OMOP converter maps `note_text` to the document text and each entry in `doc_attributes`
(`["dp", "das", "mdp"]`) to a document-level attribute read off `doc._.<attr>`.

## Label definitions

- **`dp` (diagnostic principal)** — the main diagnosis that motivated the stay. One code per
  note → single-label classification.
- **`das` (diagnostics associés)** — comorbidities / associated conditions. A
  note can have several → multi-label classification.
- **`mdp` (mode de prise en charge)** — defaults to `Z769`. When a *diagnostic relié* (`dr`) is present in the source
  annotation, the preprocessing swaps the codes: `dp` becomes the `dr`, and `mdp` keeps the
  original principal diagnosis. So `mdp` is `Z769` for the vast majority of notes and carries
  the original DP only for the DP/DR cases.

The synthetic dataset only carries `dp` (synonyms of a single code), which is why the synthetic
classifier head (`doc_classifier_syn`) is trained on `dp` alone.

## CIM-10 referential & valid codes

`data/Referentiel_CIM-10-20250108.csv` is the official ICD-10 (CIM-10) reference. It is used to
restrict predictions to codes that are valid for a given position, via its `code MCO/HAD`
column:

| `code MCO/HAD` value | Usable as          |
|----------------------|--------------------|
| `0`                  | `dp` **and** `das` |
| `1` or `2`           | `das` only         |

`mdp` valid codes are the `Z…` codes among the valid `dp` codes.

## Label artifacts (`data/*.pkl`)

The config loads the per-head label space from pickle files. There are two families, both
documented in [preprocessing.md](preprocessing.md):

**Referential-driven** (`valid_labels_all_*`, from `valid_labels.ipynb` / `scripts.build_labels`)
— the full set of codes valid for each position, independent of the data. Used by the multi-head
`configs/config.yml`:

| File                        | Content                                                  | Used by config head |
|-----------------------------|----------------------------------------------------------|---------------------|
| `valid_labels_all_dp.pkl`   | All valid `dp` codes (referential `MCO/HAD == 0`)        | `dp`                |
| `valid_labels_all_das.pkl`  | All valid `das` codes (referential `MCO/HAD ∈ {0,1,2}`) | `das`               |
| `valid_labels_all_mdp.pkl`  | All valid `mdp` codes (`Z…` codes among valid `dp`)     | `mdp`               |

**Dataset-driven** (from `build_dataset.ipynb`) — the codes actually seen in the data, plus class
weights. The label spaces are the union over datasets (`dp` = train ∪ synthetic ∪ test; `das` /
`mdp` = train ∪ test); the frequencies are counted on the **real train** set only. The single-head
`configs/config_dp.yml` uses the `dp` pair:

| File                          | Content                                                       | Used by              |
|-------------------------------|---------------------------------------------------------------|----------------------|
| `valid_labels_{dp,das,mdp}.pkl` | Per-head label space restricted to observed codes           | `config_dp.yml` (`dp`) |
| `label_freq_dict_{dp,das,mdp}.pkl` | `{code: document_count}` from train — class weights      | `config_dp.yml` (`dp`) |
| `label2id.pkl` / `id2label.pkl` | `{head: {label ↔ id}}` mappings for all three heads        | (analysis / tooling) |

To **change the target labels**, regenerate the relevant `.pkl` files and update the corresponding
`labels:` (and `class_weights:`) paths under each head in `configs/config.yml` /
`configs/config_dp.yml`.

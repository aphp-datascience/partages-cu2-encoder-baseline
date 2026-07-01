# Training

Everything is driven by a [confit](https://github.com/aphp/confit) config consumed by EDS-NLP's
training CLI. There is no training script in this repo.

## Two configurations

| Config                                              | Heads              | Use it as                          |
|-----------------------------------------------------|--------------------|------------------------------------|
| [`configs/config.yml`](../configs/config.yml)       | `dp`, `das`, `mdp` | **The default** multi-head setup   |
| [`configs/config_dp.yml`](../configs/config_dp.yml) | `dp` only          | A simpler complementary baseline   |

The two are structurally identical apart from the number of classification heads, so everything
below applies to both — just point `--config` at the one you want.

## Launching a run

The default multi-head config:

```bash
uv run python -m edsnlp.train --config configs/config.yml --seed 42
```

### The single-head (DP only) baseline

`configs/config_dp.yml` is a complementary, simpler variant that predicts only the principal
diagnosis (`dp`). Compared to the default multi-head config it:

- has a single `dp` head (`eds.single_label_head`) instead of three;
- sets `vars.label_attr` to `["dp"]`;
- uses the public `almanach/camembertv2-base` transformer (no access restriction);
- loads `dp` **class weights** from `data/label_freq_dict_dp.pkl` to counter class imbalance;
- includes an example `tuning` block for Optuna hyper-parameter search.

```bash
uv run python -m edsnlp.train --config configs/config_dp.yml --seed 42
```

`edsnlp.train` is a confit `Cli`; the `train:` block of the config supplies the command
arguments. Any value can be overridden on the command line, e.g.:

```bash
uv run python -m edsnlp.train --config configs/config.yml \
    --train.max_steps 5000 \
    --vars.train_path /path/to/my/train
```

Checkpoints are written to `output_dir` (`artifacts/` by default, gitignored). The final model
is `artifacts/model-last`.

> Training is meant to run on a GPU. On the AP-HP cluster it is launched via SLURM; the
> `sbatch.sh` / `launch_slurm*` scripts are machine-local and gitignored — write your own for
> your scheduler, calling the `edsnlp.train` command above.

## Before you run: things to edit

`config.yml` contains **absolute, user-specific paths**. Edit them for your machine:

- `vars.train_path`, `vars.synth_path`, `vars.val_path` — your parquet datasets
  (see [data.md](data.md)).
- The `labels:` path under each head (`/export/home/ldedieu/.../data/valid_labels_all_*.pkl`)
  — point them at this repo's `data/` directory.
- `nlp.components.doc_classifier.embedding.embedding.model` — the transformer. The config uses
  `PARTAGES-dev/PARTAGES-camembert-large`, which may be access-restricted on the Hugging Face
  Hub. The public `almanach/camembertv2-base` is a drop-in alternative.
- `train.output_dir` — where checkpoints go.

## Using your own data

You do **not** need MISTRAL/PARHAF or the AP-HP Spark notebooks to train. If you have your own
clinical notes, the whole path is:

**1. Materialise your notes as parquet** matching the schema in
[data.md](data.md#parquet-schema) — one row per note, with `note_text` plus the label
column(s) you are training (`dp`, and optionally `das` / `mdp`). Any tool that writes parquet
works (e.g. `pandas.DataFrame.to_parquet`); Spark is not required.

**2. Pick a config to start from.**

- If you only have `dp`, start from [`configs/config_dp.yml`](../configs/config_dp.yml)
  (single head, public transformer).
- If you also have `das` / `mdp`, start from [`configs/config.yml`](../configs/config.yml).

**3. Decide what to do about the synthetic (`dp`) head.** Both configs add a second
`doc_classifier_syn` component fed from `vars.synth_path`, and `train.train_data` is
`[train_data, syn_data]`. This is a down-weighted `dp`-only augmentation. You have three options:

- **Reuse the shipped synonyms** (simplest, and `dp`-only so it fits any `dp` head): the source
  `data/synonyms.pkl` is in this repo and built from public referentials.
  `scripts.preprocess_synonyms` writes it to a `synth_path` parquet — point `vars.synth_path`
  there and you're done.
- **Bring your own synonym set.** The synthetic parquet is just the OMOP schema restricted to
  `dp`: two columns, `note_text` (the synonym / augmented text) and `dp` (its single ICD-10
  code). `vars.syn_label_attr` stays `["dp"]`. Any parquet with those columns works.
- **Train without it.** If you don't provide a `synth_path`, training **will fail** (the reader
  can't open an empty path), so you must actively remove the component:
  - delete the `syn_data:` block,
  - set `train.train_data` to just `[${ train_data }]`,
  - delete the `doc_classifier_syn` component and its `loss_scales` entry.

**4. Regenerate the label spaces for your codes.** Each head loads its label space from a
`.pkl` (the `labels:` key). Two options:

- **Full CIM-10 referential** (recommended, matches the shipped setup): run
  `python -m scripts.build_labels` — pandas only, no Spark — to (re)produce
  `data/valid_labels_all_{dp,das,mdp}.pkl`. This is independent of your data.
- **Your own code set**: pickle a plain `list[str]` of the codes you want each head to predict
  and point the head's `labels:` at it. Any code your data uses that is missing from this list
  is unscoreable, so make sure it is a superset of the codes present in your parquet.

  If you use `config_dp.yml`'s `class_weights:`, also regenerate
  `data/label_freq_dict_dp.pkl` — a `{code: document_count}` dict counted on your train set —
  or remove the `class_weights:` key to train unweighted.

**5. Edit the paths and model** as in [*Before you run*](#before-you-run-things-to-edit):
`vars.train_path`, `vars.val_path`, each head's `labels:`, and `train.output_dir`. Keep the
public `almanach/camembertv2-base` transformer unless you have access to
`PARTAGES-dev/PARTAGES-camembert-large`.

**6. Train:**

```bash
uv run python -m edsnlp.train --config configs/config_dp.yml --seed 42
```

## Config walkthrough

### `vars`
Shared values referenced elsewhere with `${vars.…}`:

| Key                   | Meaning                                                   |
|-----------------------|----------------------------------------------------------|
| `train_path`/`synth_path`/`val_path` | Parquet dataset paths                     |
| `label_attr`          | `["dp", "das", "mdp"]` — heads trained on the real data   |
| `syn_label_attr`      | `["dp"]` — head trained on synthetic data                |
| `max_step`            | Total optimizer steps (`3000`)                           |
| `validation_interval` | Validate every N steps (`1000`)                          |

### `nlp` — the pipeline
- `eds.transformer` — the backbone, sliding `window: 256` tokens.
- `eds.doc_pooler` — `attention` pooling to one embedding per document.
- `eds.doc_classifier` — holds the three heads. Each head sets its label space
  (`labels:` → a `.pkl`), `loss` (`ce` for single-label, `bce` for multi-label), `hidden_size`
  (`1024`), `activation_mode` (`gelu`), `dropout_rate`, `layer_norm`, and `loss_weight`. The
  `das` head adds `selection: threshold` with `threshold: 0.5` (a code is predicted when its
  probability passes the threshold).
- `doc_classifier_syn` is a confit reference (`${ nlp.components.doc_classifier }`) — it
  **shares weights** with `doc_classifier` but is fed the synthetic data.

### `scorer`
`eds.doc_classif` computes per-head precision/recall/F over `label_attr`. `batch_size: 150 docs`,
`speed: true`.

### `optimizer`
`torch.optim.AdamW` with `total_steps = max_step` and **two learning-rate groups**, each with a
linear warmup schedule:

| Selector (regex)                          | Targets                | Max LR  | Warmup |
|-------------------------------------------|------------------------|---------|--------|
| `doc_classifier[.]embedding[.]embedding`  | The transformer body   | `3e-5`  | 0.15   |
| `.*`                                       | Everything else (heads)| `5e-4`  | 0.10   |

This trains the pretrained transformer gently while letting the freshly-initialised heads learn
faster.

### `train_data` / `syn_data` / `val_data`
Each reads parquet with `converter: omop` and the relevant `doc_attributes`. `train_data` is
routed to the `doc_classifier` pipe, `syn_data` to `doc_classifier_syn` (`pipe_names`). The two
training sources are combined in `train.train_data: [train_data, syn_data]`.

### `train`
The training loop: combines the two data streams, weights their losses with `loss_scales`
(`doc_classifier: 1.0`, `doc_classifier_syn: 0.1` — synthetic data contributes less),
`grad_max_norm: 1.0`, `mixed_precision: bf16`, `num_workers: 16`, `cpu: False`.

### `logger`
`json` (file) + `rich` (console). The `rich` logger surfaces the loss and the per-head
validation micro-F1 (`dp`, `das`, `mdp`).

## Hyper-parameter tuning

EDS-NLP ships an Optuna-based `edsnlp.tune` CLI. `configs/config_dp.yml` includes an example
`tuning` block (searching over the transformer LR, the head LR and `hidden_size`):

```bash
uv run python -m edsnlp.tune --config configs/config_dp.yml
```

`configs/config.yml` does not define a `tuning` block; copy the one from `config_dp.yml` and
adapt the hyper-parameter paths to the multi-head structure
(e.g. `nlp.components.doc_classifier.heads.dp.hidden_size`). See the
[EDS-NLP training documentation](https://aphp.github.io/edsnlp/) for the schema.

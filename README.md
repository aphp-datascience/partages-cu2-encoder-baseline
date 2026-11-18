# CU2 Encoder — principal diagnosis (ICD-10) of French clinical notes

Encoder model of the **PARTAGES** project, use case **CU2**: given the text of a French
hospital stay, predict its **principal diagnosis** in ICD-10 (CIM-10, ATIH version), as coded
in the French PMSI (DP and, when the DP is a Z code, DR). It is an
[EDS-NLP](https://github.com/aphp/edsnlp) pipeline (`eds.doc_classifier`, edsnlp 0.24.0),
released on the Hugging Face Hub:
**[`LucasDedieu/cu2-icd10-dp-encoder`](https://huggingface.co/LucasDedieu/cu2-icd10-dp-encoder)**.

This repository holds what you need to **use it on your data**: inference, evaluation and
fine-tuning.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). Pick the torch build matching your
hardware (a bare `uv sync` installs no torch):

```bash
uv sync --extra cu124    # CUDA >= 12.4 (cu121, cu118 and cpu also exist)
```

Check `nvidia-smi` (top right, "CUDA Version") and take the highest `cuXXX` not above it.
Inference runs on CPU but slowly; fine-tuning needs a GPU.

Get the model (the Hub repository may be private: ask the CU2 team for access, then
`uv run huggingface-cli login`):

```bash
uv run huggingface-cli download LucasDedieu/cu2-icd10-dp-encoder \
    --local-dir models/cu2-icd10-dp-encoder
```

The scripts accept the local dir or the repo id (downloaded on first use); the fine-tuning
config needs the local dir. On a machine without internet, copy that folder over.

## Architecture

A BERT-base encoder (see [Training recipe](#training-recipe-of-the-released-model)) reads the
text in sliding windows of 256 tokens; an attention pooler makes one vector per stay; two
classification heads (MLP 1024, GELU, layer norm) read it:

| Head        | Predicts                                                                   | Labels |
|-------------|----------------------------------------------------------------------------|-------:|
| `affection` | the disease: the DR when the DP is a Z code, otherwise the DP (all ICD-10 codes valid as DP/DR, Z80-Z99 included) | 12 950 |
| `mdp`       | the *mode de prise en charge*: the Z code used as DP, or `Z769` = no Z code | 720 |

Why two heads: a stay coded "DP = Z51.1 (chemotherapy), DR = C50.9" and a stay coded
"DP = C50.9" describe the same disease; splitting the code lets the model learn the disease
from both. The PMSI pair is rebuilt from the two heads: `mdp = Z769` → DP = affection, no DR;
otherwise DP = the Z code, and the affection is kept as DR or dropped according to a
per-Z-code rule ([`data/dr_rule.json`](data/dr_rule.json)). The single code compared with a
gold standard is **`target` = DR if any, else DP**.

## Data format

**You keep your PMSI coding as it is (DP and DR).** The conversion to the two heads is done
by [`scripts/prepare_heads.py`](scripts/prepare_heads.py) before fine-tuning; you never
write `affection` / `mdp` by hand.

One parquet file (or a folder of parquet files), **one row per hospital stay**:

| Column      | Needed for                     | Content |
|-------------|--------------------------------|---------|
| `note_id`   | always                         | unique id of the stay (string or int) |
| `note_text` | always                         | the text of the stay: its reports (discharge summary, operative report…) concatenated in one string |
| `dp`        | fine-tuning, evaluation        | PMSI principal diagnosis, ICD-10 code (`C509` or `C50.9`) |
| `dr`        | fine-tuning, evaluation        | PMSI related diagnosis, empty or null when none |

Other column names are fine: pass them with `--dp` / `--dr` (`prepare_heads`) or
`--gold-dp` / `--gold-dr` (`predict`). Inference only needs `note_id` and `note_text`.

## Inference

```python
import edsnlp
from scripts.heads import load_rule, reconstruct

nlp = edsnlp.load("models/cu2-icd10-dp-encoder")
rule = load_rule("data/dr_rule.json")

doc = nlp("Compte rendu d'hospitalisation ...")
doc._.affection, doc._.mdp                              # raw head outputs
dp, dr = reconstruct(doc._.mdp, doc._.affection, rule)  # PMSI pair
```

On a dataset (multiprocessing and GPU handled by edsnlp):

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-encoder \
    --data stays.parquet --out preds.parquet [--gold-dp dp --gold-dr dr]
```

The output has one row per stay: `affection`, `mdp`, `dp`, `dr`, `target`. With the gold
columns, it also prints the accuracy of `target` against your coding (exact code and first
3 characters).

## Fine-tuning on your data

1. **Build the head targets** for a train and a validation split (parquet as in
   [Data format](#data-format), with `dp` and `dr`):

   ```bash
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_train.parquet --out data/my_train
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_dev.parquet --out data/my_dev
   ```

   Codes are normalised (dots removed, upper case). A code outside a head's label space
   masks that head for the stay; the counts are printed.

2. **Train** (one GPU; set the paths in the `vars` of
   [`configs/finetune.yml`](configs/finetune.yml) or override them):

   ```bash
   uv run python -m scripts.train --config configs/finetune.yml --seed 42 \
       --vars.train_path data/my_train --vars.val_path data/my_dev
   ```

   The model is written to `artifacts/model-last` and loads with `edsnlp.load` like the
   released one. `scripts.train` only adds label smoothing (0.1) on the `affection` head, as
   in the original training (`LABEL_SMOOTHING=0` to remove it).

   Defaults: 4 000 steps of 16 stays, learning rate 1e-5 for the encoder and 1.7e-4 for the
   heads, bf16. On real AP-HP stays, 30 000 coded stays gave most of the gain. A validation
   set of 1 000-2 000 stays is enough. Out of memory: lower `sub_batch_size` (a single stay is
   never split, so very long stays may have to be removed).

3. **Evaluate** before and after on the same held-out stays with `scripts.predict
   --gold-dp dp --gold-dr dr`, passing `--model artifacts/model-last` for the fine-tuned one.

The DR rule ([`data/dr_rule.json`](data/dr_rule.json)) says, for each Z code (or Z category)
predicted as DP, whether a DR is usually coded with it. It was estimated on PMSI coding and
can be replaced by your own (same format: `{"code": {...}, "categorie": {...}}`), or disabled
with `--rule none` (the affection is then always kept as DR).

## Training recipe of the released model

No real patient data was used. One training run, all weights trained:

- **Backbone**: initialised from
  [`doctolib-lab/doctobert-fr-base`](https://huggingface.co/doctolib-lab/doctobert-fr-base)
  and fine-tuned end to end with the heads (the released encoder weights differ from it).
- **Data**, three streams in every step (shared weights, loss scales in brackets):
  - MISTRAL synthetic stays, 100 039 stays × 3 copies with 15 % of the words masked
    (32 stays per batch, scale 1);
  - fictitious PARHAF patients ([HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF),
    CC BY 4.0) and CU0 synthetic stays derived from them, 6 440 stays, labelled in DP/DR
    by an LLM teacher (Qwen3.8-27B with a PMSI coding procedure) over the whole ICD-10 (4 stays per
    batch, scale 1);
  - ICD-10 labels and synonyms of every code, 168 313 short texts (128 per batch, scale 0.1).
- **Optimisation**: AdamW, 12 000 steps, linear schedules with warm-up, learning rate 3e-5
  for the encoder and 5e-4 for the pooler and heads, gradient clipping 1.0, bf16; label
  smoothing 0.1 on `affection`, plain cross-entropy on `mdp`; seed 42; last checkpoint kept.
- **Variant**: base recipe (variant A). Four variants were compared on the same data (an
  auxiliary 3-character head, heads initialised from code descriptions, both, and masking
  the PARHAF/CU0 stream): none beat the base recipe significantly.

## Results

Accuracy of `target` (DR if any, else DP) with the DR rule; none of these stays were used
for training.

| Test set | Stays | Exact | 3 characters |
|---|---:|---:|---:|
| PARHAF CU2 test set (fictitious patients, revised gold) | 288 | 63.9 % | 79.5 % |
| MISTRAL test (synthetic stays) | 14 253 | 92.3 % | 94.3 % |

Note styles and coding habits differ between hospitals: fine-tuning on your own coded
stays is recommended.

## Repository layout

```
configs/finetune.yml     fine-tuning config (edsnlp.train)
scripts/predict.py       batch inference and accuracy
scripts/prepare_heads.py PMSI DP/DR -> affection / mdp targets
scripts/train.py         edsnlp.train + per-head label smoothing
scripts/heads.py         DP/DR <-> heads, DR rule
data/dr_rule.json        keep-the-DR rule per Z code
```

The previous multi-head baseline (`dp` / `das` / `mdp`, edsnlp fork `classif_head`) is in the
git history (commit `2531454`).

## Contributing

Work on a branch (`feat/…`, `fix/…`, `docs/…`) and open a pull request against `main`. Lint
with `uv run pre-commit install` once, or `uv run ruff check . && uv run ruff format .`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

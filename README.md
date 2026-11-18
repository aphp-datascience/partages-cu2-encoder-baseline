# CU2 Encoder — principal diagnosis (ICD-10) of French clinical notes

Encoder model of the **PARTAGES** project, use case **CU2**: given a French hospital-stay
note, predict its **principal diagnosis** in ICD-10 (CIM-10, ATIH version), as coded in the
French PMSI (DP and, when the DP is a Z code, DR). It is an
[EDS-NLP](https://github.com/aphp/edsnlp) pipeline (`eds.doc_classifier`, edsnlp 0.24.0) on a
French clinical BERT, released on the Hugging Face Hub:
**[`LucasDedieu/cu2-icd10-dp-encoder`](https://huggingface.co/LucasDedieu/cu2-icd10-dp-encoder)**.

This repository holds what you need to **use it on your data**: inference, evaluation and
fine-tuning.

## The model in one paragraph

The note goes through `doctolib-lab/doctobert-fr-base` (sliding windows of 256 tokens), an
attention pooler, then two classification heads:

| Head        | Predicts                                                                   | Labels |
|-------------|----------------------------------------------------------------------------|-------:|
| `affection` | the disease: the DR when the DP is a Z code, otherwise the DP (all ICD-10 codes valid as DP/DR, Z80-Z99 included) | 12 950 |
| `mdp`       | the *mode de prise en charge*: the Z code used as DP, or `Z769` = no Z code | 720 |

The PMSI pair is rebuilt from the two heads: `mdp = Z769` → DP = affection, no DR; otherwise
DP = the Z code, and the affection is kept as DR or dropped according to a per-Z-code rule
([`data/dr_rule.json`](data/dr_rule.json)). The single code to compare with a gold standard is
**`target` = DR if any, else DP**.

It was trained without any real patient data: synthetic notes (MISTRAL, CU0), the fictitious
patients of [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) re-annotated by an
LLM, and the CIM-10 labels and synonyms. Fine-tuning on your own coded stays is expected to
help a lot (different note styles, coding habits).

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

On a dataset (parquet with `note_id` and `note_text`; multiprocessing and GPU handled by
edsnlp):

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-encoder \
    --data notes.parquet --out preds.parquet
```

The output has one row per note: `affection`, `mdp`, `dp`, `dr`, `target`. Add
`--gold-dp <col> [--gold-dr <col>]` to print the accuracy of `target` against your coding
(exact code and first 3 characters).

## Fine-tuning on your data

1. **Build the head targets** from your PMSI coding (parquet with `note_id`, `note_text`, DP
   and optional DR columns), for a train and a validation split:

   ```bash
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_train.parquet --out data/my_train --dp dp --dr dr
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_dev.parquet --out data/my_dev --dp dp --dr dr
   ```

   Codes are normalised (dots removed, upper case). A code outside a head's label space
   masks that head for the note; the counts are printed.

2. **Train** (one GPU; set the paths in the `vars` of
   [`configs/finetune.yml`](configs/finetune.yml) or override them):

   ```bash
   uv run python -m scripts.train --config configs/finetune.yml --seed 42 \
       --vars.train_path data/my_train --vars.val_path data/my_dev
   ```

   The model is written to `artifacts/model-last` and loads with `edsnlp.load` like the
   released one. `scripts.train` only adds label smoothing (0.1) on the `affection` head, as
   in the original training (`LABEL_SMOOTHING=0` to remove it).

   Defaults: 4 000 steps of 16 notes, learning rate 1e-5 for the encoder and 1.7e-4 for the
   heads, bf16. On real AP-HP stays, 30 000 coded notes gave most of the gain. A validation
   set of 1 000-2 000 notes is enough. Out of memory: lower `sub_batch_size` (a single note is
   never split, so very long notes may have to be removed).

3. **Evaluate** before and after on the same held-out notes with `scripts.predict
   --gold-dp ... --gold-dr ...`, passing `--model artifacts/model-last` for the fine-tuned one.

The DR rule ([`data/dr_rule.json`](data/dr_rule.json)) says, for each Z code (or Z category)
predicted as DP, whether a DR is usually coded with it. It was estimated on PMSI coding and
can be replaced by your own (same format: `{"code": {...}, "categorie": {...}}`), or disabled
with `--rule none` (the affection is then always kept as DR).

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

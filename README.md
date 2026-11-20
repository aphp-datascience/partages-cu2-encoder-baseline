# CU2 Encoder — principal diagnosis (ICD-10) of French hospital stays

**PARTAGES** project, use case **CU2**. Given the reports of a French hospital stay, the model
suggests its **principal diagnosis** as coded in the PMSI: the DP, and the DR when the DP is a
Z code, in ICD-10 (CIM-10, ATIH version). It is an [EDS-NLP](https://github.com/aphp/edsnlp)
model released on the Hugging Face Hub (current version `v0.1`):
**[`PARTAGES-CU2/cu2-icd10-dp-encoder`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder)**.

This repository contains everything a hospital needs to **run the model on its own stays,
measure it, and fine-tune it on its own PMSI coding**.

**What it is for.** Coding assistance: a suggested DP/DR per stay and a ranked list of
candidate codes (top-k), to speed up or double-check coding by the DIM. **What it is not.**
An automatic coder: predictions must be validated by a coder. It does not predict associated
diagnoses (DAS) or procedures.

**Privacy.** Everything runs on your own machines: the model is downloaded once, and no text
or code ever leaves your infrastructure. The released model was trained without any real
patient data (see [Technical details](#technical-details)).

## Quick start

| Step | What you do | Section |
|---|---|---|
| 1 | Request access to the model, install the repository | [Installation](#1-installation) |
| 2 | Export your stays: reports, CCAM acts, PMSI DP/DR | [Preparing your data](#2-preparing-your-data) |
| 3 | Run the released model on held-out stays: your baseline | [Running and evaluating](#3-running-and-evaluating-the-model) |
| 4 | Fine-tune it on your own coded stays | [Fine-tuning](#4-fine-tuning-on-your-stays) |
| 5 | Compare before / after on the same held-out stays, then use the better model | [Running and evaluating](#3-running-and-evaluating-the-model) |

## 1. Installation

**Hardware.** The model is small (a BERT-base encoder, about 110 M parameters): no large
GPU is needed.

- **Inference:** any recent NVIDIA GPU is enough; a CPU also works for small volumes (slower).
- **Fine-tuning:** a GPU is recommended. The reference fine-tuning used one A100, but a
  smaller GPU works with smaller batches (see the [FAQ](#faq)).

**Software.** Python 3.12 and [uv](https://docs.astral.sh/uv/). Pick the PyTorch build that
matches your GPU driver (`nvidia-smi`, top right "CUDA Version": take the highest `cuXXX` not
above it; `cpu` without GPU):

```bash
git clone https://github.com/aphp-datascience/partages-cu2-encoder-baseline.git
cd partages-cu2-encoder-baseline
uv sync --extra cu124    # or cu121, cu118, cpu
```

**Model.** The Hub repository is private for now: ask the CU2 team for access, then log in
and download it once:

```bash
uv run huggingface-cli login
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-encoder --revision v0.1 \
    --local-dir models/cu2-icd10-dp-encoder
```

On a server without internet access, download on another machine and copy the
`models/cu2-icd10-dp-encoder` folder over. All commands below are run from the repository
root.

## 2. Preparing your data

One parquet file (or a folder of parquet files), **one row per hospital stay (RUM)**:

| Column | Needed for | Content |
|---|---|---|
| `note_id` | always | a unique id of the stay |
| `note_text` | always | the reports of the stay in chronological order, separated by a blank line: **consultation reports (CRC), discharge summaries (CRH) and operative reports (CR-OP)** |
| `acts` | strongly recommended | the CCAM procedure codes of the stay: a list, or a string such as `NFFC004;ZBQK002`; empty if none |
| `dp` | evaluation, fine-tuning | the PMSI principal diagnosis, ICD-10 (`C509` or `C50.9`) |
| `dr` | evaluation, fine-tuning | the PMSI related diagnosis, empty or null when none |

- **Keep your PMSI coding as it is.** You never convert it yourself: the scripts handle the
  model's internal format.
- **Procedures.** The model was trained with the procedures of the stay as the first line of
  the text (`Actes (CCAM) : NFFC004 - Méniscectomie … ; …`, or `non renseigné`). Pass your
  acts column with `--acts acts` and the scripts write this line, using the CCAM 2026
  nomenclature shipped in [`data/ccam_2026.csv`](data/ccam_2026.csv) (ATIH). Without acts the
  model still works, slightly less well.
- **Which stays.** Closed stays with at least one CRC, CRH or CR-OP. Keep the stays the way
  your DIM codes them; one row per RUM.
- **Splits.** For fine-tuning, set aside a **test** set first (stays never used for training,
  ideally from a later period), then a **validation** set of 1 000-2 000 stays, and use the
  rest for **training**. On real AP-HP stays, about 30 000 coded training stays gave most of
  the gain of fine-tuning.

## 3. Running and evaluating the model

On a whole file, with your coding to measure accuracy (optional):

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-encoder \
    --data my_test.parquet --out preds.parquet --acts acts --gold-dp dp --gold-dr dr --topk 10
```

`preds.parquet` has one row per stay:

| Column | Meaning |
|---|---|
| `dp`, `dr` | the suggested PMSI coding (DR only when the DP is a Z code) |
| `target` | the single code to compare with your coding: the DR if any, else the DP |
| `candidates`, `scores` | with `--topk k`: the k most likely codes, `target` first |
| `affection`, `mdp` | raw outputs of the model (see [Technical details](#technical-details)) |

With `--gold-dp/--gold-dr`, it prints the share of stays where `target` equals your coding,
on the exact code and on the 3-character category (and in the top-k). Run it before and after
fine-tuning on the **same** test stays to measure the gain.

From Python, for one stay:

```python
import edsnlp
from scripts.acts import load_ccam, with_acts
from scripts.heads import load_rule, reconstruct

nlp = edsnlp.load("models/cu2-icd10-dp-encoder")
rule, ccam = load_rule("data/dr_rule.json"), load_ccam()

doc = nlp(with_acts("Compte rendu d'hospitalisation ...", ["NFFC004"], ccam))
dp, dr = reconstruct(doc._.mdp, doc._.affection, rule)
```

## 4. Fine-tuning on your stays

1. **Prepare the training and validation files** (same format, with `dp` and `dr`):

   ```bash
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_train.parquet --out data/my_train --acts acts
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_dev.parquet --out data/my_dev --acts acts
   ```

   The counts printed tell how many stays were kept, and how many codes fall outside the
   model's label space (those stays are partly ignored, never wrongly learned).

2. **Train** on one GPU:

   ```bash
   uv run python -m scripts.train --config configs/finetune.yml --seed 42 \
       --vars.train_path data/my_train --vars.val_path data/my_dev
   ```

   The fine-tuned model is written to `artifacts/model-last`; use it exactly like the
   released one (`--model artifacts/model-last`). Settings are in
   [`configs/finetune.yml`](configs/finetune.yml) (4 000 steps of 16 stays by default). Use
   the training files as they are (no masking or augmentation).

3. **Evaluate** it on your test stays (section 3) and compare with the released model.

Note styles and coding habits differ between hospitals: fine-tuning on your own coded stays is
recommended; on AP-HP stays it brought a clear gain over the released model.

## Results of the released model

Before any fine-tuning. Accuracy of `target` (DR if any, else DP), exact code and 3-character
category; top-k: the gold code is among the k `candidates`.

**Physician-reviewed AP-HP stays.** AP-HP stays whose PMSI coding was reviewed by physicians
(May 2026), several reviewers for part of them; text = acts + CRC + CRH + CR-OP, as above.
Three views of the same set: **flattened** (every review is a row), **agreement + single**
(one row per stay; stays reviewed once, or by several reviewers who agree), **agreement**
(only stays with several agreeing reviewers: the most reliable gold, few stays). **ICD-10**:
whole classification; **surgery**: the 3 568 codes of the surgical scope (stays coded with one
of them, predictions restricted to them).

| View | Scope | Stays | Exact | 3 char. | Top-10 | Top-20 | 3 char. top-10 | 3 char. top-20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 40.3 % | 51.0 % | 68.4 % | 73.6 % | 77.7 % | 82.6 % |
| flattened | surgery | 259 | 48.6 % | 56.8 % | 77.2 % | 80.7 % | 82.6 % | 86.1 % |
| agreement + single | ICD-10 | 195 | 40.5 % | 51.8 % | 69.2 % | 75.4 % | 78.5 % | 82.6 % |
| agreement + single | surgery | 167 | 47.9 % | 56.3 % | 76.6 % | 80.2 % | 82.0 % | 85.0 % |
| agreement | ICD-10 | 47 | 68.1 % | 70.2 % | 89.4 % | 91.5 % | 91.5 % | 93.6 % |
| agreement | surgery | 42 | 76.2 % | 76.2 % | 95.2 % | 95.2 % | 95.2 % | 95.2 % |

These sets are small: the 95 % confidence interval is about ±7 points on ~200 stays and ±13
points on ~50.

**Other test sets** (no real patient data):

| Test set | Stays | Exact | 3 char. | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|
| PARHAF, CU2 test patients (fictitious patients written by physicians, gold reviewed) | 288 | 68.4 % | 82.3 % | 92.0 % | 94.8 % |
| MISTRAL test (synthetic stays generated by an LLM) | 14 253 | 92.4 % | 94.4 % | 97.6 % | 97.8 % |

## FAQ

- **We have no CCAM acts.** Leave `--acts` out: the model still works, slightly less well
  (the first line of its training texts listed the acts).
- **Our stays have no DR column.** Pass only `--dp` (or `--gold-dp`): the DR is then
  considered absent.
- **Out of GPU memory during fine-tuning** (e.g. on a 16-24 GB GPU). Lower `batch_size` and
  `sub_batch_size` in `configs/finetune.yml`. A single stay is never split across batches, so
  very long stays may have to be removed from the training file.
- **Which version should we use?** The latest tag (`v0.1`). Pin it (`--revision v0.1`) so that
  your results stay reproducible when a new version is released.
- **Can the model predict all our codes?** It knows about 13 000 ICD-10 codes (all codes valid
  as DP or DR). A code outside this list cannot be predicted; `prepare_heads` counts them.
- **Questions, bugs.** Open an issue on this repository.

## Technical details

**Model.** A BERT-base encoder reads the text in windows of 256 tokens, an attention pooler
makes one vector per stay, and two classification heads read it:

| Head | Predicts | Labels |
|---|---|---:|
| `affection` | the disease: the DR if the DP is a Z code, else the DP | 12 950 |
| `mdp` | the Z code used as DP, or `Z769` = no Z code | 720 |

Splitting the code lets the model learn a disease from both "DP = Z51.1 (chemotherapy), DR =
C50.9" and "DP = C50.9". The PMSI coding is rebuilt from the two heads: `mdp = Z769` → DP =
the affection, no DR; otherwise DP = the Z code, and the affection is its DR if
[`data/dr_rule.json`](data/dr_rule.json) says that this Z code usually has a DR (estimated
on PMSI coding; it can be replaced by your own, same format, or disabled with `--rule none`).
`--topk` ranks the codes of both heads together, by their probability of being the target.

**Training of the released model** (no real patient data; one run, all weights trained):

- encoder initialised from
  [`doctolib-lab/doctobert-fr-base`](https://huggingface.co/doctolib-lab/doctobert-fr-base),
  then trained end to end;
- three data streams: **MISTRAL**, 100 039 synthetic stays (3 copies, 15 % of words masked);
  **PARHAF** ([HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF),
  CC BY 4.0, fictitious patients) and synthetic stays rewritten from them (6 440 stays), coded
  in DP/DR over the whole ICD-10 by an LLM agent (Mistral Large 4 with ICD-10 and MCO-guide
  tools and a PMSI coding procedure); the ICD-10 labels and synonyms of every code (168 313
  texts);
- heads initialised from the code descriptions (helps rare codes); AdamW, 12 000 steps,
  learning rate 3e-5 (encoder) / 5e-4 (heads), label smoothing 0.1 on `affection`, bf16,
  seed 42.

**Versions.** `v0`: PARHAF labels from a Qwen 27B teacher. `v0.1`: Mistral Large 4 labels
(better on the PARHAF test patients and on the reviewed AP-HP stays, on par on a large AP-HP
test set).

**Repository layout.**

```
configs/finetune.yml     fine-tuning settings
scripts/predict.py       inference and accuracy on a file
scripts/prepare_heads.py PMSI DP/DR -> training files
scripts/train.py         training (edsnlp.train + label smoothing)
scripts/heads.py         DP/DR <-> model outputs, DR rule
scripts/acts.py          CCAM procedure line of the input text
data/dr_rule.json        keep-the-DR rule per Z code
data/ccam_2026.csv       CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Contributions are welcome through pull requests
(`uv run pre-commit install` once to lint your changes).

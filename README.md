# PARTAGES-CU2: encoder for French ICD-10 medical coding

Given the reports of a French hospital stay, the model suggests how the stay is coded in the
PMSI (the French hospital activity database): its **principal diagnosis (DP)** and, when the DP
is a Z code, its **related diagnosis (DR)**, in ICD-10 (CIM-10, ATIH version), together with a
ranked list of candidate codes. It was developed in the PARTAGES project (use case 2, coding
assistance for medical information departments, DIM). The use case targets the DP of surgical
stays; the model goes further and covers the DR and the whole ICD-10. It is an
[EDS-NLP](https://github.com/aphp/edsnlp) model released on the Hugging Face Hub:
**[`PARTAGES-CU2/cu2-icd10-dp-encoder`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder)**.

This repository contains everything a hospital needs to **run the model on its own stays,
measure it, and fine-tune it on its own PMSI coding**.

**Terms used below**

- **DP** (*diagnostic principal*): the health problem that motivated the stay.
- **Z code**: an ICD-10 code of chapter XXI, which describes a reason for contact rather than a
  disease (e.g. `Z51.1` chemotherapy session, `Z09.0` follow-up after surgery).
- **DR** (*diagnostic relié*): when the DP is a Z code, the disease behind it (e.g. the cancer
  treated during a chemotherapy session).
- **Target code**: the disease of the stay, i.e. the DR if there is one, otherwise the DP. The
  results below are measured on it.
- **`affection` and `mdp`**: the two outputs of the model. `affection` is the disease (the target
  code); `mdp` (*motif de prise en charge*) is the Z code used as DP, or `Z769` by default when
  the DP is not a Z code.
- **CRC / CRH / CR-OP**: consultation reports, discharge summaries, operative reports.

**In short**

- **For:** coding assistance (suggestion and double-check by the DIM). **Not for:** automatic
  coding without validation; it does not predict associated diagnoses (DAS).
- **Runs locally:** the model is downloaded once; no text or code ever leaves your
  infrastructure. The released model was trained without any real patient data.
- **Light:** a BERT-base encoder (~110 M parameters); inference on any recent GPU (or CPU for
  small volumes); for fine-tuning, a GPU with ≥ 16 GB of memory is recommended.
- **Mostly trained on surgical stays**, but covers the whole ICD-10 (see
  [Limitations](#limitations)).
- **Adapt it to your hospital:** fine-tuning on your own coded stays is recommended; on AP-HP
  stays it brought a clear gain over the released model.

## Quick start

| Step | What you do | Section |
|---|---|---|
| 1 | Request access to the model, install the repository | [Installation](#1-installation) |
| 2 | Export your stays: reports, CCAM acts, PMSI DP/DR | [Input data](#2-input-data) |
| 3 | Run the released model on held-out stays: your baseline | [Running and evaluating](#3-running-and-evaluating-the-model) |
| 4 | Fine-tune it on your own coded stays | [Fine-tuning](#4-fine-tuning-on-your-stays) |
| 5 | Compare before / after on the same held-out stays, then use the better model | [Running and evaluating](#3-running-and-evaluating-the-model) |

## 1. Installation

**Hardware.** The model is small (a BERT-base encoder, about 110 M parameters).

- **Inference:** any recent NVIDIA GPU is enough; a CPU also works for small volumes (slower).
- **Fine-tuning:** a GPU with ≥ 16 GB of memory is recommended. The reference fine-tuning used
  one A100; a smaller GPU works with smaller batches (see the [FAQ](#faq)).

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
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-encoder \
    --local-dir models/cu2-icd10-dp-encoder
```

On a server without internet access, download on another machine and copy the
`models/cu2-icd10-dp-encoder` folder over. All commands below are run from the repository
root.

## 2. Input data

The model reads **one text per stay**: the CCAM procedures of the stay on the first line
(`Actes (CCAM) : CODE - label ; CODE - label ; …`, or `Actes (CCAM) : non renseigné`), a blank
line, then the reports of the stay in chronological order (consultation, operation, discharge),
separated by blank lines. You do not build this text yourself: the scripts build it from one
parquet file (or a folder of parquet files) with **one row per hospital stay (RUM)**:

| Column | Needed for | Content |
|---|---|---|
| `note_id` | always | a unique id of the stay |
| `note_text` | always | the CRC, CR-OP and CRH of the stay, in chronological order, separated by a blank line |
| `acts` | strongly recommended | the CCAM codes of the stay: a list, or a string such as `NFFC004;ZBQK002`; empty if none |
| `dp` | evaluation, fine-tuning | the PMSI principal diagnosis (`C509` or `C50.9`) |
| `dr` | evaluation, fine-tuning | the PMSI related diagnosis (empty when none) |

- **Keep your PMSI coding as it is.** You never convert it yourself: the scripts handle the
  model's two outputs.
- **Procedures.** Pass your acts column with `--acts acts`: the scripts write the first line,
  using the CCAM 2026 nomenclature shipped in [`data/ccam_2026.csv`](data/ccam_2026.csv)
  (ATIH). Without acts the model still works, slightly less well.
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
| `target` | the target code, to compare with your coding: the DR if any, else the DP |
| `candidates`, `scores` | with `--topk k`: the k most likely codes, `target` first |
| `affection`, `mdp` | the two raw outputs of the model (see [Architecture](#architecture)) |

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

text = "Compte rendu de consultation ...\n\nCompte rendu opératoire ...\n\nCompte rendu d'hospitalisation ..."
doc = nlp(with_acts(text, ["NFFC004"], ccam))
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

## Model and training

### Architecture

![Architecture: stay text (CCAM acts + reports) → windows of 256 tokens → DoctoBERT encoder → attention pooler → stay embedding → affection head (12 950 ICD-10 codes) and mdp head (720 Z codes, Z769 by default) → PMSI rule → DP, DR, target code and top-k candidates](docs/architecture.svg)

A BERT-base encoder initialised from
[`doctolib-lab/doctobert-fr-base`](https://huggingface.co/doctolib-lab/doctobert-fr-base)
reads the whole stay in windows of 256 tokens; an attention pooler turns them into one stay
embedding, fed to two classification heads. The whole model is trained end to end.

| Head | Predicts | Labels |
|---|---|---:|
| `affection` | the disease: the DR if the DP is a Z code, else the DP | 12 950 |
| `mdp` | the Z code used as DP, or `Z769` by default | 720 |

Splitting the code lets the model learn a disease from both "DP = Z51.1 (chemotherapy), DR =
C50.9" and "DP = C50.9". The PMSI coding is rebuilt from the two heads: `mdp = Z769` → DP =
`affection`, no DR; otherwise DP = the Z code, and `affection` is its DR if
[`data/dr_rule.json`](data/dr_rule.json) says that this Z code usually has a DR (estimated
on PMSI coding; it can be replaced by your own, same format, or disabled with `--rule none`).
`--topk` ranks the codes of both heads together, by their probability of being the target.

### Training data

No real patient data was used for training.

| Source | Stays / texts | Distinct target codes | Codes |
|---|---:|---:|---|
| **MISTRAL**: synthetic surgical stays generated by an LLM | 100 039 | 4 133 (96 % in the surgical scope) | generation code, recoded by an LLM agent for 49 162 stays (see below) |
| **PARHAF** ([HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF), CC BY 4.0): fictitious patients written by physicians | 3 300 | 1 008 | LLM agent |
| **CU0**: synthetic stays rewritten from the PARHAF patients | 3 140 | 900 | LLM agent |
| **Synonyms**: ICD-10 labels and synonyms of every code | 168 313 | 12 948 | the code itself |

The stays of MISTRAL, PARHAF and CU0 together cover 4 142 of the 12 950 `affection` codes; the
other codes are only seen through their labels and synonyms.

**Recoding with an LLM agent.** The original codes are noisy:

- **MISTRAL** stays were generated from a code that the generated text does not always match;
- **PARHAF** patients were written by physicians from a scenario that proposed a DP code (drawn
  from national claims data), and the reports follow it more or less closely: the proposed code
  is wrong for roughly a third to a half of the patients;
- **CU0** stays inherit the code of their source PARHAF patient.

They were recoded in DP/DR by **Mistral Large 4** used as an agent: it reads the stay, must query
offline tools (search in the ICD-10, code sheet with inclusions, exclusions and PMSI status,
comparison of the codes of a category, search in the ATIH MCO methodological guide) following a
mandatory PMSI coding procedure, and answers a DP and a DR. On PARHAF patients with a
physician-reviewed coding, it agrees with the reviewed code in about 80 % of the stays. All PARHAF
and CU0 stays were recoded; for MISTRAL, 10 500 stays drawn at random or in difficult cases, plus
the 39 084 stays where a smaller model disagreed with the original code (the code changed for
34 674 of the 49 162 recoded stays). Only fictitious or synthetic texts were sent to the API.

### Training parameters

- **Input:** windows of 256 tokens over the whole stay; the CCAM line is part of the text.
- **Augmentation:** MISTRAL is seen as 3 copies with 15 % of the words masked.
- **Heads:** initialised from the embeddings of the code descriptions (helps rare codes);
  cross-entropy, label smoothing 0.1 on `affection`.
- **Sampling:** each step mixes 32 MISTRAL stays, 4 PARHAF/CU0 stays and 128 synonyms (loss
  weight 0.1 for the synonyms).
- **Optimisation:** AdamW, 12 000 steps, learning rate 3e-5 for the encoder and 5e-4 for the
  heads (linear warm-up and decay), gradient clipping 1.0, bf16, seed 42.

## Results (before any fine-tuning on real hospital data)

Accuracy of the target code: exact code, gold code among the top-10 or top-20 `candidates`,
and 3-character category.

**Physician-reviewed AP-HP stays** (May 2026; text = acts + CRC + CRH + CR-OP). Views:
**flattened** = every review is a row; **agreement + single** = one row per stay, reviewed once
or by several reviewers who agree; **agreement** = only stays with several agreeing reviewers.
**ICD-10** = whole classification; **surgery** = stays coded in the surgical scope of the use
case (~3 500 codes), predictions restricted to it.

| View | Scope | Stays | Exact | Top-10 | Top-20 | 3 char. |
|---|---|---:|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 42.3 % | 66.5 % | 71.6 % | 53.5 % |
| flattened | surgery | 259 | 49.0 % | 73.0 % | 78.0 % | 58.3 % |
| agreement + single | ICD-10 | 195 | 42.6 % | 68.2 % | 74.4 % | 54.4 % |
| agreement + single | surgery | 167 | 47.9 % | 72.5 % | 78.4 % | 57.5 % |
| agreement | ICD-10 | 47 | 70.2 % | 87.2 % | 89.4 % | 72.3 % |
| agreement | surgery | 42 | 76.2 % | 90.5 % | 92.9 % | 76.2 % |

Small sets: 95 % confidence interval about ±7 points on ~200 stays, ±13 points on ~50.

**Other test set** (no real patient data): PARHAF CU2 test patients (288 fictitious patients
written by physicians, gold reviewed): 75.7 % exact, 95.8 % top-10, 96.9 % top-20, 88.9 % 3 char.

## Limitations

- **Mostly trained on surgery.** MISTRAL, by far the largest source, only covers ~4 000 codes,
  almost all surgical, and most PARHAF patients are surgical too; the other codes are learnt
  from few stays or from their synonyms only. Although the model was built to cover the whole
  ICD-10, it is clearly more accurate on surgical codes (compare the **surgery** rows above).
  Fine-tuning on your own stays is the way to improve the other specialties.
- One DP (and DR) per stay; no associated diagnoses.
- Trained on synthetic or fictitious notes: expect a domain shift on real hospital notes, which
  fine-tuning on your own coded stays reduces.
- Evaluated on real stays of a single hospital group (AP-HP, single-RUM stays with CRC, CRH or
  CR-OP reports; AP-HP data used for evaluation only, never for training). Other report types
  and multi-RUM stays have not been evaluated specifically.

## FAQ

- **We have no CCAM acts.** Leave `--acts` out: the model still works, slightly less well
  (the first line of its training texts listed the acts).
- **Our stays have no DR column.** Pass only `--dp` (or `--gold-dp`): the DR is then
  considered absent.
- **Out of GPU memory during fine-tuning** (e.g. on a 16-24 GB GPU). Lower `batch_size` and
  `sub_batch_size` in `configs/finetune.yml`. A single stay is never split across batches, so
  very long stays may have to be removed from the training file.
- **Can the model predict all our codes?** It knows the 12 950 ICD-10 codes valid as DP or DR
  and the 720 Z codes used as DP. A code outside these lists cannot be predicted;
  `prepare_heads` counts them.
- **Questions, bugs.** Open an issue on this repository.

## Repository layout

```
configs/finetune.yml     fine-tuning settings
scripts/predict.py       inference and accuracy on a file
scripts/prepare_heads.py PMSI DP/DR -> training files
scripts/train.py         training (edsnlp.train + label smoothing)
scripts/heads.py         DP/DR <-> model outputs, DR rule
scripts/acts.py          CCAM procedure line of the input text
data/dr_rule.json        keep-the-DR rule per Z code
data/ccam_2026.csv       CCAM 2026 nomenclature (ATIH)
docs/architecture.svg    architecture diagram
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Contributions are welcome through pull requests
(`uv run pre-commit install` once to lint your changes).

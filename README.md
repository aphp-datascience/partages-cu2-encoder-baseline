# CU2 Encoder — principal diagnosis (ICD-10) of French clinical notes

Encoder model of the **PARTAGES** project, use case **CU2**: given the text of a French
hospital stay, predict its **principal diagnosis** in ICD-10 (CIM-10, ATIH version), as coded
in the French PMSI (DP and, when the DP is a Z code, DR). It is an
[EDS-NLP](https://github.com/aphp/edsnlp) pipeline (`eds.doc_classifier`, edsnlp 0.24.0),
released on the Hugging Face Hub (current version: `v0.1`):
**[`PARTAGES-CU2/cu2-icd10-dp-encoder`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder)**.

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
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-encoder --revision v0.1 \
    --local-dir models/cu2-icd10-dp-encoder
```

The scripts accept the local dir or the repo id, optionally with a revision
(`PARTAGES-CU2/cu2-icd10-dp-encoder@v0.1`, downloaded on first use); the fine-tuning
config needs the local dir. On a machine without internet, copy that folder over.

## Architecture

A BERT-base encoder (see [Training recipe](#training-recipe-of-the-released-model)) reads the
text in sliding windows of 256 tokens; an attention pooler makes one vector per stay; two
classification heads read it:

| Head        | Predicts                                              | Labels |
|-------------|-------------------------------------------------------|-------:|
| `affection` | the disease: the DR if the DP is a Z code, else the DP | 12 950 |
| `mdp`       | the Z code used as DP, or `Z769` = no Z code           |    720 |

`affection` covers all ICD-10 codes valid as DP or DR (Z80-Z99 included: history,
transplant, implant states); `mdp` (*mode de prise en charge*) covers the Z codes valid as DP.

Why two heads: a stay coded "DP = Z51.1 (chemotherapy), DR = C50.9" and a stay coded
"DP = C50.9" describe the same disease; splitting the code lets the model learn the disease
from both.

The PMSI coding is rebuilt from the two predictions:

| `mdp` predicts | DP | DR |
|---|---|---|
| `Z769` (no Z code) | the `affection` code | none |
| a Z code usually coded with a DR (e.g. `Z511`, chemotherapy) | the Z code | the `affection` code |
| a Z code usually coded without a DR (e.g. `Z380`, newborn) | the Z code | none |

"Usually coded with a DR" is read from [`data/dr_rule.json`](data/dr_rule.json): for each Z
code (or, failing that, its 3-character category), whether PMSI stays with this DP mostly
carry a DR. The single code compared with a gold standard is **`target` = DR if any, else
DP**.

## Data format

**You keep your PMSI coding as it is (DP and DR).** The conversion to the two heads is done
by [`scripts/prepare_heads.py`](scripts/prepare_heads.py) before fine-tuning; you never
write `affection` / `mdp` by hand.

One parquet file (or a folder of parquet files), **one row per hospital stay**:

| Column      | Needed for                     | Content |
|-------------|--------------------------------|---------|
| `note_id`   | always                         | unique id of the stay (string or int) |
| `note_text` | always                         | the reports of the stay, in chronological order, separated by a blank line: consultation reports (CRC), discharge summaries (CRH) and operative reports (CR-OP) |
| `dp`        | fine-tuning, evaluation        | PMSI principal diagnosis, ICD-10 code (`C509` or `C50.9`) |
| `dr`        | fine-tuning, evaluation        | PMSI related diagnosis, empty or null when none |
| `acts`      | recommended, always            | the CCAM procedure codes of the stay (list, or a string such as `NFFC004;ZBQK002`); empty when none |

Other column names are fine: pass them with `--dp` / `--dr` / `--acts` (`prepare_heads`) or
`--gold-dp` / `--gold-dr` / `--acts` (`predict`). Inference only needs `note_id`,
`note_text` and, ideally, `acts`.

**Procedures (CCAM acts).** The model was trained on texts whose first line lists the
procedures of the stay, then a blank line, then the reports:

```
Actes (CCAM) : NFFC004 - Méniscectomie latérale ou médiale du genou, par arthroscopie

Compte rendu opératoire ...
```

(`Actes (CCAM) : non renseigné` when the stay has none). With `--acts <column>`,
`prepare_heads` and `predict` build this line from the codes, using the CCAM 2026
nomenclature shipped in [`data/ccam_2026.csv`](data/ccam_2026.csv) (ATIH); texts that already
start with it are left as is. Give the same line at fine-tuning and at inference.

## Inference

```python
import edsnlp
from scripts.heads import load_rule, reconstruct

from scripts.acts import load_ccam, with_acts

nlp = edsnlp.load("models/cu2-icd10-dp-encoder")
rule = load_rule("data/dr_rule.json")
ccam = load_ccam()

text = with_acts("Compte rendu d'hospitalisation ...", ["NFFC004"], ccam)  # CCAM header
doc = nlp(text)
doc._.affection, doc._.mdp                              # raw head outputs
dp, dr = reconstruct(doc._.mdp, doc._.affection, rule)  # PMSI pair
```

On a dataset (multiprocessing and GPU handled by edsnlp):

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-encoder \
    --data stays.parquet --out preds.parquet --acts acts [--gold-dp dp --gold-dr dr] [--topk 10]
```

The output has one row per stay: `affection`, `mdp`, `dp`, `dr`, `target`. With the gold
columns, it also prints the accuracy of `target` against your coding (exact code and first
3 characters).

`--topk 10` adds `candidates` and `scores`: the 10 most likely targets, `target` first.
Both heads are ranked together, by the probability that each code ends up as the target
(an `affection` code is weighted by the chance that `mdp` keeps it; a Z code coded without
DR gets its `mdp` probability), so a Z code that `mdp` ranked second can still appear.

## Fine-tuning on your data

1. **Build the head targets** for a train and a validation split (parquet as in
   [Data format](#data-format), with `dp` and `dr`):

   ```bash
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_train.parquet --out data/my_train --acts acts
   uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
       --src my_dev.parquet --out data/my_dev --acts acts
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
    over the whole ICD-10 by an LLM agent (Mistral Large 4 with ICD-10 / MCO-guide tools and a
    PMSI coding procedure; 80-82 % exact on gold-labelled PARHAF test sets) (4 stays per batch,
    scale 1);
  - ICD-10 labels and synonyms of every code, 168 313 short texts (128 per batch, scale 0.1).
- **Optimisation**: AdamW, 12 000 steps, linear schedules with warm-up, learning rate 3e-5
  for the encoder and 5e-4 for the pooler and heads, gradient clipping 1.0, bf16; label
  smoothing 0.1 on `affection`, plain cross-entropy on `mdp`; seed 42; last checkpoint kept.
- **Initialisation of the heads from the code descriptions**: before training, the pooler
  and the `affection` head start from each code's description instead of random weights
  (row of a code = mean hidden vector of its PMSI label, ICD-10 inclusion notes and
  synonyms; bias = log prior frequency of the code). Four variants were compared on the
  same data (base recipe, this one, an auxiliary 3-character head, masking the PARHAF/CU0
  stream); this one was kept: no significant loss on any test set, and it helps rare codes
  on real stays (codes seen 1-5 times in training: +3.6 points, exact).
- **Versions**: `v0` used labels from a Qwen 27B teacher for the PARHAF / CU0 stream; `v0.1`
  only replaces them by the Mistral Large 4 labels (PARHAF CU2 test set: +2.6 points exact,
  paired over 2 seeds).

## Results

Model `PARTAGES-CU2/cu2-icd10-dp-encoder`, revision `v0.1`. Accuracy of `target` (DR if any,
else DP) with the DR rule; none of these stays were used for training.

| Test set | Stays | Exact | 3 characters | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|
| PARHAF CU2 test set (fictitious patients, revised gold) | 288 | 68.4 % | 82.3 % | 92.0 % | 94.8 % |
| MISTRAL test (synthetic stays) | 14 253 | 92.4 % | 94.4 % | 97.6 % | 97.8 % |

**Physician-reviewed AP-HP stays (most reliable gold).** A sample of AP-HP stays whose PMSI
coding was reviewed by physicians (May 2026), several reviewers for part of them. The text is
built as asked of partner hospitals: CCAM header + consultation reports (CRC), discharge
summaries (CRH) and operative reports (CR-OP) of the stay. Three views of the same set:

- **flattened**: every annotation is a row (a stay reviewed twice counts twice);
- **agreement + single**: one row per stay, keeping stays reviewed once and stays whose
  reviewers agree on the target (stays where they disagree are dropped);
- **agreement**: only stays reviewed by several physicians who agree (the cleanest gold, but
  few stays).

Each view is scored on the whole ICD-10 (**ICD-10**) and on the surgical scope of 3 568 codes
(**surgery**: stays whose gold is one of these codes, predictions restricted to them).

| View | Scope | Stays | Exact | 3 char. | Top-10 | Top-20 | 3 char. top-10 | 3 char. top-20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 40.3 % | 51.0 % | 68.4 % | 73.6 % | 77.7 % | 82.6 % |
| flattened | surgery | 259 | 48.6 % | 56.8 % | 77.2 % | 80.7 % | 82.6 % | 86.1 % |
| agreement + single | ICD-10 | 195 | 40.5 % | 51.8 % | 69.2 % | 75.4 % | 78.5 % | 82.6 % |
| agreement + single | surgery | 167 | 47.9 % | 56.3 % | 76.6 % | 80.2 % | 82.0 % | 85.0 % |
| agreement | ICD-10 | 47 | 68.1 % | 70.2 % | 89.4 % | 91.5 % | 91.5 % | 93.6 % |
| agreement | surgery | 42 | 76.2 % | 76.2 % | 95.2 % | 95.2 % | 95.2 % | 95.2 % |

Before any fine-tuning. Small sets: the 95 % confidence interval is about ±7 points on ~200
stays and ±13 points on ~50. Compared with `v0`, `v0.1` is on par on the large AP-HP test set
(32.4 vs 32.7 % exact; better on rare codes, slightly worse on frequent ones) and ahead on the
reviewed stays where annotators agree (+4 to +6 points, not significant).

Top-k: the gold is among the `candidates` of `scripts.predict --topk k` (the predicted
`target` first, then the most likely targets across both heads).

Note styles and coding habits differ between hospitals: fine-tuning on your own coded
stays is recommended: on AP-HP stays it brought a clear gain over the released model.

## Repository layout

```
configs/finetune.yml     fine-tuning config (edsnlp.train)
scripts/predict.py       batch inference and accuracy
scripts/prepare_heads.py PMSI DP/DR -> affection / mdp targets
scripts/train.py         edsnlp.train + per-head label smoothing
scripts/heads.py         DP/DR <-> heads, DR rule
scripts/acts.py          CCAM procedure header of the input text
data/dr_rule.json        keep-the-DR rule per Z code
data/ccam_2026.csv       CCAM 2026 nomenclature (ATIH): code -> label
```

The previous multi-head baseline (`dp` / `das` / `mdp`, edsnlp fork `classif_head`) is in the
git history (commit `2531454`).

## Contributing

Work on a branch (`feat/…`, `fix/…`, `docs/…`) and open a pull request against `main`. Lint
with `uv run pre-commit install` once, or `uv run ruff check . && uv run ruff format .`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

# Inference & evaluation

Reference: [`notebooks/inference.ipynb`](../notebooks/inference.ipynb).

## Loading a trained model

The synthetic head is excluded at load time (it is only useful during training):

```python
import edsnlp

nlp = edsnlp.load("artifacts/model-last", exclude=["doc_classifier_syn"])
```

## Predicting on a single note

The predicted codes are read off the document's extension attributes (`doc._.dp`, `doc._.das`,
`doc._.mdp`):

```python
text = "Compte rendu d'hospitalisation ..."
doc = nlp(text)

doc._.dp     # predicted principal diagnosis (single ICD-10 code)
doc._.das    # predicted associated diagnoses (list of codes)
doc._.mdp    # predicted "mode de prise en charge"
```

## Scoring a dataset

Read a parquet dataset, map the pipeline over it, and pull predictions + gold labels back into a
pandas DataFrame. Use distinct attribute names for the gold labels (e.g. `dp_gold`) so they
don't collide with the predictions:

```python
import edsnlp

val_data = edsnlp.data.read_parquet(
    "/path/to/val",
    tokenizer=nlp.tokenizer,
    converter="omop",
    doc_attributes={"dp": "dp_gold"},   # load the gold dp into doc._.dp_gold
    shuffle="fragment",
)

val_data = val_data.map_pipeline(nlp)
val_data = val_data.set_processing(num_cpu_workers=8, show_progress=True)

note_nlp = val_data.to_pandas(
    converter="omop",
    doc_attributes=["dp_gold", "dp"],   # gold + prediction
)
```

Then compute metrics with scikit-learn:

```python
from sklearn.metrics import precision_recall_fscore_support

def compute_metric(df, granularity=None):
    df = df.dropna(subset=["dp", "dp_gold"])
    y_true, y_pred = df["dp_gold"], df["dp"]
    if granularity:                     # e.g. granularity=3 → compare 3-char code prefixes
        y_true, y_pred = y_true.str[:granularity], y_pred.str[:granularity]
    for avg in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        print(f"{avg} - P: {p:.4f}, R: {r:.4f}, F1: {f:.4f}")
```

Slicing the code to its first 3 characters (`granularity=3`) scores at the ICD-10 **category**
level instead of the full code, which is a more forgiving metric.

> The metric above is `dp`-only for clarity. The training-time scorer (`eds.doc_classif` in the
> config) reports per-head micro-F1 for `dp`, `das` and `mdp` during validation.

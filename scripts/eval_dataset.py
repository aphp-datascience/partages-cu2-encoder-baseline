"""Score a trained doc_classifier model on a parquet dataset (e.g. PARHAF).

Loads `model-last`, maps the pipeline over the dataset, and compares the predicted `dp`
against the reference `dp`. Reports micro/macro F1 both at the full code and at the
3-character category level, and — importantly — under **two denominators**: over all
documents (codes absent from the training label space count as errors) and over the
subset whose code is in it. Neither is "the" number; say which one you are quoting.

    uv run python scripts/eval_dataset.py \
        --model path/to/artifacts/model-last \
        --data  path/to/parhaf \
        --vocab path/to/valid_labels_dp.pkl
"""

import argparse
import pickle
from pathlib import Path

import edsnlp
from sklearn.metrics import precision_recall_fscore_support


def prf(y_true, y_pred, tag: str) -> None:
    for gran in (None, 3):
        yt = y_true.str[:gran] if gran else y_true
        yp = y_pred.str[:gran] if gran else y_pred
        for avg in ("micro", "macro"):
            p, r, f, _ = precision_recall_fscore_support(
                yt, yp, average=avg, zero_division=0
            )
            lvl = "cat3" if gran else "code"
            print(f"  [{tag}] {lvl} {avg:<5} P={p:.4f} R={r:.4f} F1={f:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--num-cpu", type=int, default=8)
    args = ap.parse_args()

    nlp = edsnlp.load(str(args.model), exclude=["doc_classifier_syn"])
    vocab = set(pickle.loads(args.vocab.read_bytes()))

    data = (
        edsnlp.data.read_parquet(
            str(args.data),
            tokenizer=nlp.tokenizer,
            converter="omop",
            doc_attributes={"dp": "dp_gold"},
        )
        .map_pipeline(nlp)
        .set_processing(num_cpu_workers=args.num_cpu, show_progress=True)
    )

    df = data.to_pandas(converter="omop", doc_attributes=["dp_gold", "dp"])
    df = df.dropna(subset=["dp_gold"])
    n_all = len(df)
    seen = df[df["dp_gold"].isin(vocab)]
    n_unseen = n_all - len(seen)

    print(
        f"\ndocuments: {n_all} | out-of-vocabulary codes (counted as errors): "
        f"{n_unseen} ({100 * n_unseen / n_all:.1f}%)"
    )
    print("\n=== all documents (out-of-vocabulary code = error) ===")
    prf(df["dp_gold"], df["dp"].fillna("__NA__"), "all")
    print("\n=== documents whose code is in the label space ===")
    prf(seen["dp_gold"], seen["dp"].fillna("__NA__"), "seen")


if __name__ == "__main__":
    main()

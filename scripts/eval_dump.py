"""Like `eval_dataset.py`, but ALSO writes out the per-document prediction.

Used to compare the encoder against another source of predictions (a few-shot
LLM, a fine-tuned LLM) **document by document**, which aggregate metrics cannot
do: two systems scoring 30% may be wrong on the same documents or on disjoint
ones, and the conclusion is not the same.

Kept separate from `eval_dataset.py` on purpose: that one runs at the end of
every training job and must not change while a grid is in flight.

    uv run python scripts/eval_dump.py \
        --model path/to/artifacts/model-last \
        --data  path/to/parhaf_patient \
        --vocab path/to/valid_labels_dp.json \
        --out   enc_preds.pkl
"""

import argparse
from pathlib import Path

import edsnlp
from _labels import load_labels
from sklearn.metrics import precision_recall_fscore_support


def prf(y_true, y_pred, tag: str) -> None:
    for gran in (None, 3):
        yt = y_true.str[:gran] if gran else y_true
        yp = y_pred.str[:gran] if gran else y_pred
        for avg in ("micro", "macro"):
            p, r, f, _ = precision_recall_fscore_support(
                yt, yp, average=avg, zero_division=0
            )
            print(
                f"  [{tag}] {'cat3' if gran else 'code'} {avg:<5} "
                f"P={p:.4f} R={r:.4f} F1={f:.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-cpu", type=int, default=8)
    args = ap.parse_args()

    nlp = edsnlp.load(str(args.model), exclude=["doc_classifier_syn"])
    vocab = set(load_labels(args.vocab))

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
    df["dp"] = df["dp"].fillna("__NA__")

    n_all = len(df)
    seen = df[df["dp_gold"].isin(vocab)]
    print(
        f"\ndocuments: {n_all} | out-of-vocabulary codes (counted as errors): "
        f"{n_all - len(seen)}"
    )
    print("\n=== all documents (out-of-vocabulary code = error) ===")
    prf(df["dp_gold"], df["dp"], "all")
    print("\n=== documents whose code is in the label space ===")
    prf(seen["dp_gold"], seen["dp"], "seen")

    df[["note_id", "dp_gold", "dp"]].to_pickle(args.out)
    print(f"\n-> {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()

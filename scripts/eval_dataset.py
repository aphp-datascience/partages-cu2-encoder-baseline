"""Évalue un modèle doc_classifier entraîné sur un jeu parquet (ex : PARHAF).

Charge `model-last`, mappe la pipeline sur le jeu, compare `dp` prédit au `dp` de référence.
Rapporte micro/macro F1 au code complet ET à la catégorie (3 caractères), et surtout les
**deux dénominateurs** : sur tous les documents (les DP absents du vocab d'entraînement
comptent comme erreurs) et sur le seul sous-ensemble à DP connu — la décision assumée.

    python scripts/eval_dataset.py \
        --model $CU2_RUNS/encoder/raw/artifacts/model-last \
        --data  $CU2_DATA/encoder_v3/parhaf \
        --vocab $CU2_DATA/encoder_v3/valid_labels_dp.pkl
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
        f"\ndocuments: {n_all} | DP hors vocab (comptés erreurs): {n_unseen} "
        f"({100 * n_unseen / n_all:.1f}%)"
    )
    print("\n=== tous les documents (DP inconnu = erreur) ===")
    prf(df["dp_gold"], df["dp"].fillna("__NA__"), "all")
    print("\n=== documents à DP connu du vocab ===")
    prf(seen["dp_gold"], seen["dp"].fillna("__NA__"), "seen")


if __name__ == "__main__":
    main()

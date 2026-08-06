"""Comme `eval_dataset.py`, mais écrit AUSSI la prédiction par document.

Sert à comparer l'encodeur à une autre source de prédiction (LLM few-shot, LLM
affiné) **patient par patient**, ce que les métriques agrégées ne permettent
pas : deux systèmes à 30 % peuvent se tromper sur les mêmes patients ou sur des
patients disjoints, et la conclusion n'est pas la même.

Fichier séparé de `eval_dataset.py` à dessein : celui-ci tourne à la fin de
chaque entraînement, on n'y touche pas pendant qu'une grille est en vol.

    python scripts/eval_dump.py --model .../model-last --data .../parhaf_patient \
        --vocab .../valid_labels_dp.pkl --out enc_preds.pkl
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
    df["dp"] = df["dp"].fillna("__NA__")

    n_all = len(df)
    seen = df[df["dp_gold"].isin(vocab)]
    print(
        f"\ndocuments: {n_all} | DP hors vocab (comptés erreurs): {n_all - len(seen)}"
    )
    print("\n=== tous les documents (DP inconnu = erreur) ===")
    prf(df["dp_gold"], df["dp"], "all")
    print("\n=== documents à DP connu du vocab ===")
    prf(seen["dp_gold"], seen["dp"], "seen")

    df[["note_id", "dp_gold", "dp"]].to_pickle(args.out)
    print(f"\n-> {args.out}  ({len(df)} lignes)")


if __name__ == "__main__":
    main()

"""Add the two head targets (`affection`, `mdp`) to a dataset coded with PMSI DP / DR.

    uv run python -m scripts.prepare_heads --model <hf-repo-or-dir> \
        --src my_train.parquet --out data/my_train [--dp dp --dr dr]

Input: parquet with `note_id`, `note_text`, the DP column and (optionally) the DR
column.
Output: `<out>/data.parquet` with `note_id`, `note_text`, `affection`, `mdp`, ready for
`configs/finetune.yml`. A target outside the model's label space becomes None (that
head is masked for the note) and is counted, so the head sizes never change.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.heads import DEFAULT_MDP, resolve_model, split_targets


def model_labels(model: str) -> dict[str, set[str]]:
    path = Path(resolve_model(model))
    vocab = json.loads((path / "doc_classifier" / "vocab.json").read_text())
    return {name: set(h["labels"]) for name, h in vocab["heads"].items()}


def norm(code) -> str | None:
    if not isinstance(code, str) or not code.strip():
        return None
    return code.replace(".", "").strip().upper()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dp", default="dp", help="column of the PMSI DP")
    ap.add_argument("--dr", default="dr", help="column of the PMSI DR (may be absent)")
    ap.add_argument(
        "--acts",
        help="column of CCAM codes: the 'Actes (CCAM)' header is added to note_text",
    )
    args = ap.parse_args()

    labels = model_labels(args.model)
    df = pd.read_parquet(args.src)
    dr = df[args.dr] if args.dr in df else pd.Series([None] * len(df), index=df.index)
    heads = [
        split_targets(norm(p), norm(r)) for p, r in zip(df[args.dp], dr, strict=True)
    ]
    mdp = [m if m in labels["mdp"] else None for m, _ in heads]
    aff = [a if a in labels["affection"] else None for _, a in heads]
    oov = Counter()
    for (_m, a), m2, a2 in zip(heads, mdp, aff, strict=True):
        oov["mdp"] += m2 is None
        oov["affection"] += a is not None and a2 is None

    out = df[["note_id", "note_text"]].assign(affection=aff, mdp=mdp)
    if args.acts:  # add the CCAM header, as in training
        from scripts.acts import load_ccam, with_acts

        ccam = load_ccam()
        out["note_text"] = [
            with_acts(t, a, ccam)
            for t, a in zip(out["note_text"], df[args.acts], strict=True)
        ]
    out = out[out["affection"].notna() | out["mdp"].notna()]
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "data.parquet", index=False)
    print(
        f"{len(out)}/{len(df)} notes -> {args.out / 'data.parquet'} ; "
        f"DP in Z: {sum(m not in (None, DEFAULT_MDP) for m in mdp)} ; "
        f"affection masked: {sum(a is None for a in aff)} ; "
        f"outside the label space (masked): mdp {oov['mdp']}, "
        f"affection {oov['affection']}"
    )


if __name__ == "__main__":
    main()

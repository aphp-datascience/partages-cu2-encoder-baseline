"""Predict the principal diagnosis of clinical notes with the CU2 two-head encoder.

    uv run python -m scripts.predict --model <hf-repo-or-dir> --data notes.parquet \
        --out preds.parquet [--rule data/dr_rule.json] [--gold-dp dp --gold-dr dr]

Input: a parquet file (or a folder of parquet files) with `note_id` and `note_text`.
Output: one row per note with the raw head outputs (`affection`, `mdp`), the
reconstructed PMSI pair (`dp`, `dr`) and `target` = DR if any, else DP. With
`--topk K`, also `candidates` (and `scores`): the K most likely targets, ranked
across both heads (`scripts.heads.candidates`), the first one being `target`.
If gold columns are given, also prints the exact-match and 3-character accuracy of
`target` (and the top-K accuracy) against the gold DR-else-DP.
"""

import argparse
from pathlib import Path

import edsnlp
import pyarrow.parquet as pq

from scripts.heads import (
    DEFAULT_MDP,
    candidates,
    load_rule,
    reconstruct,
    resolve_model,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local dir")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--rule",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dr_rule.json",
        help="keep-the-DR rule per Z code; 'none' always keeps the affection as DR",
    )
    ap.add_argument("--gold-dp", help="column holding the gold PMSI DP (optional)")
    ap.add_argument("--gold-dr", help="column holding the gold PMSI DR (optional)")
    ap.add_argument("--topk", type=int, default=0, help="also output K candidates")
    ap.add_argument("--num-cpu", type=int, default=4, help="CPU workers (tokenisation)")
    args = ap.parse_args()
    rule = None if str(args.rule) == "none" else load_rule(args.rule)

    files = [args.data] if args.data.is_file() else sorted(args.data.rglob("*.parquet"))
    cols = set(pq.read_schema(files[0]).names)
    missing = {"note_id", "note_text"} - cols
    if missing:
        raise SystemExit(f"{args.data}: missing columns {sorted(missing)}")
    gold = {c: f"gold_{c}" for c in (args.gold_dp, args.gold_dr) if c}
    for c in gold:
        if c not in cols:
            raise SystemExit(f"{args.data}: no column {c!r}")

    nlp = edsnlp.load(resolve_model(args.model))
    if args.topk:  # keep the probabilities of each head's top labels
        pipe = nlp.get_pipe("doc_classifier")
        pipe.store_probs, pipe.probs_topk = True, max(args.topk, 20)
    docs = (
        edsnlp.data.read_parquet(
            str(args.data),
            tokenizer=nlp.tokenizer,
            converter="omop",
            doc_attributes=gold,
        )
        .map_pipeline(nlp)
        .set_processing(num_cpu_workers=args.num_cpu, show_progress=True)
    )
    df = docs.to_pandas(
        converter="omop",
        doc_attributes=[
            *gold.values(),
            "affection",
            "mdp",
            *(["prob"] * bool(args.topk)),
        ],
    )

    pairs = [
        reconstruct(m, a, rule) for m, a in zip(df["mdp"], df["affection"], strict=True)
    ]
    df["dp"] = [p[0] for p in pairs]
    df["dr"] = [p[1] for p in pairs]
    df["target"] = df["dr"].fillna(df["dp"])
    keep = ["note_id", "affection", "mdp", "dp", "dr", "target", *gold.values()]
    if args.topk:
        cands = [
            candidates(
                t,
                (p or {}).get("affection") or {},
                (p or {}).get("mdp") or {},
                rule,
                args.topk,
            )
            for t, p in zip(df["target"], df["prob"], strict=True)
        ]
        df["candidates"] = ["|".join(c for c, _ in cs) for cs in cands]
        df["scores"] = ["|".join(f"{s:.4g}" for _, s in cs) for cs in cands]
        keep += ["candidates", "scores"]
    out = df[keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_z = int((out["mdp"] != DEFAULT_MDP).sum())
    print(f"{len(out)} notes -> {args.out} ({n_z} with a Z code as DP)")

    if args.gold_dp:
        g_dp = out[gold[args.gold_dp]]
        g_dr = out[gold[args.gold_dr]] if args.gold_dr else None
        g = g_dp if g_dr is None else g_dr.where(g_dr.notna() & (g_dr != ""), g_dp)
        g = g.str.replace(".", "", regex=False).str.strip().str.upper()
        s = g.notna()
        exact = (out.loc[s, "target"] == g[s]).mean()
        cat3 = (out.loc[s, "target"].str[:3] == g[s].str[:3]).mean()
        msg = (
            f"target (DR else DP), n={int(s.sum())}: exact {exact:.2%}, cat3 {cat3:.2%}"
        )
        if args.topk:
            top = [
                gg in c.split("|")
                for gg, c in zip(g[s], out.loc[s, "candidates"], strict=True)
            ]
            msg += f", top-{args.topk} {sum(top) / len(top):.2%}"
        print(msg)


if __name__ == "__main__":
    main()

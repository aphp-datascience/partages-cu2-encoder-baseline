"""`edsnlp.train` with per-head label smoothing (edsnlp 0.24 has no such option).

    uv run python -m scripts.train --config configs/finetune.yml --seed 42
    accelerate launch --multi_gpu -m scripts.train --config configs/finetune.yml

LABEL_SMOOTHING (default 0.1) is applied to the single-label heads listed in LS_HEADS
(default `affection`, as when the released model was trained; the `mdp` head uses plain
cross-entropy). LABEL_SMOOTHING=0 gives the plain loss. Nothing is written into the
saved model: it reloads with plain `edsnlp.load`.
"""

import functools
import os
import runpy
import sys

import torch

EPS = float(os.environ.get("LABEL_SMOOTHING", "0.1") or 0)
LS_HEADS = {h for h in os.environ.get("LS_HEADS", "affection").split(",") if h}
_LOGGED: set = set()


def install() -> None:
    from edsnlp.pipes.trainable.doc_classifier.doc_classifier import (
        TrainableDocClassifier,
    )
    from edsnlp.pipes.trainable.doc_classifier.heads import DocClassifierHead

    loss, forward = DocClassifierHead.compute_loss, TrainableDocClassifier.forward

    # A head does not know its name: tag the heads before each forward (not in
    # __init__, whose signature confit validates).
    @functools.wraps(forward)
    def forward_named(self, *args, **kwargs):
        for name, head in self.heads.items():
            head._ls_name = name
        return forward(self, *args, **kwargs)

    def compute_loss_ls(self, logits: torch.Tensor, target: torch.Tensor):
        name = getattr(self, "_ls_name", None)
        if self.multi_label or (LS_HEADS and name not in LS_HEADS):
            return loss(self, logits, target)
        if name not in _LOGGED:
            _LOGGED.add(name)
            print(f"[train] label smoothing {EPS} on head {name}", flush=True)
        weights = self.class_weights
        weights = weights.to(logits) if weights is not None else None
        return torch.nn.functional.cross_entropy(
            logits, target.to(logits.device), weight=weights, label_smoothing=EPS
        )

    TrainableDocClassifier.forward = forward_named
    DocClassifierHead.compute_loss = compute_loss_ls


if __name__ == "__main__":
    if EPS > 0:
        install()
    sys.argv[0] = "edsnlp.train"
    runpy.run_module("edsnlp.train", run_name="__main__", alter_sys=True)

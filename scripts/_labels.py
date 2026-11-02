"""Read a label file the way `eds.doc_classifier` does: `.txt` = one label per line,
anything else = JSON. Pickles are no longer accepted by the pipe, so not here either."""

from __future__ import annotations

import json
from pathlib import Path


def load_labels(path: str | Path):
    path = Path(path)
    if path.suffix == ".txt":
        return [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())

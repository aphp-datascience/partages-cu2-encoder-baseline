"""CCAM procedure header, written as the first line of every training text.

The model was trained on texts starting with

    Actes (CCAM) : NEQK010 - Radiographie de l'articulation coxofémorale ... ; ...

then a blank line, then the reports of the stay ("non renseigné" when the stay has no
procedure). Give your stays the same first line for the same input distribution.
`data/ccam_2026.csv` is the CCAM nomenclature 2026 (ATIH): code -> label.
"""

import csv
import re
from pathlib import Path

CCAM_CODE = re.compile(r"[A-Z]{4}\d{3}")
HEADER = "Actes (CCAM) :"
DICT = Path(__file__).resolve().parent.parent / "data" / "ccam_2026.csv"


def load_ccam(path: Path = DICT) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["code"]: row["libelle"] for row in csv.DictReader(handle)}


def parse_codes(value) -> list[str]:
    """A list/array of codes or a string ("AAFA001;ZCQK002", "AAFA001 ZCQK002", ...)."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    items = (
        value
        if isinstance(value, (list, tuple)) or hasattr(value, "tolist")
        else [value]
    )
    return [m for item in items for m in CCAM_CODE.findall(str(item).upper())]


def header(codes: list[str], labels: dict[str, str]) -> str:
    """Codes absent from the nomenclature are dropped (as in training)."""
    acts = [f"{c} - {labels[c]}" for c in dict.fromkeys(codes) if c in labels]
    return f"{HEADER} " + (" ; ".join(acts) if acts else "non renseigné")


def with_acts(text: str, codes, labels: dict[str, str]) -> str:
    """Prepend the header, unless the text already starts with one."""
    if text.lstrip().startswith(HEADER):
        return text
    return header(parse_codes(codes), labels) + "\n\n" + text.strip()

"""Science-corpus filtering helpers: group normalization, doc hashing, chemistry drop."""

import hashlib

DEFAULT_GROUPS = [
    "life_sciences",
    "physical_sciences",
    "mathematics",
    "environmental",
]

CHEM_MARKERS = [
    "stoichiometry",
    "molar mass",
    "chemical formula",
    "chemical equation",
    "periodic table",
    "electronegativity",
    "oxidation number",
    "titration",
    "molarity",
    "mole ratio",
    "hydrochloric acid",
    "sulfuric acid",
    "sodium chloride",
    "valence electron",
]


def _norm_groups(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [g.strip() for g in raw.split(",") if g.strip()]
    return [g for g in raw if g]


def _doc_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _marker_hits(text: str) -> int:
    low = text.lower()
    hits = 0
    for m in CHEM_MARKERS:
        if m in low:
            hits += 1
    return hits
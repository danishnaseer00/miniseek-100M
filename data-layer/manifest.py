"""Manifest persistence for the corpus builder (resume tracking)."""

import json
import os


def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def _new_manifest() -> dict:
    return {"shards_done": [], "train_tokens_est": 0, "train_docs": 0,
            "val_tokens_est": 0, "val_docs": 0, "group_hits": {}, "chem_dropped": 0}


def _load_manifest(out_dir: str) -> dict:
    p = _manifest_path(out_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return _new_manifest()


def _save_manifest(out_dir: str, man: dict):
    with open(_manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
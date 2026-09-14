"""HuggingFace shard discovery / download for the corpus builder."""

import os

from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset

from filters import _norm_groups

DATASET = "mdonigian/fineweb-edu-curated"


def _list_shards(dataset: str) -> list:
    files = list_repo_files(dataset, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def _download_shard(shard: str, dataset: str, shard_cache: str) -> str:
    """Download one parquet shard into the shard cache (resumable)."""
    local = hf_hub_download(
        dataset,
        filename=shard,
        repo_type="dataset",
        cache_dir=shard_cache,
    )
    print(f"  shard ready: {local}")
    return local


def enumerate_labels(dataset=DATASET, shard_cache="./data/shard_cache", max_rows=20000):
    shard = _list_shards(dataset)[0]
    local = _download_shard(shard, dataset, shard_cache)
    ds = load_dataset("parquet", data_files=local, split="train", streaming=True)
    counts = {}
    n = 0
    for row in ds:
        n += 1
        for g in _norm_groups(row.get("assigned_groups")):
            counts[g] = counts.get(g, 0) + 1
        if n >= max_rows:
            break
    print(f"Observed assigned_groups over {n} rows:")
    for g, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {g:40s} {c:>8}  ({100*c/max(n,1):5.1f}% of rows)")
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from huggingface_hub import list_repo_files, hf_hub_download 
from datasets import load_dataset 

DATASET = "mdonigian/fineweb-edu-curated"
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


def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def _load_manifest(out_dir: str) -> dict:
    p = _manifest_path(out_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"shards_done": [], "train_tokens_est": 0, "train_docs": 0,
            "val_tokens_est": 0, "val_docs": 0, "group_hits": {}, "chem_dropped": 0}


def _save_manifest(out_dir: str, man: dict):
    with open(_manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)


def build(
    out_dir: str = "./data/science1b",
    groups: list = None,
    max_train_tokens: int = 1_000_000_000,
    val_ratio: float = 0.02,
    exclude_chemistry: bool = True,
    chemistry_threshold: int = 2,
    min_doc_tokens: int = 32,
    max_doc_tokens: int = 16_000,
    resume: bool = False,
    max_rows: int = None,
    log_every: int = 5000,
    dataset: str = DATASET,
    shard_cache: str = "./data/shard_cache",
):
    groups = groups or DEFAULT_GROUPS
    target = set(groups)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(shard_cache, exist_ok=True)

    man = _load_manifest(out_dir) if resume else {
        "shards_done": [], "train_tokens_est": 0, "train_docs": 0,
        "val_tokens_est": 0, "val_docs": 0, "group_hits": {}, "chem_dropped": 0}
    done_flag = os.path.join(out_dir, "done.flag")

    if resume and man.get("train_tokens_est", 0) >= max_train_tokens:
        print(f"Token budget already met ({man['train_tokens_est']:,}). Nothing to do.")
        _save_manifest(out_dir, man)
        return man

    shards = _list_shards(dataset)
    print(f"Dataset {dataset}: {len(shards)} shards")
    print(f"Target groups: {sorted(target)} | exclude_chemistry={exclude_chemistry} "
          f"(threshold={chemistry_threshold}) | train target={max_train_tokens:,} tokens")
    print(f"Shard cache: {shard_cache}")

    done = set(man["shards_done"])
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "validation.jsonl")
    t0 = time.time()

    for shard in shards:
        if shard in done:
            print(f"  shard {shard}: already done, skip")
            continue

        print(f"\n=== shard {shard} ===")
        local = _download_shard(shard, dataset, shard_cache)
        shard_out = os.path.join(out_dir, f"shard_{os.path.splitext(shard)[0]}.jsonl")

        # Merge previously completed shards into the aggregate files on first pass.
        if not man["shards_done"]:
            print("  (no completed shards yet - building aggregate files fresh)")
            for p in (train_path, val_path):
                if os.path.exists(p):
                    os.remove(p)
        merge_previous = len(man["shards_done"]) > 0 and not os.path.exists(shard_out)

        rows_seen = 0
        shard_train = 0
        shard_val = 0
        with open(shard_out, "w", encoding="utf-8") as fshard:
            ds = load_dataset("parquet", data_files=local, split="train", streaming=True)
            for row in ds:
                rows_seen += 1
                if max_rows is not None and rows_seen > max_rows:
                    break

                text = row.get("text")
                if not text or not text.strip():
                    continue

                matched = sorted(set(_norm_groups(row.get("assigned_groups"))) & target)
                if not matched:
                    continue

                tok_count = int(row.get("token_count") or 0)
                if tok_count <= 0:
                    tok_count = max(1, len(text) // 4)
                if tok_count < min_doc_tokens or tok_count > max_doc_tokens:
                    continue

                if exclude_chemistry and "physical_sciences" in matched and _marker_hits(text) >= chemistry_threshold:
                    man["chem_dropped"] += 1
                    continue

                for g in matched:
                    man["group_hits"][g] = man["group_hits"].get(g, 0) + 1

                h = _doc_hash(str(row.get("text_hash") or row.get("url") or text))
                is_val = (h % 100) < int(val_ratio * 100)

                record = {
                    "url": row.get("url"),
                    "text": text,
                    "groups": matched,
                    "complexity_score": row.get("complexity"),
                    "token_count": tok_count,
                    "token_count_known": int(row.get("token_count") or 0) > 0,
                }
                fshard.write(json.dumps(record, ensure_ascii=False) + "\n")
                if is_val:
                    man["val_docs"] += 1
                    man["val_tokens_est"] += tok_count
                    shard_val += 1
                else:
                    man["train_docs"] += 1
                    man["train_tokens_est"] += tok_count
                    shard_train += 1

                if man["train_tokens_est"] >= max_train_tokens:
                    print(f"  token budget reached ({man['train_tokens_est']:,}) mid-shard {shard}; stopping.")
                    break

                if rows_seen % log_every == 0:
                    fps = man["train_tokens_est"] / max(1, time.time() - t0)
                    eta = (max_train_tokens - man["train_tokens_est"]) / fps if fps > 0 else float("inf")
                    print(f"  rows {rows_seen:,} | train {man['train_tokens_est']/1e6:,.1f}M/{max_train_tokens/1e6:,.0f}M "
                          f"({100*man['train_tokens_est']/max_train_tokens:.1f}%) | "
                          f"val {man['val_tokens_est']/1e6:,.1f}M | {fps/1e3:.2f}k tok/s | ETA {eta/3600:.1f}h")

        man["shards_done"].append(shard)
        _save_manifest(out_dir, man)
        print(f"  shard {shard} done: +{shard_train:,} train / +{shard_val:,} val docs")

        if man["train_tokens_est"] >= max_train_tokens:
            break

    # Merge all per-shard files into train/validation aggregates.
    agg_train = []
    agg_val = []
    print("\nMerging shards -> train.jsonl / validation.jsonl")
    for shard in man["shards_done"]:
        p = os.path.join(out_dir, f"shard_{os.path.splitext(shard)[0]}.jsonl")
        if not os.path.exists(p):
            print(f"  WARNING: missing {p}; skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                (agg_val if (h := _doc_hash(str(rec.get("url") or rec["text"]))) % 100 < int(val_ratio * 100) else agg_train).append(rec)
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in agg_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for rec in agg_val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  train.jsonl: {len(agg_train):,} docs  validation.jsonl: {len(agg_val):,} docs")

    stats = {
        "dataset": dataset,
        "shards_total": len(shards),
        "shards_done": man["shards_done"],
        "target_groups": sorted(target),
        "exclude_chemistry": exclude_chemistry,
        "chemistry_threshold": chemistry_threshold,
        "train_docs": len(agg_train),
        "val_docs": len(agg_val),
        "train_tokens_est": man["train_tokens_est"],
        "val_tokens_est": man["val_tokens_est"],
        "group_hits": man["group_hits"],
        "chem_dropped": man["chem_dropped"],
        "duration_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    if man["train_tokens_est"] >= max_train_tokens:
        open(done_flag, "w").close()

    print("\nBuild complete.")
    print(json.dumps(stats, indent=2))
    return man


def encode_range(args):
    """Process-pool worker: read one byte-range chunk and tokenize each doc line."""
    import numpy as np
    from src.tokenizer import Tokenizer

    path, start_byte, end_byte, tokenizer_name, max_length = args
    tokenizer = Tokenizer(tokenizer_name, max_length=max_length)
    pieces = []
    lens = []
    with open(path, "rb") as f:
        f.seek(start_byte)
        data = f.read(end_byte - start_byte)
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line.decode("utf-8", errors="replace")).get("text")
        except Exception:
            continue
        if not text or not text.strip():
            continue
        toks = tokenizer.encode(text, add_special_tokens=True, truncate=False)
        if len(toks) <= 1:
            continue
        pieces.append(np.asarray(toks, dtype=np.int32))
        lens.append(len(toks))
    return (pieces, lens)


def tokenize_corpus(out_dir: str = "./data/science1b", tokenizer_name: str = "/data/gpt2-tokenizer",
                    max_length: int = 1024, workers: int = 16):
    """Produce train.npz / validation.npz (same layout as src.data.WikiTextDataset).

    Workers read the source file by BYTE RANGE — the parent only passes a small
    (path, start, end, tokenizer_name, max_length) tuple per chunk, never the
    full line strings. This keeps parent RAM under ~2 GB regardless of corpus size.
    Tokenizer is loaded from a LOCAL directory on the volume so workers make zero
    huggingface.co calls and cannot hit HTTP 429 rate limits.

    encode_range lives in this same module so multiprocessing.Pool can pickle it.
    """
    import numpy as np
    from multiprocessing import Pool

    CHUNK_BYTES = 256 * 1024 * 1024  # ~256 MB per worker task

    for split in ("train", "validation"):
        src = os.path.join(out_dir, f"{split}.jsonl")
        out = os.path.join(out_dir, f"{split}.npz")
        if not os.path.exists(src):
            print(f"SKIP {split}: no {src}")
            continue

        # Build byte-offset index: cumulative byte position after each line
        print(f"building byte index for {split}...")
        offsets = []
        with open(src, "rb") as f:
            offsets.append(f.tell())
            for line in f:
                if not line.endswith(b"\n"):
                    break
                offsets.append(f.tell())
        total_bytes = offsets[-1] if offsets else 0
        print(f"{split}: {len(offsets)-1:,} docs, {total_bytes/1e9:.2f} GB")

        # Chunk by byte ranges so each worker reads a contiguous ~256 MB slice
        chunks = []
        i = 0
        while i < len(offsets) - 1:
            chunk_start = offsets[i]
            j = i + 1
            while j < len(offsets) and offsets[j] - chunk_start < CHUNK_BYTES:
                j += 1
            chunks.append((src, chunk_start, offsets[j - 1] if j < len(offsets) else total_bytes,
                           tokenizer_name, max_length))
            i = j
        print(f"split into {len(chunks)} chunks (~{CHUNK_BYTES/1e6:.0f} MB each), tokenizing with {workers} workers...")

        pieces = []
        lens = []
        with Pool(workers) as pool:
            for chunk_pieces, chunk_lens in pool.imap(encode_range, chunks):
                pieces.extend(chunk_pieces)
                lens.extend(chunk_lens)

        flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
        doc_starts = np.cumsum(np.concatenate([[0], np.asarray(lens, dtype=np.int64)])) if lens else np.zeros(1, dtype=np.int64)
        np.savez(out, tokens=flat, doc_starts=doc_starts)
        print(f"Wrote {out}: {len(pieces):,} docs, {len(flat):,} real GPT-2 tokens")
    return True


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


def main():
    ap = argparse.ArgumentParser(description="Build the Phase-3 science corpus")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out", default="./data/science1b")
    ap.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    ap.add_argument("--max-train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--no-exclude-chemistry", action="store_true")
    ap.add_argument("--chemistry-threshold", type=int, default=2)
    ap.add_argument("--min-doc-tokens", type=int, default=32)
    ap.add_argument("--max-doc-tokens", type=int, default=16_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None, help="rows per shard (dry-run knob)")
    ap.add_argument("--shard-cache", default="./data/shard_cache")
    ap.add_argument("--enumerate-labels", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenize", action="store_true", help="build train/validation .npz token caches")
    ap.add_argument("--log-every", type=int, default=5000)
    args = ap.parse_args()

    if args.enumerate_labels:
        enumerate_labels(dataset=args.dataset, shard_cache=args.shard_cache)
        return

    if args.dry_run:
        args.out = os.path.join(args.out, "_dryrun")
        args.max_train_tokens = 10_000_000
        args.max_rows = args.max_rows or 20_000
        args.resume = False
        print("DRY RUN: 10M-token sample with max-rows limit, no train.jsonl kept for training.")

    build(
        out_dir=args.out,
        groups=args.groups,
        max_train_tokens=args.max_train_tokens,
        val_ratio=args.val_ratio,
        exclude_chemistry=not args.no_exclude_chemistry,
        chemistry_threshold=args.chemistry_threshold,
        min_doc_tokens=args.min_doc_tokens,
        max_doc_tokens=args.max_doc_tokens,
        resume=args.resume,
        max_rows=args.max_rows,
        log_every=args.log_every,
        dataset=args.dataset,
        shard_cache=args.shard_cache,
    )

    if args.tokenize:
        tokenize_corpus(out_dir=args.out)


if __name__ == "__main__":
    main()
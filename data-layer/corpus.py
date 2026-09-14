import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import load_dataset

from filters import (_doc_hash, _norm_groups, _marker_hits)
from manifest import _load_manifest, _new_manifest, _save_manifest
from shards import DATASET, _list_shards, _download_shard
from tokenization import tokenize_corpus

DEFAULT_GROUPS = [
    "life_sciences",
    "physical_sciences",
    "mathematics",
    "environmental",
]


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

    man = _load_manifest(out_dir) if resume else _new_manifest()
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


def main():
    ap = argparse.ArgumentParser(description="Build the science corpus")
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
        from shards import enumerate_labels
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
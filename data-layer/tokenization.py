"""Tokenize corpus .jsonl into .npz caches (same layout as src.data.WikiTextDataset)."""

import os

import numpy as np
from multiprocessing import Pool

from src.tokenize_workers import encode_range


def tokenize_corpus(out_dir: str = "./data/science1b", tokenizer_name: str = "/data/gpt2-tokenizer",
                    max_length: int = 1024, workers: int = 16):
    """Produce train.npz / validation.npz.

    Workers read the source file by BYTE RANGE — the parent only passes a small
    (path, start, end, tokenizer_name, max_length) tuple per chunk, never the
    full line strings. This keeps parent RAM under ~2 GB regardless of corpus size.
    Tokenizer is loaded from a LOCAL directory so workers make zero
    huggingface.co calls and cannot hit HTTP 429 rate limits.
    """
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
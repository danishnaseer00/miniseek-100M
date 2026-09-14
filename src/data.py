import json
import os
from typing import Optional, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class WikiTextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
        cache_dir: Optional[str] = "./data",
        max_docs: Optional[int] = None,
        dataset: str = "wikitext-103-raw-v1",
        corpus_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.dataset = dataset
        self.corpus_dir = corpus_dir

        tok_name = getattr(tokenizer, "name", "tok")

        if corpus_dir:
            os.makedirs(corpus_dir, exist_ok=True)
            self.cache_path = os.path.join(corpus_dir, f"{split}.npz")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            cache_prefix = "wikitext103" if dataset == "wikitext-103-raw-v1" else dataset.replace("/", "_")
            self.cache_path = os.path.join(
                cache_dir, f"{cache_prefix}_{split}_{tok_name}_{max_length}.npz"
            )

        self.flat_tokens, self.doc_starts = self._load_or_tokenize(cache_dir, tok_name, max_docs)

        self.num_docs = len(self.doc_starts) - 1
        self.doc_lengths = np.diff(self.doc_starts).astype(np.int64)
        self._doc_order = np.arange(self.num_docs, dtype=np.int64)
        self.set_epoch(0)

        print(f"Documents: {self.num_docs:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Sequences (max_length={max_length}): {self.num_sequences:,}")
        print(f"Token cache: {self.cache_path}")

    def _load_or_tokenize(self, cache_dir: str, tok_name: str, max_docs: Optional[int]):
        if max_docs is None and os.path.exists(self.cache_path):
            print(f"Loading token cache: {self.cache_path}")
            data = np.load(self.cache_path)
            return data["tokens"], data["doc_starts"]

        if self.corpus_dir:
            jsonl_path = os.path.join(self.corpus_dir, f"{self.split}.jsonl")
            print(f"Loading local corpus: {jsonl_path}")
            docs: List[str] = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    docs.append(json.loads(line)["text"])
        else:
            print(f"Loading WikiText-103 {self.split} split...")
            dataset = load_dataset("wikitext", self.dataset, split=self.split, cache_dir=cache_dir)
            docs = dataset["text"]

        pieces: List[np.ndarray] = []
        lens: List[int] = []
        for text in docs:
            if not text or not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) <= 1:
                continue
            pieces.append(np.asarray(tokens, dtype=np.int32))
            lens.append(len(tokens))
            if max_docs is not None and len(pieces) >= max_docs:
                print(f"Stopped at {max_docs} documents (max_docs={max_docs})")
                break

        flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
        doc_starts = np.cumsum(np.concatenate([[0], np.asarray(lens, dtype=np.int64)])) if lens else np.zeros(1, dtype=np.int64)

        if max_docs is None:
            np.savez(self.cache_path, tokens=flat, doc_starts=doc_starts)
            print(f"Token cache saved: {self.cache_path}")

        return flat, doc_starts

    def set_epoch(self, epoch: int):
        rng = np.random.RandomState(epoch)
        rng.shuffle(self._doc_order)

        lens = self.doc_lengths[self._doc_order]
        self.doc_offsets = np.concatenate([[0], np.cumsum(lens)])
        self.total_tokens = int(self.doc_offsets[-1])
        self.num_sequences = self.total_tokens // self.max_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length + 1

        chunk = self._get_tokens(start, end)
        n_real = len(chunk)

        if n_real < self.max_length + 1:
            pad = np.full(self.max_length + 1 - n_real, self.tokenizer.eos_token_id, dtype=np.int32)
            chunk = np.concatenate([chunk, pad])

        input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
        targets = torch.from_numpy(chunk[1:].astype(np.int64))

        if n_real <= self.max_length:
            targets[n_real - 1 :] = -1

        return {"input_ids": input_ids, "targets": targets}

    def _get_tokens(self, start: int, end: int) -> np.ndarray:
        di = int(np.searchsorted(self.doc_offsets, start, side="right")) - 1
        di = max(0, min(di, self.num_docs - 1))

        pieces = []
        cur = start
        while cur < end:
            if di >= self.num_docs:
                break
            real = int(self._doc_order[di])
            goff = int(self.doc_offsets[di])
            glen = int(self.doc_starts[real + 1] - self.doc_starts[real])

            seg_start = max(0, cur - goff)
            seg_end = min(glen, seg_start + (end - cur))
            if seg_end > seg_start:
                g0 = int(self.doc_starts[real]) + seg_start
                g1 = int(self.doc_starts[real]) + seg_end
                pieces.append(self.flat_tokens[g0:g1])
                cur += int(seg_end - seg_start)
            di += 1

        if not pieces:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(pieces)


def create_dataloader(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 0,
    cache_dir: Optional[str] = "./data",
    max_docs: Optional[int] = None,
    dataset: str = "wikitext-103-raw-v1",
    corpus_dir: Optional[str] = None,
) -> DataLoader:
    dataset_obj = WikiTextDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        cache_dir=cache_dir,
        max_docs=max_docs,
        dataset=dataset,
        corpus_dir=corpus_dir,
    )

    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset_stats(split: str = "train", cache_dir: str = "./data", dataset: str = "wikitext-103-raw-v1") -> Dict:
    ds = load_dataset("wikitext", dataset, split=split, cache_dir=cache_dir)

    total_chars = sum(len(text) for text in ds["text"])
    total_docs = len([text for text in ds["text"] if len(text.strip()) > 0])
    total_words = sum(len(text.split()) for text in ds["text"])

    return {
        "split": split,
        "total_chars": total_chars,
        "total_docs": total_docs,
        "total_words": total_words,
    }
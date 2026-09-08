import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Dict, List
import os

class WikiTextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
        cache_dir: Optional[str] = "./data",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
        
        self.texts = [text for text in dataset["text"] if len(text.strip()) > 0]
        
        print(f"Tokenizing {len(self.texts)} texts...")
        self.tokens = []
        for text in self.texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            self.tokens.extend(tokens)
        
        print(f"Total tokens: {len(self.tokens):,}")
        
        self.num_sequences = len(self.tokens) // max_length
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length + 1
        
        chunk = self.tokens[start:end]
        
        if len(chunk) < self.max_length + 1:
            chunk = chunk + [self.tokenizer.eos_token_id] * (self.max_length + 1 - len(chunk))
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {"input_ids": input_ids, "targets": targets}


def create_dataloader(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = "./data",
) -> DataLoader:
    dataset = WikiTextDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        cache_dir=cache_dir,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset_stats(tokenizer, split: str = "train", cache_dir: str = "./data") -> Dict:
    from datasets import load_dataset
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
    
    total_chars = sum(len(text) for text in dataset["text"])
    total_docs = len([text for text in dataset["text"] if len(text.strip()) > 0])
    
    return {
        "split": split,
        "total_chars": total_chars,
        "total_docs": total_docs,
        "total_words": total_chars // 5,
    }

from datasets import load_dataset
import os


def download_wikitext103(cache_dir: str = "./data"):
    print("Downloading WikiText-103...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir)
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", cache_dir=cache_dir)
    test_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", cache_dir=cache_dir)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_chars = sum(len(text) for text in train_dataset["text"])
    print(f"Train characters: {train_chars:,}")
    print(f"Estimated train tokens: {train_chars // 4:,}")
    
    print("\nWikiText-103 downloaded successfully!")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    download_wikitext103()

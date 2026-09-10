import torch
import yaml
import argparse
from typing import Optional
from .model import create_model
from .tokenizer import Tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    vocab_size: Optional[int] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    if vocab_size is None:
        vocab_size = config.get("vocab_size", 50257)

    model = create_model({
        "vocab_size": vocab_size,
        "dim": config.get("dim", 512),
        "n_layers": config.get("n_layers", 8),
        "n_heads": config.get("n_heads", 8),
        "mlp_hidden_dim": config.get("mlp_hidden_dim", 1376),
        "max_seq_len": config.get("max_seq_len", 2048),
        "dropout": 0.0,
        "tie_embeddings": config.get("tie_embeddings", True),
    })
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    device: str = "cuda",
) -> str:
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def interactive_generation(
    checkpoint_path: str,
    tokenizer_name: str = "gpt2",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cuda",
):
    print("Loading model...")
    tokenizer = Tokenizer(tokenizer_name, max_length=2048)
    model, config = load_model_from_checkpoint(checkpoint_path, device, vocab_size=tokenizer.vocab_size)
    
    print("\nMiniseek Interactive Generation")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == "quit":
            break
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        
        print(f"\nGenerated:\n{generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    interactive_generation(
        checkpoint_path=args.checkpoint,
        tokenizer_name=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

import torch
from typing import Optional
from .model import create_model


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
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> str:
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text

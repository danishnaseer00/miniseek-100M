import torch
from src.model import create_model


def test_model_forward():
    config = {
        "vocab_size": 50257,
        "dim": 256,
        "n_layers": 6,
        "n_heads": 8,
        "mlp_hidden_dim": 704,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "tie_embeddings": True,
    }
    
    model = create_model(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Expected: ~{config['dim']**2 * 12 / 1e6:.1f}M")
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, targets)
    
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config["vocab_size"])
    assert outputs["loss"].item() > 0
    
    print("\nForward pass test passed!")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Initial loss: {outputs['loss'].item():.4f}")
    
    print("\nTesting generation...")
    prompt = torch.randint(0, config["vocab_size"], (1, 32))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")
    assert generated.shape == (1, 32 + 20)
    
    print("\nGeneration test passed!")
    
    return model


if __name__ == "__main__":
    model = test_model_forward()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

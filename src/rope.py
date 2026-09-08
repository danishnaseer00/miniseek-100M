import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    print("Testing RoPE implementation from scratch...")
    
    dim = 64
    max_seq_len = 512
    batch_size = 2
    n_heads = 8
    seq_len = 128
    
    rope = RotaryPositionalEmbedding(dim, max_seq_len)
    
    cos, sin = rope(seq_len)
    print(f"cos shape: {cos.shape} (expected: [{seq_len}, {dim}])")
    print(f"sin shape: {sin.shape} (expected: [{seq_len}, {dim}])")
    
    q = torch.randn(batch_size, n_heads, seq_len, dim)
    k = torch.randn(batch_size, n_heads, seq_len, dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"\nq shape: {q.shape}")
    print(f"q_rot shape: {q_rot.shape}")
    
    assert q.shape == q_rot.shape, "Shape mismatch after RoPE"
    assert not torch.allclose(q, q_rot), "RoPE should modify the values"
    
    print("\n" + "="*50)
    print("RoPE Implementation Details:")
    print("="*50)
    print(f"Dimension: {dim}")
    print(f"Base: {rope.base}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Inverse frequencies shape: {rope.inv_freq.shape}")
    print(f"\nSample inv_freq[:5]: {rope.inv_freq[:5]}")
    print(f"Sample cos[:5, :5]:\n{cos[:5, :5]}")
    
    print("\n✓ RoPE test passed!")
    
    print("\n" + "="*50)
    print("Testing different sequence lengths:")
    print("="*50)
    
    for test_seq_len in [32, 64, 128, 256]:
        cos_test, sin_test = rope(test_seq_len)
        assert cos_test.shape == (test_seq_len, dim), f"Failed for seq_len={test_seq_len}"
        print(f"  seq_len={test_seq_len:3d}: ✓ cos shape {cos_test.shape}")
    
    print("\n✓ All sequence length tests passed!")
    
    print("\n" + "="*50)
    print("Testing extrapolation beyond max_seq_len:")
    print("="*50)
    
    extended_seq_len = max_seq_len + 100
    cos_ext, sin_ext = rope(extended_seq_len)
    print(f"Extended to {extended_seq_len}, cos shape: {cos_ext.shape}")
    assert cos_ext.shape == (extended_seq_len, dim), "Extension failed"
    print("✓ Dynamic extension works!")

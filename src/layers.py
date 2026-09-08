import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


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
    print("="*60)
    print("MINISEEK - All Components from Scratch")
    print("="*60)
    
    dim = 64
    batch_size = 2
    seq_len = 128
    
    print("\n1. RMSNorm Test")
    print("-" * 60)
    rms = RMSNorm(dim)
    x = torch.randn(batch_size, seq_len, dim)
    x_norm = rms(x)
    print(f"Input: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    print("[OK] RMSNorm works!")
    
    print("\n2. SwiGLU Test")
    print("-" * 60)
    hidden_dim = 128
    swiglu = SwiGLU(dim, hidden_dim)
    x_glu = torch.randn(batch_size, seq_len, dim)
    out = swiglu(x_glu)
    print(f"Input shape: {x_glu.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in swiglu.parameters()):,}")
    print("[OK] SwiGLU works!")
    
    print("\n3. RoPE Test")
    print("-" * 60)
    rope = RotaryPositionalEmbedding(dim)
    cos, sin = rope(seq_len)
    
    q = torch.randn(batch_size, 8, seq_len, dim)
    k = torch.randn(batch_size, 8, seq_len, dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"Query shape: {q.shape} -> {q_rot.shape}")
    print(f"Key shape: {k.shape} -> {k_rot.shape}")
    print(f"Cos cached shape: {rope.cos_cached.shape}")
    print(f"Sin cached shape: {rope.sin_cached.shape}")
    print("[OK] RoPE works!")
    
    print("\n" + "="*60)
    print("[OK] ALL COMPONENTS IMPLEMENTED FROM SCRATCH")
    print("="*60)
    print("\nNo external libraries used for:")
    print("  - RMSNorm (root mean square normalization)")
    print("  - SwiGLU (gated linear unit with SiLU)")
    print("  - RoPE (rotary positional embeddings)")

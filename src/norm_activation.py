import torch
import torch.nn as nn


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


if __name__ == "__main__":
    print("Testing RMSNorm from scratch...")
    
    dim = 256
    batch_size = 4
    seq_len = 128
    
    rms_norm = RMSNorm(dim)
    x = torch.randn(batch_size, seq_len, dim)
    
    output = rms_norm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert x.shape == output.shape, "Shape mismatch in RMSNorm"
    
    print(f"\nSample input mean: {x[0, 0, :10].mean():.4f}")
    print(f"Sample input std: {x[0, 0, :10].std():.4f}")
    print(f"Sample output mean: {output[0, 0, :10].mean():.4f}")
    print(f"Sample output std: {output[0, 0, :10].std():.4f}")
    
    print("\n✓ RMSNorm test passed!")
    
    print("\n" + "="*50)
    print("Testing SwiGLU from scratch...")
    print("="*50)
    
    hidden_dim = 512
    swiglu = SwiGLU(dim, hidden_dim)
    
    x = torch.randn(batch_size, seq_len, dim)
    output = swiglu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert x.shape == output.shape, "Shape mismatch in SwiGLU"
    
    total_params = sum(p.numel() for p in swiglu.parameters())
    print(f"\nSwiGLU parameters: {total_params:,}")
    print(f"  - w1: {dim * hidden_dim:,} params")
    print(f"  - w2: {hidden_dim * dim:,} params")
    print(f"  - w3: {dim * hidden_dim:,} params")
    
    print("\n✓ SwiGLU test passed!")

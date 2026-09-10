import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .rope import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
    
    def forward(self, x, cos, sin, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is None:
            mask = self.causal_mask[:T, :T]
        
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.o_proj(out)

import torch
import torch.nn as nn
from typing import Optional
from .norm_activation import RMSNorm, SwiGLU
from .rope import RotaryPositionalEmbedding
from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.feed_forward = SwiGLU(dim=dim, hidden_dim=mlp_hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(self, x, cos, sin, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), cos, sin, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MiniseekDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_hidden_dim: int = 1376,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        self.rope = RotaryPositionalEmbedding(dim // n_heads, max_seq_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(dim)
        
        if tie_embeddings:
            self.output = nn.Linear(dim, vocab_size, bias=False)
            self.output.weight = self.tok_embeddings.weight
        else:
            self.output = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        
        x = self.tok_embeddings(input_ids)
        
        cos, sin = self.rope(T)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ):
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            outputs = self.forward(input_ids_cond)
            logits = outputs["logits"]
            
            seen = set(input_ids_cond[0].tolist())
            if seen and repetition_penalty != 1.0:
                seen_logits = logits[0, -1, list(seen)]
                logits[0, -1, list(seen)] = torch.where(
                    seen_logits >= 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty,
                )
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = nn.functional.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative - sorted_probs > top_p
                sorted_logits[mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(
                    -1, sorted_indices, sorted_logits
                )
            
            probs = nn.functional.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> MiniseekDecoder:
    return MiniseekDecoder(
        vocab_size=config.get("vocab_size", 50257),
        dim=config.get("dim", 512),
        n_layers=config.get("n_layers", 8),
        n_heads=config.get("n_heads", 8),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 1376),
        max_seq_len=config.get("max_seq_len", 2048),
        dropout=config.get("dropout", 0.0),
        tie_embeddings=config.get("tie_embeddings", True),
    )

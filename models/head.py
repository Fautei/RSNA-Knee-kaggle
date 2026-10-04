import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class GatedCrossAttentionPooling(nn.Module):
    def __init__(
            self, 
            d_model: int = 768, 
            n_queries: int = 16, num_heads: int = 8, 
            gate_type: str = "elementwise", dropout: float = 0.1,
            inner_dim: Optional[int] = None):  
        super().__init__()
        
        self.d_model = d_model
        self.n_queries = n_queries
        self.num_heads = num_heads
        self.gate_type = gate_type
        
        # Если inner_dim не указан, используем d_model
        self.inner_dim = inner_dim if inner_dim is not None else d_model
        assert self.inner_dim % num_heads == 0, f"inner_dim ({self.inner_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.inner_dim // num_heads

        # Learnable queries
        self.query_tokens = nn.Parameter(torch.zeros(1, n_queries, self.inner_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        self.proj_k = nn.Linear(d_model, self.inner_dim)
        self.proj_v = nn.Linear(d_model, self.inner_dim)

        # Gate projection
        if gate_type == "headwise":
            self.proj_gate = nn.Linear(self.inner_dim, num_heads)
        else:
            self.proj_gate = nn.Linear(self.inner_dim, self.inner_dim)
            
        self.norm_q = nn.LayerNorm(self.inner_dim)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(self.inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(self.inner_dim, self.inner_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        L = math.prod(spatial)

        tokens = x.flatten(2).transpose(1, 2)  # (B, L, C)
        
        tokens_norm = self.norm_kv(tokens)
        k = self.proj_k(tokens_norm)
        v = self.proj_v(tokens_norm)

        q = self.query_tokens.expand(B, -1, -1)
        q_norm = self.norm_q(q)

        q_attn = q_norm.view(B, self.n_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_attn, k, v)
        out = out.transpose(1, 2).reshape(B, self.n_queries, self.inner_dim)

        # Gating
        if self.gate_type == "headwise":
            gate = torch.sigmoid(self.proj_gate(q_norm))
            gate = gate.unsqueeze(-1)
            out_reshaped = out.view(B, self.n_queries, self.num_heads, self.head_dim)
            out = (out_reshaped * gate).reshape(B, self.n_queries, self.inner_dim)
        else:
            gate = torch.sigmoid(self.proj_gate(q_norm))
            out = out * gate
        
        out = self.norm_out(out)
        out = self.dropout(out)
        out = self.proj_out(out)

        return out.flatten(1)


if __name__ == "__main__":
    gap = GatedCrossAttentionPooling(inner_dim=128)
    out = gap(torch.randn((16, 768, 8, 8)))
    print(out.shape)
# arch/attentions/multi_head_attention.py
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention from "Attention Is All You Need"
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,  # For compatibility with retention interface
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        pad_mask: [batch, seq_len] True for PAD tokens (optional)
        returns: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)  # [batch, seq_len, d_model]
        v = self.v_proj(x)  # [batch, seq_len, d_model]
        
        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, n_heads, seq_len, seq_len]
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply padding mask if provided
        if pad_mask is not None:
            # pad_mask: [batch, seq_len], True for PAD tokens
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [batch, n_heads, seq_len, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)  # [batch, seq_len, d_model]
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output
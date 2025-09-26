# arch/attentions/retention.py
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiScaleRetention(nn.Module):
    """
    Retention block (chunkwise-recurrent) that can replace self-attention.
    Based on RetNet's multi-scale retention: per-head complex rotation + exponential decay
    and a gating mechanism. Causal by construction.

    Args:
        d_model: hidden size
        n_heads: number of heads
        max_seq_len: needed for rotary/xpos cache if you add it later
        scales: number of retention scales per head (usually = n_heads or a divisor)
        decay_min, decay_max: log-spaced decay bases for multi-scale kernels
        dropout: dropout on output proj
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        scales: int | None = None,
        decay_min: float = 0.3,
        decay_max: float = 0.99,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scales = scales or n_heads

        # project tokens to per-head features
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        # per-scale mixing of retained state
        self.state_mix = nn.Parameter(torch.randn(n_heads, self.scales, self.head_dim) / math.sqrt(self.head_dim))

        # learnable decays (in (0, 1)); parameterize via sigmoid
        # initialize log-spaced between decay_min..decay_max
        init_decays = torch.logit(torch.linspace(decay_min, decay_max, self.scales))
        self.logit_decay = nn.Parameter(init_decays[None, :].repeat(n_heads, 1))  # [H, S]

        # output projection (like attention out-proj)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # gate to mix current token with retained context (GLU-ish)
        self.gate = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, is_causal: bool = True):
        """
        x: [B, T, d_model]
        pad_mask: [B, T] True for PAD (optional)
        returns: [B, T, d_model]
        """
        B, T, D = x.size()
        H, Hd = self.n_heads, self.head_dim

        h = self.in_proj(x)                        # [B, T, D]
        h = h.view(B, T, H, Hd)                    # [B, T, H, Hd]

        # retained states per head and per scale
        device = x.device
        S = torch.zeros(B, H, self.scales, Hd, device=device)  # [B,H,S,Hd]
        decays = torch.sigmoid(self.logit_decay)               # (0,1), [H,S]

        outputs = []
        for t in range(T):
            ht = h[:, t]                                      # [B, H, Hd]
            if pad_mask is not None and pad_mask[:, t].any():
                # zero-out PAD tokens to avoid polluting state
                mask_t = (~pad_mask[:, t]).float().unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
            else:
                mask_t = None

            # expand ht for scales
            ht_exp = ht.unsqueeze(2).expand(-1, -1, self.scales, -1)            # [B,H,S,Hd]

            # update retained state: S = a*S + ht
            a = decays.unsqueeze(0).unsqueeze(-1)                                # [1,H,S,1]
            S = a * S + ht_exp

            if mask_t is not None:
                S = S * mask_t                                                   # keep PAD neutral

            # readout: sum over scales with learned mixers
            read = (S * self.state_mix.unsqueeze(0)).sum(dim=2)                  # [B,H,Hd]

            outputs.append(read)

        y = torch.stack(outputs, dim=1).contiguous().view(B, T, D)               # [B,T,D]

        # gated mixing of current token features and retained context
        g = torch.sigmoid(self.gate(x))                                          # [B,T,D]
        y = g * y + (1 - g) * x

        y = self.out_proj(self.dropout(y))
        return y

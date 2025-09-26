# arch/attentions/__init__.py
from .multi_head_attention import MultiHeadAttention
from .retention import MultiScaleRetention

def make_attention(kind: str, **kwargs):
    kind = kind.lower()
    if kind in ("mha", "multihead", "multi_head"):
        return MultiHeadAttention(**kwargs)
    if kind in ("retention", "msr", "retnet"):
        return MultiScaleRetention(**kwargs)
    raise ValueError(f"Unknown attention kind: {kind}")

import torch
from arch.attentions.retention import MultiScaleRetention

def test_retention_forward_shapes():
    layer = MultiScaleRetention(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    assert y.shape == x.shape

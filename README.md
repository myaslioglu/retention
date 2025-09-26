# Transformer + Retention Layer

This project extends [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer) by adding a **Retention Layer**, inspired by the paper:

> **Attention Is All You Need Until You Need Retention**  
> arXiv:2501.09166 (2025)

The Retention Layer replaces or augments standard self-attention with a recurrent memory mechanism that captures **long-term dependencies** and allows for **persistent memory** beyond the fixed context window.

---

## 🔑 Key Additions

### Retention Layer (`MultiScaleRetention`)
- Implements **multi-scale exponential decays** per head, approximating different memory horizons.
- Maintains a **recurrent state** per head and scale:
  S_t = a · S_{t-1} + h_t
- Combines multi-scale states with learned mixers.
- Adds a **gating mechanism** to blend retained context with current token features:
  y_t = σ(W_g x_t) ⊙ Retained_t + (1 - σ(W_g x_t)) ⊙ x_t
- Causal by construction, with optional padding mask.

### Configurable Attention
Choose between standard **Multi-Head Attention (MHA)** or **Retention**:

```toml
[model]
attention_kind = "retention"   # "mha" or "retention"
```

### Encoder / Decoder Integration
- Encoder self-attention: can be MHA or Retention  
- Decoder self-attention: can be MHA or Retention  
- Decoder cross-attention: remains MHA for stability  

### Tests
Basic shape test included:

```python
def test_retention_forward_shapes():
    layer = MultiScaleRetention(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    assert y.shape == x.shape
```

---

## ⚡️ Why Retention?

Attention is powerful but quadratic in sequence length and limited to its context window. Retention offers:

- **Linear-time recurrence** (O(L·d) vs O(L²·d))  
- **O(1) memory per token at inference** (streaming-friendly)  
- **Long-term persistence** potential (cross-sequence memory)  
- **Multi-scale kernels** approximating diverse time horizons  

For background, see:
- [RetNet: Retentive Network](https://arxiv.org/abs/2307.08621) (Sun et al., 2023)  
- *Attention Is All You Need Until You Need Retention* (arXiv:2501.09166, 2025)

---

## 🚀 Usage

Train as usual:

```bash
python train.py --config config.toml
```

Switch attention mode in `config.toml`:

```toml
[model]
attention_kind = "retention"
```

---

## 📂 Structure (new/modified)

```
arch/
  attentions/
    multi_head_attention.py
    retention.py       # NEW
    __init__.py        # factory: make_attention()
  encoder/
    encoder_block.py   # retention support
  decoder/
    decoder_block.py   # retention support (self-attn)
tests/
  test_retention.py    # new test
```

---

## 🧩 Notes & Limitations

This is an **educational approximation**:
- Memory resets each forward pass (no persistent memory across sessions yet).  
- No episodic buffer or eviction/compression strategies.  
- Retention here **replaces** self-attention; the paper may describe hybrid attention+retention setups.  
- Update rule uses simple exponential decay recurrence.  

---

## 🙏 Credits

- Original Transformer implementation: [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer)  
- Retention concepts:  
  - Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, 2023  
  - *Attention Is All You Need Until You Need Retention*, arXiv:2501.09166, 2025  

---

## 📜 License

This fork inherits the license of the original repo. See `LICENSE` in [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer).

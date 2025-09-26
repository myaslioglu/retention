# Transformer + Retention Layer

This repo extends [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer) by integrating a **Retention Layer**, inspired by the paper:

> **Attention Is All You Need Until You Need Retention**  
> arXiv:2501.09166, 2025

The Retention Layer augments/replaces standard self-attention with a recurrent, memory-based mechanism that captures **long-term dependencies** and enables **persistent memory** beyond the fixed context window.

---

## 🔑 Key Additions

### 1. Retention Layer (`MultiScaleRetention`)
- Implements **multi-scale exponential decays** per head, approximating different memory horizons.
- Maintains a **recurrent state** per head and scale:  
  \[
  S_t = a \cdot S_{t-1} + h_t
  \]
- Adds a **learned mixer** to combine multi-scale states.
- Includes a **gating mechanism** to blend the retained context with the current token features:
  \[
  y_t = \sigma(W_g x_t) \odot \text{Retained}_t + (1 - \sigma(W_g x_t)) \odot x_t
  \]
- Naturally **causal** (no look-ahead).

### 2. Configurable Attention Backend
- The repo now supports choosing between:
  - Standard **Multi-Head Attention (MHA)**  
  - **Retention (MSR)** as a drop-in replacement
- Switch via `config.toml`:

```toml
[model]
attention_kind = "retention"   # options: "mha", "retention"

3. Encoder / Decoder Integration

Encoder self-attention: can be MHA or Retention.

Decoder self-attention: can be MHA or Retention.

Decoder cross-attention: left as standard MHA for stability and alignment.


4. Tests

A basic test checks shape correctness:

def test_retention_forward_shapes():
    layer = MultiScaleRetention(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    assert y.shape == x.shape


---

⚡️ Why Retention?

Traditional attention is quadratic in sequence length and forgets beyond its context window. Retention introduces:

Linear-time recurrence (O(L·d) vs O(L²·d)).

O(1) memory per token at inference, enabling streaming.

Long-term persistence, with potential for cross-session memory when extended.

Multi-scale kernels that approximate different time horizons.


For a gentle introduction, see RetNet (Sun et al. 2023) and the recent Attention Is All You Need Until You Need Retention.


---

🚀 Usage

Train with standard settings:

python train.py --config config.toml

Switch attention mode in config to "retention" to use the new layer.


---

📂 Repository Structure (key changes)

arch/
  attentions/
    multi_head_attention.py
    retention.py       # <-- NEW
    __init__.py        # factory: make_attention()
  encoder/
    encoder_block.py   # retention wired in
  decoder/
    decoder_block.py   # retention wired in (self-attention only)
tests/
  test_retention.py    # shape test


---

🧩 Limitations vs. the Paper

This implementation is an educational approximation:

No persistent memory across batches/sessions (state resets per forward pass).

No external memory store, episodic buffer, or eviction/compression.

Retention here replaces self-attention; the paper may describe hybrid setups where attention and retention co-exist.

Update rule is a simplified exponential decay recurrence.


Still, it demonstrates how to slot a Retention Layer into a Transformer with minimal code changes.


---

🙏 Credits

Original Transformer implementation: MayukhSobo/Transformer

Retention concepts:

Sun et al., Retentive Network: A Successor to Transformer for Large Language Models, 2023

Attention Is All You Need Until You Need Retention, arXiv:2501.09166, 2025




---

📜 License

This fork inherits the license of the original repo. Please consult LICENSE in MayukhSobo/Transformer.


---

---

Do you also want me to make a **separate section in the README with example plots** (e.g. how attention vs. retention scales with sequence length), or keep it lean and code-focused?


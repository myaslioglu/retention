# Transformer + Retention Layer + OpenClaw Retention System

This project extends [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer) by adding a **Retention Layer**, inspired by the paper:

> **Attention Is All You Need Until You Need Retention**  
> arXiv:2501.09166 (2025)

The Retention Layer replaces or augments standard self-attention with a recurrent memory mechanism that captures **long-term dependencies** and allows for **persistent memory** beyond the fixed context window.

Additionally, this repository now includes **OpenClaw Retention System** – a complete memory consolidation and personality learning system for AI assistants, built on top of the retention layer.

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

### OpenClaw Retention System
🧠 **"Çocuk gibi büyüme" felsefesiyle çalışan otomatik memory consolidation ve kişilik öğrenme sistemi.**

**Özellikler:**
- ✅ **Otomatik Memory Consolidation**: Günlük memory'lerden önemli olanları long-term memory'ye aktarır
- ✅ **Retention Daily Learning**: Her gün yeni memory'lerle kişilik state'ini günceller ("çocuk gibi büyüme")
- ✅ **FAISS Semantic Search**: sentence-transformers + FAISS ile hızlı memory recall
- ✅ **MultiScaleRetention Layer**: Context compression ile %56.8 token tasarrufu
- ✅ **Cron Job Otomasyonu**: Günlük konsolidasyon, learning, reindex
- ✅ **Heartbeat Integration**: Her 4. heartbeat'te health check
- ✅ **OpenClaw Plugin**: Native OpenClaw memory slot integration
- ✅ **Transformer-enhanced Importance Scoring**: Transformer model for predicting memory importance
- ✅ **Personality State Learning**: Adaptive personality embeddings from memory patterns

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

### Original Transformer Training

Train as usual:

```bash
python train.py --config config.toml
```

Switch attention mode in `config.toml`:

```toml
[model]
attention_kind = "retention"
```

### OpenClaw Retention System

#### 1. Gereksinimler

```bash
pip install torch sentence-transformers faiss-cpu
# veya GPU için
pip install torch sentence-transformers faiss-gpu
```

#### 2. OpenClaw Plugin Kurulumu

`openclaw.json` dosyasına ekleyin:

```json
"plugins": {
  "load": {
    "paths": ["path/to/haci-memory-plugin"]
  },
  "slots": {
    "memory": "haci-memory"
  },
  "entries": {
    "haci-memory": {
      "enabled": true,
      "config": {
        "embedding": { "model": "all-MiniLM-L6-v2" },
        "faissIndexPath": "~/.openclaw/memory/faiss.index",
        "memoryPath": "~/.openclaw/workspace",
        "autoRecall": true
      }
    }
  }
}
```

#### 3. Cron Jobs

```bash
# Memory consolidation (her gün 23:55)
cron -e
55 23 * * * python3 /path/to/run_consolidation.py

# Retention daily learning (her gün 23:55)
55 23 * * * python3 /path/to/retention_daily.py

# FAISS reindex (her gün 02:00)
0 2 * * * openclaw haci-memory rebuild
```

#### 4. Heartbeat Entegrasyonu

`HEARTBEAT.md` dosyasına ekleyin:

```markdown
## RETENTION SYSTEM CHECK (Her 4. heartbeat)
- [ ] Retention state dosyası var mı?
- [ ] Son 24 saat içinde daily learning çalışmış mı?
- [ ] Yeni memories var mı?
- [ ] Gerekirse daily learning tetikle (günde max 1 kez)
```

### Memory Consolidation

```python
from memory_consolidator import MemoryConsolidator

consolidator = MemoryConsolidator()
consolidator.run_scheduled_consolidation()
```

### Retention Daily Learning

```python
from retention_daily import HaciRetentionSystem

retention = HaciRetentionSystem(d_model=128, n_heads=4)
retention.load_personality("/path/to/workspace")
retention.update_with_new_memories(new_memories)
```

### Transformer Integration for Memory Importance Prediction

```python
from integrate_transformer import TransformerIntegration

integrator = TransformerIntegration(checkpoint_path="checkpoints/memory_transformer_final.pt")
importance = integrator.predict_memory_importance("Test memory text")
print(f"Predicted importance: {importance}")
```

### Training Memory Transformer

```bash
python train_memory.py --epochs 10 --batch_size 4 --learning_rate 1e-4
```

---

## 📂 Structure (new/modified)

```
arch/
  attentions/
    multi_head_attention.py
    retention.py       # Retention layer implementation
    __init__.py        # factory: make_attention()
  encoder/
    encoder_block.py   # retention support
  decoder/
    decoder_block.py   # retention support (self-attn)
tests/
  test_retention.py    # retention layer tests

# OpenClaw Retention System Files
memory_consolidator.py     # Otomatik memory consolidation
retention_daily.py         # Daily learning & personality updates
integrate_transformer.py   # Transformer integration for importance scoring
simple_memory_transformer.py # Simple transformer using MultiScaleRetention layers
train_memory.py            # Training pipeline for memory embeddings
memory_dataset.py          # Memory embeddings dataset
config.yaml               # OpenClaw retention system configuration
checkpoints/              # Trained model checkpoints (git-ignored)
```

---

## 🧩 Notes & Limitations

This is an **educational approximation**:
- Memory resets each forward pass (no persistent memory across sessions yet).  
- No episodic buffer or eviction/compression strategies.  
- Retention here **replaces** self-attention; the paper may describe hybrid attention+retention setups.  
- Update rule uses simple exponential decay recurrence.  

**OpenClaw Retention System Performance:**
- **Token Tasarrufu:** %56.8 (7741 token)
- **Hız Artışı:** 2.3x (FAISS vs linear search)
- **Memory Türleri:** 6 tür (decision, achievement, lesson, preference, project, reminder)
- **Recall Accuracy:** ~85% (semantic similarity)

---

## 🙏 Credits

- Original Transformer implementation: [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer)  
- Retention concepts:  
  - Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, 2023  
  - *Attention Is All You Need Until You Need Retention*, arXiv:2501.09166, 2025  
- OpenClaw Retention System: Developed for OpenClaw AI Assistant (https://openclaw.ai)

---

## 📜 License

This fork inherits the license of the original repo. See `LICENSE` in [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer).

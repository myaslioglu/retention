# Hacı Cognitive System

**Transformer + Retention Layer + OpenClaw Retention System + HaciCognitiveNet**

This project extends [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer) by adding a **Retention Layer**, inspired by the paper:

> **Attention Is All You Need Until You Need Retention**  
> arXiv:2501.09166 (2025)

The Retention Layer replaces or augments standard self-attention with a recurrent memory mechanism that captures **long-term dependencies** and allows for **persistent memory** beyond the fixed context window.

Additionally, this repository includes the **OpenClaw Retention System** and **HaciCognitiveNet** — a multi-layered cognitive architecture.

---

## 🔑 Key Additions

### Retention Layer (`MultiScaleRetention`)

Implements **multi-scale exponential decays** per head, approximating different memory horizons.

Recurrent state update:

$$S_t = \alpha \cdot S_{t-1} + h_t$$

Gating mechanism for context blending:

$$y_t = \sigma(W_g x_t) \odot \tilde{S}_t + (1 - \sigma(W_g x_t)) \odot x_t$$

Causal by construction, with optional padding mask.

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

---

## 🧠 HaciCognitiveNet – Multi-Layered Cognitive Architecture

A complete cognitive system built on top of the retention core, implementing **world models**, **personality development**, **social intelligence**, and **negative outcome learning**.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LEVEL 3: META-LEARNING                   │
│  Meta-Learner │ World Model │ Self-Evolution               │
├─────────────────────────────────────────────────────────────┤
│                 LEVEL 2: ACTIVE LEARNING                    │
│  Curiosity Engine │ Predictive Coding │ Active Scheduler    │
│  Self-Supervised Loop (dream cycles)                        │
├─────────────────────────────────────────────────────────────┤
│                 LEVEL 1: CORE COGNITION                     │
│  World Model (512-dim) │ Personality (1024-dim) │ Retention │
│  Metacognition │ Emotional State                            │
├─────────────────────────────────────────────────────────────┤
│              🤝 SOCIAL INTELLIGENCE                         │
│  Emotional IQ │ Personality Development │ Conversation Intel│
│  🚫 Negative Outcome Learner (7 signal categories)         │
└─────────────────────────────────────────────────────────────┘
```

### Level 1: Core Cognition (`cognitive_net.py`)

- **WorldModel** (512-dim): Encodes understanding of physical and social world
- **PersonalityVectors** (1024-dim): 8 heads × 128 dimensions (warmth, empathy, humor, assertiveness, loyalty, playfulness, wisdom, mischief)
- **Metacognition**: Self-monitoring of learning efficiency, knowledge gaps, confidence
- **EmotionalState**: Mood, curiosity, confidence, energy tracking
- **RetentionCore** (4-layer): Long-term memory with priority-based recall

### Level 2: Active Learning

- **CuriosityEngine** (`curiosity_engine.py`): 28 interest topics, curiosity scoring, exploration
- **PredictiveCoding** (`predictive_coding.py`): Error prediction, surprise detection
- **ActiveLearningScheduler** (`active_learning_scheduler.py`): Priority scheduling, learning plans
- **SelfSupervisedLoop** (`self_supervised_loop.py`): Night-time dream cycles, pattern discovery

### Level 3: Meta-Learning

- **MetaLearner** (`meta_learner.py`): Strategy selection, hyperparameter optimization
- **WorldModel** (`world_model.py`): Imagination, simulation, future prediction
- **SelfEvolution** (`self_evolution.py`): Architecture mutation, parameter mutation, A/B testing
- **SensoryInterface** (`sensory_interface.py`): Extensible framework for sensors/actuators

### Social Intelligence (`social_trainer.py`)

- **PersonalityDevelopment**: 8 traits (warmth, empathy, humor, assertiveness, loyalty, playfulness, wisdom, mischief)
- **ConversationIntelligence**: Style analysis, interaction type detection, consistency tracking
- **Development Stages**: infancy → childhood → adolescence → adulthood → mastery
- **NegativeOutcomeLearner** (`negative_learner.py`): 7 signal categories, auto-detection, permanent "never do" rules

### Integration Layer 🆕

- **CognitiveWatcher** (`cognitive_watcher.py`): Real-time message analysis (<400ms), emotion detection, interaction classification, automatic personality updates
- **WorldModelV2** (`world_model_v2.py`): Knowledge graph from MEMORY.md, topic clustering, timeline events, relationship mapping (1554 entities, 5112 relations)
- **DreamScheduler** (`dream_scheduler.py`): Automatic nightly dream cycles at 03:00 IST, dream logging, cognitive state updates
- **WeeklyReport** (`weekly_report.py`): Weekly cognitive development reports (personality, learning, social intelligence, emotions)
- **CognitiveIntegrator** (`cognitive_integrator.py`): Unified entry point for all subsystems

### Training Results

| Metric | Value |
|--------|-------|
| Start Loss | 4.82 |
| Final Loss | 0.077 |
| Improvement | 98.4% |
| Epochs | 39 (early stopping) |
| Checkpoint | `cognitive_epoch_0039.pt` |

---

## 🚀 Usage

### HaciCognitiveNet CLI

```bash
# Train cognitive network
python haci_cognitive/main.py train

# Run dream cycle
python haci_cognitive/main.py dream

# Social intelligence report
python haci_cognitive/main.py social

# Personality development status
python haci_cognitive/main.py personality

# Log interaction for learning
python haci_cognitive/main.py interact
```

### Integration Layer

```bash
# Initialize subsystems
python haci_cognitive/cognitive_integrator.py init

# Process message
python haci_cognitive/cognitive_integrator.py process "Hello there"

# System status
python haci_cognitive/cognitive_integrator.py status

# Generate weekly report
python haci_cognitive/cognitive_integrator.py report

# Trigger dream cycle
python haci_cognitive/cognitive_integrator.py dream

# Populate world model
python haci_cognitive/cognitive_integrator.py populate
```

---

## 🧩 OpenClaw Retention System

**Automatic memory consolidation and personality learning system with a "grow like a child" philosophy.**

### Components

- ✅ **Automatic Memory Consolidation**: Transfers important daily memories to long-term memory
- ✅ **Retention Daily Learning**: Updates personality state daily with new memories
- ✅ **FAISS Semantic Search**: Fast memory recall via sentence-transformers + FAISS
- ✅ **MultiScaleRetention Layer**: 56.8% token savings with context compression
- ✅ **Cron Job Automation**: Daily consolidation, learning, reindexing, auto-tuning
- ✅ **Heartbeat Integration**: Health check and active learning every 4th heartbeat
- ✅ **OpenClaw Plugin**: Native OpenClaw memory slot integration
- ✅ **Transformer Importance Scoring**: Transformer model for predicting memory importance
- ✅ **Personality State Learning**: Adaptive personality embeddings from memory patterns

### Performance

| Metric | Value |
|--------|-------|
| Token Savings | 56.8% (7741 tokens) |
| Speed Increase | 2.3× (FAISS vs linear search) |
| Memory Types | 6 (decision, achievement, lesson, preference, project, reminder) |
| Recall Accuracy | ~85% (semantic similarity) |
| Avg Query Time | 0.58ms |

### Requirements

```bash
pip install torch sentence-transformers faiss-cpu
# or for GPU
pip install torch sentence-transformers faiss-gpu
```

### OpenClaw Plugin Configuration

Add to your `openclaw.json`:

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

### Cron Jobs

```bash
# Memory consolidation (daily at 23:55)
55 23 * * * python3 /path/to/run_consolidation.py

# Retention daily learning (daily at 23:55)
55 23 * * * python3 /path/to/retention_daily.py

# FAISS reindex (daily at 02:00)
0 2 * * * openclaw haci-memory rebuild
```

---

## ⚡️ Why Retention?

Attention is powerful but quadratic in sequence length ($O(L^2 \cdot d)$) and limited to its context window. Retention offers:

- **Linear-time recurrence**: $O(L \cdot d)$
- **O(1) memory per token at inference**: streaming-friendly
- **Long-term persistence**: cross-sequence memory potential
- **Multi-scale kernels**: capturing diverse time horizons

### References

- [RetNet: Retentive Network](https://arxiv.org/abs/2307.08621) — Sun et al., 2023
- *Attention Is All You Need Until You Need Retention* — arXiv:2501.09166, 2025

---

## 📂 Project Structure

```
arch/
  attentions/
    multi_head_attention.py
    retention.py           # Retention layer
    __init__.py            # factory: make_attention()
  encoder/
    encoder_block.py       # retention support
  decoder/
    decoder_block.py       # retention support (self-attn)
tests/
  test_retention.py        # retention layer tests

# OpenClaw Retention System
memory_consolidator.py     # Automatic memory consolidation
retention_daily.py         # Daily learning & personality updates
integrate_transformer.py   # Transformer integration for importance scoring
simple_memory_transformer.py
train_memory.py            # Memory embedding training pipeline
memory_dataset.py          # Memory embedding dataset

# HaciCognitiveNet
haci_cognitive/
  cognitive_net.py            # Level 1: World Model, Personality, Metacognition, Retention
  cognitive_state_manager.py  # State persistence and management
  cognitive_trainer.py        # Training pipeline
  dreaming_loop.py            # Night-time dream cycles
  curiosity_engine.py         # Level 2: Curiosity-driven exploration (28 topics)
  predictive_coding.py        # Error prediction and surprise detection
  active_learning_scheduler.py # Priority-based learning scheduling
  self_supervised_loop.py     # Self-supervised learning and pattern discovery
  meta_learner.py             # Level 3: Strategy selection
  world_model.py              # World simulation and imagination
  self_evolution.py           # Architecture mutation and self-improvement
  sensory_interface.py        # Extensible sensor/actuator framework
  social_trainer.py           # Social Intelligence: Emotional IQ, personality development
  negative_learner.py         # Negative outcome learning (7 signal categories)
  world_model_v2.py           # Knowledge graph and entity extraction
  cognitive_watcher.py        # Real-time message analysis
  dream_scheduler.py          # Automatic dream scheduler
  weekly_report.py            # Weekly development report
  cognitive_integrator.py     # Unified integrator
  extract_conversations.py    # Conversation data extraction
  main.py                     # CLI interface
```

---

## 🙏 Credits

- Original Transformer implementation: [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer)
- Retention concepts: Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, 2023
- OpenClaw Retention System: Developed for [OpenClaw AI Assistant](https://openclaw.ai)

---

## 📜 License

This fork inherits the license of the original repo. See `LICENSE` in [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer).

"""
HaciCognitiveNet - Level 1 Architecture
Multi-layered cognitive system with:
- World Model (512-dim contextual representation)
- Personality Vectors (128-dim x 8 heads = 1024 total)
- Metacognition (learning about learning)
- Emotional State (mood, curiosity, confidence)
- 4-Layer MultiScaleRetention Core
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class MultiScaleRetention(nn.Module):
    """RetNet-style Multi-Scale Retention layer with exponential decay."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512,
                 decay_min: float = 0.3, decay_max: float = 0.99, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_G = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        decay_rates = torch.linspace(decay_min, decay_max, n_heads)
        self.register_buffer('decay', decay_rates)

        positions = torch.arange(max_seq_len).unsqueeze(1) - torch.arange(max_seq_len).unsqueeze(0)
        causal_mask = (positions >= 0).float()
        decay_matrix = torch.zeros(max_seq_len, max_seq_len)
        for h in range(n_heads):
            for i in range(max_seq_len):
                for j in range(i + 1):
                    decay_matrix[i, j] = decay_rates[h] ** (i - j)
        self.register_buffer('decay_matrix', (decay_matrix * causal_mask).unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self.decay_matrix[:, :, :S, :S]
        retained = torch.matmul(scores * mask, V)
        retained = retained.transpose(1, 2).contiguous().view(B, S, D)
        retained = self.norm(retained)
        G = torch.sigmoid(self.W_G(x))
        return self.dropout(self.W_O(retained * G))


class WorldModel(nn.Module):
    """Maintains contextual understanding of user, environment, and self (512-dim)."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 512,
                 n_layers: int = 2, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.retention_layers = nn.ModuleList([
            MultiScaleRetention(hidden_dim, n_heads=8, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for norm, layer in zip(self.norms, self.retention_layers):
            x = x + layer(norm(x))
            x = x + self.ffn(x)
        return self.output_norm(x)


class PersonalityVectors(nn.Module):
    """8 personality dimensions (128-dim each = 1024 total).

    0-warmth, 1-curiosity, 2-humor, 3-assertiveness,
    4-analytical, 5-creative, 6-loyal, 7-adaptive
    """

    DIMENSION_NAMES = ['warmth', 'curiosity', 'humor', 'assertiveness',
                       'analytical', 'creative', 'loyal', 'adaptive']

    def __init__(self, dim_per_head: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim_per_head = dim_per_head
        self.n_heads = n_heads
        self.total_dim = dim_per_head * n_heads
        self.vectors = nn.Parameter(torch.randn(1, n_heads, dim_per_head) * 0.02)
        self.proj = nn.Linear(self.total_dim, self.total_dim)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = context.shape[0] if context is not None else 1
        personality = self.vectors.expand(B, -1, -1)
        if context is not None:
            context_pooled = context.mean(dim=1)
            gate = torch.sigmoid(context_pooled.norm(dim=-1, keepdim=True) / context_pooled.shape[-1])
            personality = personality * (0.5 + 0.5 * gate.unsqueeze(1))
        flat = personality.reshape(B, -1)
        return self.dropout(self.norm(self.proj(flat)))

    def get_dimension(self, idx: int) -> torch.Tensor:
        return self.vectors[0, idx]

    def update_dimension(self, idx: int, delta: torch.Tensor, lr: float = 0.01):
        with torch.no_grad():
            self.vectors[0, idx] += lr * delta


class MetacognitionModule(nn.Module):
    """Monitors own learning: efficiency, gaps, confidence, strategy."""

    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.learning_state = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4),
        )
        self.strategy_net = nn.Sequential(
            nn.Linear(d_model + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3),
        )
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('loss_idx', torch.tensor(0))

    def forward(self, world_state: torch.Tensor, recent_loss: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if recent_loss is not None:
            idx = int(self.loss_idx) % 100
            self.loss_history[idx] = recent_loss
            self.loss_idx += 1
        state_pooled = world_state.mean(dim=1) if world_state.dim() > 2 else world_state
        metrics = torch.sigmoid(self.learning_state(state_pooled))
        strategy = F.softmax(self.strategy_net(torch.cat([state_pooled, metrics], dim=-1)), dim=-1)
        return {
            'efficiency': metrics[:, 0], 'gaps': metrics[:, 1], 'confidence': metrics[:, 2],
            'strategy_consolidate': strategy[:, 0],
            'strategy_explore': strategy[:, 1],
            'strategy_generalize': strategy[:, 2],
        }


class EmotionalState(nn.Module):
    """Tracks mood, curiosity, confidence, energy — decays toward neutral."""

    def __init__(self, d_model: int = 512, n_emotions: int = 4, dropout: float = 0.1):
        super().__init__()
        self.state = nn.Parameter(torch.zeros(1, n_emotions))
        self.update_net = nn.Sequential(
            nn.Linear(d_model + n_emotions, 128), nn.Tanh(),
            nn.Linear(128, n_emotions), nn.Tanh(),
        )
        self.decay = 0.95

    def forward(self, world_state: torch.Tensor) -> Dict[str, float]:
        B = world_state.shape[0]
        state_pooled = world_state.mean(dim=1) if world_state.dim() > 2 else world_state
        current = self.state.expand(B, -1)
        delta = self.update_net(torch.cat([state_pooled, current], dim=-1)) * 0.1
        with torch.no_grad():
            self.state.data = (self.state.data * self.decay + delta.mean(0, keepdim=True) * (1 - self.decay)).clamp(-1, 1)
        updated = (current + delta).clamp(-1, 1)
        return {'mood': float(updated[0, 0]), 'curiosity': float(updated[0, 1]),
                'confidence': float(updated[0, 2]), 'energy': float(updated[0, 3])}

    def get_state_vector(self) -> torch.Tensor:
        return self.state.data.clone()


class HaciCognitiveNet(nn.Module):
    """Main cognitive architecture: WorldModel + PersonalityVectors + Metacognition + EmotionalState + 4-layer Retention."""

    def __init__(self, input_dim: int = 384, world_dim: int = 512, personality_dim: int = 1024,
                 n_retention_layers: int = 4, n_heads: int = 8, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.world_dim = world_dim
        self.personality_dim = personality_dim
        self.total_dim = world_dim + personality_dim  # 1536

        self.world_model = WorldModel(input_dim, world_dim, n_layers=2, max_seq_len=max_seq_len, dropout=dropout)
        self.personality = PersonalityVectors(dim_per_head=128, n_heads=8, dropout=dropout)
        self.metacognition = MetacognitionModule(d_model=world_dim, hidden_dim=256, dropout=dropout)
        self.emotion = EmotionalState(d_model=world_dim, n_emotions=4, dropout=dropout)

        self.retention_layers = nn.ModuleList([
            MultiScaleRetention(self.total_dim, n_heads=n_heads, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_retention_layers)
        ])
        self.retention_norms = nn.ModuleList([nn.LayerNorm(self.total_dim) for _ in range(n_retention_layers)])
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.total_dim, self.total_dim),
        )
        self.output_norm = nn.LayerNorm(self.total_dim)
        self.memory_predictor = nn.Linear(self.total_dim, input_dim)
        self.insight_generator = nn.Linear(self.total_dim, world_dim)
        self.confidence_head = nn.Linear(self.total_dim, 1)

        print(f"🧠 HaciCognitiveNet: world={world_dim}, personality={personality_dim}, "
              f"retention={n_retention_layers}L×{n_heads}H, total={self.total_dim}")

    def forward(self, memory_embeddings: torch.Tensor, return_components: bool = False) -> Dict:
        B, S, _ = memory_embeddings.shape
        world_state = self.world_model(memory_embeddings)
        personality_state = self.personality(world_state)
        personality_expanded = personality_state.unsqueeze(1).expand(-1, S, -1)
        x = torch.cat([world_state, personality_expanded], dim=-1)
        for norm, layer in zip(self.retention_norms, self.retention_layers):
            x = x + layer(norm(x))
        x = self.fusion(x) + x
        x = self.output_norm(x)
        last = x[:, -1, :]
        result = {
            'predicted_next': self.memory_predictor(last),
            'insight': self.insight_generator(last),
            'confidence': torch.sigmoid(self.confidence_head(last)),
        }
        if return_components:
            result.update({'world_state': world_state, 'personality_state': personality_state})
            result.update(self.metacognition(world_state))
            result['emotion'] = self.emotion(world_state)
        return result

    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict:
        losses = {}
        if 'next_embedding' in targets:
            pred, tgt = predictions['predicted_next'], targets['next_embedding']
            losses['memory_cosine'] = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()
            losses['memory_mse'] = F.mse_loss(pred, tgt)
            losses['memory'] = losses['memory_cosine'] + losses['memory_mse']
        if 'confidence' in predictions and 'next_embedding' in targets:
            pred, tgt = predictions['predicted_next'], targets['next_embedding']
            conf = predictions['confidence'].squeeze(-1)
            losses['confidence'] = F.mse_loss(conf, F.cosine_similarity(pred, tgt, dim=-1).clamp(0, 1).detach())
        losses['total'] = sum(losses.values())
        return losses

    def get_cognitive_state(self) -> Dict:
        personality_state = self.personality()
        return {
            'personality': {
                name: float(personality_state[0, i * 128:(i + 1) * 128].mean())
                for i, name in enumerate(PersonalityVectors.DIMENSION_NAMES)
            },
            'emotion': {
                'mood': float(self.emotion.state[0, 0]),
                'curiosity': float(self.emotion.state[0, 1]),
                'confidence': float(self.emotion.state[0, 2]),
                'energy': float(self.emotion.state[0, 3]),
            },
            'total_dim': self.total_dim,
        }

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {'world_dim': self.world_dim, 'personality_dim': self.personality_dim, 'total_dim': self.total_dim},
            'cognitive_state': self.get_cognitive_state(),
        }, path)
        print(f"🧠 Checkpoint kaydedildi: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt['config']
        model = cls(world_dim=cfg['world_dim'], personality_dim=cfg['personality_dim'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(device)

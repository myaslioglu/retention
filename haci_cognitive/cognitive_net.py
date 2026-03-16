"""
HaciCognitiveNet - Level 1 Architecture
Multi-layered cognitive system with:
- World Model (512-dim contextual representation)
- Personality Vectors (128-dim x 8 heads = 1024 total)
- Metacognition (learning about learning)
- Emotional State (mood, curiosity, confidence)
- 4-Layer MultiScaleRetention Core
- Dreaming Loop Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class MultiScaleRetention(nn.Module):
    """
    RetNet-style Multi-Scale Retention (MSR) layer.
    Processes sequences with exponential decay for memory.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512,
                 decay_min: float = 0.3, decay_max: float = 0.99, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        # Projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_G = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Decay factors (different per head)
        decay_rates = torch.linspace(decay_min, decay_max, n_heads)
        self.register_buffer('decay', decay_rates)
        
        # Causal mask + decay matrix
        positions = torch.arange(max_seq_len).unsqueeze(1) - torch.arange(max_seq_len).unsqueeze(0)
        causal_mask = (positions >= 0).float()
        decay_matrix = torch.zeros(max_seq_len, max_seq_len)
        for h in range(n_heads):
            for i in range(max_seq_len):
                for j in range(i + 1):
                    decay_matrix[i, j] = decay_rates[h] ** (i - j)
        decay_matrix = decay_matrix * causal_mask
        self.register_buffer('decay_matrix', decay_matrix.unsqueeze(0).unsqueeze(0))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        
        # Retention computation with exponential decay
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self.decay_matrix[:, :, :S, :S]
        retained = torch.matmul(scores * mask, V)
        
        retained = retained.transpose(1, 2).contiguous().view(B, S, D)
        retained = self.norm(retained)
        
        # Gating
        G = torch.sigmoid(self.W_G(x))
        output = self.W_O(retained * G)
        
        return self.dropout(output)


class WorldModel(nn.Module):
    """
    World Model: Maintains contextual understanding of user, environment, and self.
    Input: Current state embedding (various sources)
    Output: Updated world representation (512-dim)
    """
    
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
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for norm, layer in zip(self.norms, self.retention_layers):
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + x
            
            residual = x
            x = self.ffn(x)
            x = residual + x
        
        return self.output_norm(x)  # [B, S, 512]


class PersonalityVectors(nn.Module):
    """
    Personality Vectors: 8 different personality dimensions, each 128-dim.
    Total: 1024 dimensions of personality representation.
    
    Dimensions:
    0. Warmth (sıcaklık) - friendliness, empathy
    1. Curiosity (merak) - desire to learn, explore
    2. Humor (mizah) - playful, witty
    3. Assertiveness (kararlılık) - direct, confident
    4. Analytical (analitik) - logical, systematic
    5. Creative (yaratıcı) - novel, imaginative
    6. Loyal (sadık) - protective, dependable
    7. Adaptive (uyumlu) - flexible, responsive
    """
    
    DIMENSION_NAMES = [
        'warmth', 'curiosity', 'humor', 'assertiveness',
        'analytical', 'creative', 'loyal', 'adaptive'
    ]
    
    def __init__(self, dim_per_head: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim_per_head = dim_per_head
        self.n_heads = n_heads
        self.total_dim = dim_per_head * n_heads  # 1024
        
        # Learnable personality embeddings
        self.vectors = nn.Parameter(torch.randn(1, n_heads, dim_per_head) * 0.02)
        
        # Projection to combine personality dimensions
        self.proj = nn.Linear(self.total_dim, self.total_dim)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get personality state, optionally modulated by context.
        Returns: [B, 1024]
        """
        B = context.shape[0] if context is not None else 1
        
        # Expand for batch
        personality = self.vectors.expand(B, -1, -1)  # [B, 8, 128]
        
        if context is not None:
            # Context modulation: project context to personality dim
            context_pooled = context.mean(dim=1)  # [B, 512]
            # Simple gating based on context magnitude
            context_mag = context_pooled.norm(dim=-1, keepdim=True)  # [B, 1]
            gate = torch.sigmoid(context_mag / context_pooled.shape[-1])
            personality = personality * (0.5 + 0.5 * gate.unsqueeze(1))  # [B, 8, 128]
        
        flat = personality.reshape(B, -1)  # [B, 1024]
        flat = self.proj(flat)
        flat = self.norm(flat)
        
        return self.dropout(flat)
    
    def get_dimension(self, idx: int) -> torch.Tensor:
        """Get specific personality dimension."""
        return self.vectors[0, idx]
    
    def update_dimension(self, idx: int, delta: torch.Tensor, lr: float = 0.01):
        """Update a personality dimension (during learning)."""
        with torch.no_grad():
            self.vectors[0, idx] += lr * delta


class MetacognitionModule(nn.Module):
    """
    Metacognition: "How am I learning?" - monitors and optimizes own learning.
    
    Tracks:
    - Learning efficiency (loss trajectory)
    - Knowledge gaps (what I don't know)
    - Confidence calibration (am I confident correctly?)
    - Strategy effectiveness (which approach works best?)
    """
    
    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Learning state tracker
        self.learning_state = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [efficiency, gaps, confidence, strategy_score]
        )
        
        # Strategy selector (which learning strategy to use)
        self.strategy_net = nn.Sequential(
            nn.Linear(d_model + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [consolidate, explore, generalize]
        )
        
        # Running statistics
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('loss_idx', torch.tensor(0))
    
    def forward(self, world_state: torch.Tensor, recent_loss: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Analyze current learning state and recommend strategy.
        
        Returns dict with:
        - efficiency: learning efficiency score [0-1]
        - gaps: knowledge gap estimate [0-1]  
        - confidence: confidence calibration [0-1]
        - strategy: recommended strategy [consolidate, explore, generalize]
        """
        # Update loss history
        if recent_loss is not None:
            idx = int(self.loss_idx) % 100
            self.loss_history[idx] = recent_loss
            self.loss_idx += 1
        
        # Analyze world state
        state_pooled = world_state.mean(dim=1) if world_state.dim() > 2 else world_state
        
        learning_metrics = torch.sigmoid(self.learning_state(state_pooled))
        
        efficiency = learning_metrics[:, 0]
        gaps = learning_metrics[:, 1]
        confidence = learning_metrics[:, 2]
        strategy_score = learning_metrics[:, 3]
        
        # Strategy selection
        strategy_input = torch.cat([state_pooled, learning_metrics], dim=-1)
        strategy_logits = self.strategy_net(strategy_input)
        strategy = F.softmax(strategy_logits, dim=-1)
        
        return {
            'efficiency': efficiency,
            'gaps': gaps,
            'confidence': confidence,
            'strategy_consolidate': strategy[:, 0],
            'strategy_explore': strategy[:, 1],
            'strategy_generalize': strategy[:, 2],
        }


class EmotionalState(nn.Module):
    """
    Emotional State: Current mood, curiosity level, and confidence.
    Evolves over time based on interactions and learning.
    """
    
    def __init__(self, d_model: int = 512, n_emotions: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_emotions = n_emotions  # [mood, curiosity, confidence, energy]
        
        # Initial emotional state
        self.state = nn.Parameter(torch.zeros(1, n_emotions))
        
        # Emotion update network
        self.update_net = nn.Sequential(
            nn.Linear(d_model + n_emotions, 128),
            nn.Tanh(),
            nn.Linear(128, n_emotions),
            nn.Tanh()  # [-1, 1] range for deltas
        )
        
        # Decay toward neutral
        self.decay = 0.95
    
    def forward(self, world_state: torch.Tensor) -> Dict[str, float]:
        """
        Update and return emotional state.
        """
        B = world_state.shape[0]
        state_pooled = world_state.mean(dim=1) if world_state.dim() > 2 else world_state
        
        # Get current state
        current = self.state.expand(B, -1)
        
        # Compute update
        update_input = torch.cat([state_pooled, current], dim=-1)
        delta = self.update_net(update_input) * 0.1  # Small updates
        
        with torch.no_grad():
            self.state.data = (self.state.data * self.decay + delta.mean(dim=0, keepdim=True) * (1 - self.decay)).clamp(-1, 1)
        
        updated = (current + delta).clamp(-1, 1)
        
        return {
            'mood': float(updated[0, 0]),
            'curiosity': float(updated[0, 1]),
            'confidence': float(updated[0, 2]),
            'energy': float(updated[0, 3]),
        }
    
    def get_state_vector(self) -> torch.Tensor:
        """Get current emotional state as tensor."""
        return self.state.data.clone()


class HaciCognitiveNet(nn.Module):
    """
    Main HaciCognitiveNet - Level 1 Architecture
    
    Integrates:
    - World Model (512-dim)
    - Personality Vectors (1024-dim)
    - Metacognition
    - Emotional State
    - 4-Layer Retention Core
    """
    
    def __init__(self, 
                 input_dim: int = 384,  # sentence-transformer embedding dim
                 world_dim: int = 512,
                 personality_dim: int = 1024,
                 n_retention_layers: int = 4,
                 n_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.world_dim = world_dim
        self.personality_dim = personality_dim
        self.total_dim = world_dim + personality_dim  # 1536
        
        # === Component 1: World Model ===
        self.world_model = WorldModel(
            input_dim=input_dim,
            hidden_dim=world_dim,
            n_layers=2,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # === Component 2: Personality Vectors ===
        self.personality = PersonalityVectors(
            dim_per_head=128,
            n_heads=8,
            dropout=dropout
        )
        
        # === Component 3: Metacognition ===
        self.metacognition = MetacognitionModule(
            d_model=world_dim,
            hidden_dim=256,
            dropout=dropout
        )
        
        # === Component 4: Emotional State ===
        self.emotion = EmotionalState(
            d_model=world_dim,
            n_emotions=4,
            dropout=dropout
        )
        
        # === Component 5: Retention Core (4 layers) ===
        self.retention_layers = nn.ModuleList([
            MultiScaleRetention(
                d_model=self.total_dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            for _ in range(n_retention_layers)
        ])
        self.retention_norms = nn.ModuleList([
            nn.LayerNorm(self.total_dim) for _ in range(n_retention_layers)
        ])
        
        # === Fusion Layer ===
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.total_dim, self.total_dim),
        )
        self.output_norm = nn.LayerNorm(self.total_dim)
        
        # === Output Projections ===
        # Next memory prediction
        self.memory_predictor = nn.Linear(self.total_dim, input_dim)
        
        # Insight generation (what new thing did I learn?)
        self.insight_generator = nn.Linear(self.total_dim, world_dim)
        
        # Confidence estimator
        self.confidence_head = nn.Linear(self.total_dim, 1)
        
        print(f"🧠 HaciCognitiveNet initialized:")
        print(f"   World Model: {world_dim}-dim")
        print(f"   Personality: {personality_dim}-dim (8 dimensions)")
        print(f"   Retention Core: {n_retention_layers} layers, {n_heads} heads")
        print(f"   Total: {self.total_dim}-dim")
    
    def forward(self, 
                memory_embeddings: torch.Tensor,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            memory_embeddings: [B, S, 384] - sequence of memory embeddings
            return_components: if True, return all intermediate states
        
        Returns:
            dict with 'predicted_next', 'insight', 'confidence', and optionally components
        """
        B, S, _ = memory_embeddings.shape
        
        # 1. World Model
        world_state = self.world_model(memory_embeddings)  # [B, S, 512]
        
        # 2. Personality (context-aware)
        personality_state = self.personality(world_state)  # [B, 1024]
        personality_expanded = personality_state.unsqueeze(1).expand(-1, S, -1)  # [B, S, 1024]
        
        # 3. Fusion: World + Personality
        combined = torch.cat([world_state, personality_expanded], dim=-1)  # [B, S, 1536]
        
        # 4. Retention Core (4 layers)
        x = combined
        for norm, layer in zip(self.retention_norms, self.retention_layers):
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + x
        
        # 5. Final fusion
        x = self.fusion(x) + x  # residual
        x = self.output_norm(x)
        
        # 6. Outputs
        last_hidden = x[:, -1, :]  # [B, 1536]
        
        predicted_next = self.memory_predictor(last_hidden)  # [B, 384]
        insight = self.insight_generator(last_hidden)  # [B, 512]
        confidence = torch.sigmoid(self.confidence_head(last_hidden))  # [B, 1]
        
        result = {
            'predicted_next': predicted_next,
            'insight': insight,
            'confidence': confidence,
        }
        
        if return_components:
            result['world_state'] = world_state
            result['personality_state'] = personality_state
            result['combined'] = combined
            result['retention_output'] = x
            
            # Metacognition
            meta = self.metacognition(world_state)
            result.update(meta)
            
            # Emotional state
            emotion = self.emotion(world_state)
            result['emotion'] = emotion
        
        return result
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Multi-objective loss:
        1. Memory prediction loss (cosine + MSE)
        2. Insight consistency loss
        3. Confidence calibration loss
        """
        losses = {}
        
        # Memory prediction loss
        if 'next_embedding' in targets:
            pred = predictions['predicted_next']
            tgt = targets['next_embedding']
            
            cosine_sim = F.cosine_similarity(pred, tgt, dim=-1)
            losses['memory_cosine'] = 1.0 - cosine_sim.mean()
            losses['memory_mse'] = F.mse_loss(pred, tgt)
            losses['memory'] = losses['memory_cosine'] + losses['memory_mse']
        
        # Confidence loss (should correlate with prediction accuracy)
        if 'confidence' in predictions and 'next_embedding' in targets:
            pred = predictions['predicted_next']
            tgt = targets['next_embedding']
            conf = predictions['confidence'].squeeze(-1)
            
            actual_accuracy = F.cosine_similarity(pred, tgt, dim=-1).clamp(0, 1)
            losses['confidence'] = F.mse_loss(conf, actual_accuracy.detach())
        
        # Total loss
        total = sum(losses.values())
        losses['total'] = total
        
        return losses
    
    def get_cognitive_state(self) -> Dict:
        """Get current cognitive state snapshot."""
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
            'model_dim': self.total_dim,
            'world_dim': self.world_dim,
            'personality_dim': self.personality_dim,
        }
    
    def save_checkpoint(self, path: str):
        """Save full model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'world_dim': self.world_dim,
                'personality_dim': self.personality_dim,
                'total_dim': self.total_dim,
            },
            'cognitive_state': self.get_cognitive_state(),
        }
        torch.save(checkpoint, path)
        print(f"🧠 HaciCognitiveNet saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'auto'):
        """Load model from checkpoint."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            input_dim=384,
            world_dim=config['world_dim'],
            personality_dim=config['personality_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"🧠 HaciCognitiveNet loaded from {path}")
        return model


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("=== HaciCognitiveNet Test ===")
    model = HaciCognitiveNet()
    
    # Test forward
    test_input = torch.randn(2, 10, 384)  # batch=2, seq=10, embedding=384
    output = model(test_input, return_components=True)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"Predicted next: {output['predicted_next'].shape}")
    print(f"Insight: {output['insight'].shape}")
    print(f"Confidence: {output['confidence'].shape}")
    
    print(f"\nEmotional state: {output['emotion']}")
    print(f"Metacognition strategy: consolidate={output['strategy_consolidate']:.3f}, "
          f"explore={output['strategy_explore']:.3f}, "
          f"generalize={output['strategy_generalize']:.3f}")
    
    # Test loss
    targets = {'next_embedding': torch.randn(2, 384)}
    losses = model.compute_loss(output, targets)
    print(f"\nLosses: {losses}")
    
    # Test cognitive state
    state = model.get_cognitive_state()
    print(f"\nCognitive state: {state}")
    
    print("\n✅ HaciCognitiveNet test passed!")

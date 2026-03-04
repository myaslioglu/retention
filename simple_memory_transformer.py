"""
Simple Memory Transformer using only MultiScaleRetention layer.
For quick integration without full transformer dependencies.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from retention_layer import MultiScaleRetention

logger = logging.getLogger(__name__)


class SimpleMemoryTransformer(nn.Module):
    """
    Simple transformer using only MultiScaleRetention layer.
    Input: memory embeddings [batch, seq_len, embedding_dim]
    Output: predicted next embedding [batch, embedding_dim]
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Retention layers
        self.retention_layers = nn.ModuleList([
            MultiScaleRetention(
                d_model=embedding_dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                scales=n_heads,
                decay_min=0.3,
                decay_max=0.99,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(n_layers)
        ])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection (predict next embedding)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"SimpleMemoryTransformer initialized:")
        logger.info(f"  - Embedding dim: {embedding_dim}")
        logger.info(f"  - Layers: {n_layers}")
        logger.info(f"  - Heads: {n_heads}")
        logger.info(f"  - Max seq len: {max_seq_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [batch, seq_len, embedding_dim]
        returns: [batch, seq_len, embedding_dim]
        """
        # Apply retention layers
        for i in range(self.n_layers):
            # Layer norm before retention
            residual = x
            x = self.layer_norms[i](x)
            x = self.retention_layers[i](x)
            x = self.dropout(x)
            x = residual + x
            
            # FFN
            residual = x
            x = self.ffn(x)
            x = residual + x
        
        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        return x
    
    def predict_next(self, memory_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict next embedding.
        memory_sequence: [seq_len, embedding_dim] or [batch, seq_len, embedding_dim]
        returns: [embedding_dim] or [batch, embedding_dim]
        """
        if memory_sequence.dim() == 2:
            memory_sequence = memory_sequence.unsqueeze(0)
        
        predictions = self.forward(memory_sequence)
        next_pred = predictions[:, -1, :]
        
        if next_pred.shape[0] == 1:
            return next_pred.squeeze(0)
        return next_pred
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "cosine"
    ) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        if loss_type == "cosine":
            cosine_sim = torch.nn.functional.cosine_similarity(
                predictions.flatten(1),
                targets.flatten(1),
                dim=1
            )
            loss = 1.0 - cosine_sim.mean()
        elif loss_type == "mse":
            loss = torch.nn.functional.mse_loss(predictions, targets)
        elif loss_type == "combined":
            cosine_sim = torch.nn.functional.cosine_similarity(
                predictions.flatten(1),
                targets.flatten(1),
                dim=1
            )
            cosine_loss = 1.0 - cosine_sim.mean()
            mse_loss = torch.nn.functional.mse_loss(predictions, targets)
            loss = cosine_loss + mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'max_seq_len': self.max_seq_len
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'auto'):
        """Load model checkpoint."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            embedding_dim=checkpoint['embedding_dim'],
            n_heads=checkpoint['n_heads'],
            n_layers=checkpoint['n_layers'],
            max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    model = SimpleMemoryTransformer(embedding_dim=384, n_layers=2)
    
    # Test forward
    test_input = torch.randn(2, 5, 384)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predict_next
    next_pred = model.predict_next(test_input[0])
    print(f"Next prediction shape: {next_pred.shape}")
    
    print("Test passed!")
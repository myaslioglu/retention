"""
HaciCognitiveNet Trainer - Self-supervised training loop
Trains the cognitive network on memory embeddings with multi-objective loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CognitiveTrainer:
    """
    Training loop for HaciCognitiveNet.
    
    Training objectives:
    1. Memory prediction (predict next embedding from sequence)
    2. Insight consistency (insights should be coherent)
    3. Confidence calibration (confidence should match accuracy)
    4. Personality stability (personality shouldn't drift too fast)
    """
    
    def __init__(self,
                 model,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'auto',
                 checkpoint_dir: str = None):
        """
        Args:
            model: HaciCognitiveNet instance
            lr: Learning rate
            weight_decay: L2 regularization
            device: 'auto', 'cpu', 'cuda'
            checkpoint_dir: Where to save checkpoints
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for different components
        world_params = list(model.world_model.parameters())
        personality_params = list(model.personality.parameters())
        retention_params = (list(model.retention_layers.parameters()) + 
                          list(model.retention_norms.parameters()) + 
                          list(model.fusion.parameters()) + 
                          list(model.output_norm.parameters()))
        head_params = (list(model.memory_predictor.parameters()) + 
                      list(model.insight_generator.parameters()) + 
                      list(model.confidence_head.parameters()))
        
        self.optimizer = optim.AdamW([
            {'params': world_params, 'lr': lr},
            {'params': personality_params, 'lr': lr * 0.5},  # Slower for personality
            {'params': retention_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 2},  # Faster for output heads
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        logger.info(f"🏋️ CognitiveTrainer initialized (device={device})")
    
    def create_training_data(self, memory_texts: List[str], 
                            embedding_fn=None) -> List[Dict]:
        """
        Create training sequences from memory texts.
        
        Each training sample is a sequence of memory embeddings with a target.
        """
        if embedding_fn is None:
            # Simple deterministic embeddings for demo
            embedding_fn = self._simple_embed
        
        sequences = []
        for i, text in enumerate(memory_texts):
            embedding = embedding_fn(text)
            
            # Create sequence: this memory + context from neighbors
            if i > 0:
                context = embedding_fn(memory_texts[i - 1])
            else:
                context = torch.zeros_like(embedding)
            
            # Target: next memory embedding (or self for self-prediction)
            if i < len(memory_texts) - 1:
                target = embedding_fn(memory_texts[i + 1])
            else:
                target = embedding
            
            sequences.append({
                'input': torch.stack([context, embedding]),  # [2, 384]
                'target': target,  # [384]
            })
        
        return sequences
    
    def _simple_embed(self, text: str) -> torch.Tensor:
        """Simple deterministic embedding (for testing)."""
        hash_val = hash(text) % (2**31)
        torch.manual_seed(hash_val)
        return torch.randn(384)
    
    def train_epoch(self, 
                    train_data: List[Dict],
                    batch_size: int = 4,
                    verbose: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_memory_loss = 0
        total_confidence_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [train_data[idx] for idx in batch_indices]
            
            # Prepare batch
            inputs = torch.stack([s['input'] for s in batch]).to(self.device)  # [B, 2, 384]
            targets = torch.stack([s['target'] for s in batch]).to(self.device)  # [B, 384]
            
            # Forward
            self.optimizer.zero_grad()
            
            output = self.model(inputs, return_components=True)
            
            # Compute losses
            losses = self.model.compute_loss(output, {'next_embedding': targets})
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate
            total_loss += losses['total'].item()
            total_memory_loss += losses.get('memory', torch.tensor(0)).item()
            total_confidence_loss += losses.get('confidence', torch.tensor(0)).item()
            n_batches += 1
            self.global_step += 1
        
        self.scheduler.step()
        
        # Average losses
        avg_losses = {
            'total': total_loss / max(n_batches, 1),
            'memory': total_memory_loss / max(n_batches, 1),
            'confidence': total_confidence_loss / max(n_batches, 1),
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        self.epoch += 1
        
        # Record history
        self.training_history.append({
            'epoch': self.epoch,
            'step': self.global_step,
            **avg_losses,
            'timestamp': datetime.now().isoformat(),
        })
        
        if verbose:
            logger.info(f"  Epoch {self.epoch}: loss={avg_losses['total']:.4f}, "
                       f"memory={avg_losses['memory']:.4f}, "
                       f"lr={avg_losses['lr']:.6f}")
        
        return avg_losses
    
    def train(self,
              memory_texts: List[str],
              n_epochs: int = 50,
              batch_size: int = 4,
              eval_every: int = 5,
              patience: int = 10,
              verbose: bool = True) -> Dict:
        """
        Full training loop with early stopping.
        """
        if verbose:
            logger.info(f"🏋️ Starting training: {len(memory_texts)} memories, {n_epochs} epochs")
        
        # Create training data
        train_data = self.create_training_data(memory_texts)
        
        if verbose:
            logger.info(f"   Training samples: {len(train_data)}")
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            losses = self.train_epoch(train_data, batch_size, verbose=verbose)
            
            # Check for improvement
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"   Early stopping at epoch {self.epoch} (patience={patience})")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Save checkpoint
        self.save_checkpoint()
        
        elapsed = time.time() - start_time
        
        result = {
            'epochs_trained': self.epoch,
            'total_steps': self.global_step,
            'best_loss': self.best_loss,
            'final_loss': losses['total'],
            'duration_sec': elapsed,
            'checkpoint': str(self.checkpoint_dir / f"cognitive_epoch_{self.epoch:04d}.pt"),
        }
        
        if verbose:
            logger.info(f"✅ Training complete: {result['epochs_trained']} epochs, "
                       f"best_loss={result['best_loss']:.4f}, "
                       f"duration={elapsed:.1f}s")
        
        return result
    
    def save_checkpoint(self):
        """Save model and training state checkpoint."""
        path = self.checkpoint_dir / f"cognitive_epoch_{self.epoch:04d}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
        }, path)
        
        logger.info(f"💾 Checkpoint saved: {path}")
        return path
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"📂 Checkpoint loaded: {path} (epoch {self.epoch})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== CognitiveTrainer Test ===\n")
    
    from cognitive_net import HaciCognitiveNet
    
    # Create model
    model = HaciCognitiveNet()
    
    # Create trainer
    workspace = os.path.expanduser("~/.openclaw/workspace")
    checkpoint_dir = os.path.join(workspace, "haci_cognitive", "checkpoints")
    
    trainer = CognitiveTrainer(
        model=model,
        lr=1e-4,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Create dummy memory texts
    memories = [
        "Başkan kahve seviyor, özellikle espresso",
        "Duygu'nun pasaport durumu kritik",
        "Galatasaray taraftarı Başkan",
        "WhatsApp watcher moondream ile çalışıyor",
        "Retention system kuruldu",
        "Memory consolidation 23:55'te çalışıyor",
        "FAISS index 3 günde bir reindex ediliyor",
        "OpenClaw workspace yapılandırıldı",
        "Python virtual environment kuruldu",
        "Metacognition modülü eklendi",
        "Dreaming loop başlatıldı",
        "Personality vectors 8 boyutlu",
        "World model 512-dim çalışıyor",
        "Emotional state tracking aktif",
        "HaciCognitiveNet Level 1 tamamlandı",
    ]
    
    # Train
    result = trainer.train(
        memory_texts=memories,
        n_epochs=20,
        batch_size=4,
        verbose=True,
    )
    
    print(f"\n📊 Training Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Test cognitive state
    state = model.get_cognitive_state()
    print(f"\n🧠 Cognitive State:")
    print(f"  Personality: {state['personality']}")
    print(f"  Emotion: {state['emotion']}")
    
    print("\n✅ CognitiveTrainer test passed!")

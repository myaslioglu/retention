"""
Memory Transformer Training Pipeline
Adapted from original train.py for memory embedding prediction.
"""

import logging
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from pathlib import Path
import sys

# Add transformer module to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_memory_transformer import SimpleMemoryTransformer
from memory_dataset import MemoryEmbeddingsDataset, get_memory_dataset

# Simple config class
class SimpleConfig:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        self.model = type('ModelConfig', (), config_dict.get('model', {}))()
        self.training = type('TrainingConfig', (), config_dict.get('training', {}))()
        self.dataset = type('DatasetConfig', (), config_dict.get('dataset', {}))()
        self.loss = type('LossConfig', (), config_dict.get('loss', {}))()
        self.experiment = type('ExperimentConfig', (), config_dict.get('experiment', {}))()

logger = logging.getLogger(__name__)


class MemoryTrainer:
    """
    Trainer for MemoryTransformer model.
    """
    
    def __init__(
        self,
        model: SimpleMemoryTransformer,
        train_dataset: MemoryEmbeddingsDataset,
        val_dataset: MemoryEmbeddingsDataset = None,
        config: SimpleConfig = None,
        device: str = 'auto'
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or model.config
        
        # Device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Training parameters
        if config:
            self.batch_size = self.config.training.batch_size
            self.epochs = self.config.training.epochs
            self.learning_rate = self.config.training.learning_rate
        else:
            self.batch_size = 16
            self.epochs = 10
            self.learning_rate = 0.0005
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * len(train_dataset) // self.batch_size
        )
        
        # Loss type
        if config and hasattr(config, 'loss'):
            # Try to get loss type from config
            if hasattr(self.config.loss, 'type'):
                self.loss_type = self.config.loss.type
            else:
                self.loss_type = 'combined'
        else:
            self.loss_type = 'combined'
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # IterableDataset doesn't support shuffle
            num_workers=0  # No multiprocessing for memory dataset
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Logging
        self.wandb_run = None
        self.setup_wandb()
        
        logger.info(f"MemoryTrainer initialized:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - Loss type: {self.loss_type}")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Val samples: {len(val_dataset) if val_dataset else 0}")
    
    def setup_wandb(self):
        """Setup Weights & Biases logging if enabled."""
        if (self.config and hasattr(self.config, 'experiment') and 
            self.config.experiment.active and 
            self.config.experiment.backend == 'wandb' and
            WANDB_AVAILABLE):
            try:
                self.wandb_run = wandb.init(
                    project=self.config.experiment.name,
                    config={
                        'model': dict(self.config.model),
                        'training': dict(self.config.training),
                        'loss': dict(self.config.loss),
                        'embedding_dim': self.model.embedding_dim,
                        'loss_type': self.loss_type
                    },
                    mode=self.config.experiment.tracking
                )
                logger.info("W&B logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.wandb_run = None
        else:
            self.wandb_run = None
            logger.info("W&B logging disabled or not available")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (src_embeddings, tgt_embeddings) in enumerate(self.train_loader):
            # Move to device
            src_embeddings = src_embeddings.to(self.device)
            tgt_embeddings = tgt_embeddings.to(self.device)
            
            # Forward pass
            predictions = self.model(src_embeddings)
            
            # Compute loss (predict next embedding from sequence)
            # We predict the next embedding for each position
            # Targets are the next embeddings in sequence
            loss = self.model.compute_loss(
                predictions[:, -1, :],  # Last position prediction
                tgt_embeddings,          # Next embedding target
                loss_type=self.loss_type
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimization step
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")
                
                if self.wandb_run:
                    self.wandb_run.log({
                        'epoch': epoch,
                        'batch_loss': loss.item(),
                        'batch_idx': batch_idx,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for src_embeddings, tgt_embeddings in self.val_loader:
                src_embeddings = src_embeddings.to(self.device)
                tgt_embeddings = tgt_embeddings.to(self.device)
                
                predictions = self.model(src_embeddings)
                loss = self.model.compute_loss(
                    predictions[:, -1, :],
                    tgt_embeddings,
                    loss_type=self.loss_type
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def train(self, save_dir: Path = None):
        """
        Full training loop.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"Starting epoch {epoch}/{self.epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} training loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {epoch} validation loss: {val_loss:.6f}")
            
            # Log epoch metrics
            if self.wandb_run:
                self.wandb_run.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint
            if save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = save_dir / f"memory_transformer_epoch_{epoch}_loss_{val_loss:.4f}.pt"
                self.model.save_checkpoint(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping check (simple)
            if epoch > 5 and val_loss > best_val_loss * 1.1:
                logger.info(f"Validation loss increased, early stopping at epoch {epoch}")
                break
        
        # Save final model
        if save_dir:
            final_path = save_dir / "memory_transformer_final.pt"
            self.model.save_checkpoint(str(final_path))
            logger.info(f"Saved final model to {final_path}")
        
        # Close W&B
        if self.wandb_run:
            self.wandb_run.finish()
        
        return best_val_loss


def split_dataset(dataset: MemoryEmbeddingsDataset, train_ratio: float = 0.8):
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: MemoryEmbeddingsDataset
        train_ratio: Proportion of data for training
    
    Returns:
        (train_dataset, val_dataset)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Since it's an IterableDataset, we need to convert to list for splitting
    # For simplicity, we'll create two datasets with different indices
    # This is a simplified approach - for production, implement proper splitting
    
    # For now, return same dataset for both (no validation)
    # TODO: Implement proper dataset splitting
    return dataset, None


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Memory Transformer")
    parser.add_argument("--memory_dir", type=str, 
                       default="/Users/muratyaslioglu/.openclaw/workspace/memory",
                       help="Directory containing memory files")
    parser.add_argument("--config", type=str, default="transformer/config.toml",
                       help="Path to config file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                       help="Learning rate")
    parser.add_argument("--window_size", type=int, default=5,
                       help="Memory window size")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="SentenceTransformer model name")
    parser.add_argument("--loss_type", type=str, default="combined",
                       choices=["cosine", "mse", "combined"],
                       help="Loss function type")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists() and config_path.suffix == '.toml':
        # Try to load TOML config (original format)
        try:
            import tomllib
            with open(config_path, 'rb') as f:
                config_dict = tomllib.load(f)
            config = SimpleConfig(config_dict)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            config = None
    else:
        logger.warning(f"Config file not found or not TOML: {config_path}, using defaults")
        # Create minimal config
        config = None
    
    # Update config with command line arguments
    if config:
        config.training.batch_size = args.batch_size
        config.training.epochs = args.epochs
        config.training.learning_rate = args.learning_rate
        config.loss.type = args.loss_type
    
    # Create dataset
    memory_dir = Path(args.memory_dir)
    if not memory_dir.exists():
        logger.error(f"Memory directory not found: {memory_dir}")
        return
    
    dataset = MemoryEmbeddingsDataset(
        memory_dir=memory_dir,
        embedding_model=args.embedding_model,
        window_size=args.window_size
    )
    
    if len(dataset) == 0:
        logger.error("No training data available")
        return
    
    # Split dataset (simplified - no validation for now)
    train_dataset = dataset
    val_dataset = None  # TODO: Implement validation split
    
    # Create model
    embedding_dim = dataset.embedding_dim
    model = SimpleMemoryTransformer(
        embedding_dim=embedding_dim,
        n_heads=4,
        n_layers=2,
        max_seq_len=512
    )
    
    # Create trainer
    trainer = MemoryTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    save_dir = Path(args.save_dir)
    best_loss = trainer.train(save_dir=save_dir)
    
    logger.info(f"Training completed. Best validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
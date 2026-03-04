"""
Memory Embeddings Dataset for Transformer Training
Adapts the original WMT14 dataset to memory embeddings for OpenClaw retention system.
"""

import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional
import torch
from torch.utils.data import IterableDataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MemoryEmbeddingsDataset(IterableDataset):
    """
    Dataset that loads memory embeddings from OpenClaw memory files.
    
    Each sample consists of:
    - src_embedding: Current memory embedding (context)
    - tgt_embedding: Next memory embedding (prediction target)
    
    This creates a sequence prediction task where the model learns to predict
    the next memory state given the current one.
    """
    
    def __init__(
        self,
        memory_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_seq_len: int = 512,
        embedding_dim: int = 384,
        window_size: int = 5
    ):
        """
        Args:
            memory_dir: Directory containing memory markdown files
            embedding_model: SentenceTransformer model name
            max_seq_len: Maximum sequence length (not used for embeddings, kept for compatibility)
            embedding_dim: Dimension of embeddings (auto-detected from model)
            window_size: Number of consecutive memories to use as sequence
        """
        self.memory_dir = Path(memory_dir)
        self.embedding_model_name = embedding_model
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Load memory files
        self.memory_files = sorted(self.memory_dir.glob("*.md"))
        logger.info(f"Found {len(self.memory_files)} memory files")
        
        # Pre-load memory texts
        self.memory_texts = self._load_memory_texts()
        logger.info(f"Loaded {len(self.memory_texts)} memory texts")
        
        # Pre-compute embeddings (could be done lazily, but small dataset)
        self.embeddings = self._compute_embeddings()
        logger.info(f"Computed {len(self.embeddings)} memory embeddings")
    
    def _load_memory_texts(self) -> list:
        """Load memory texts from markdown files."""
        texts = []
        for mem_file in self.memory_files:
            try:
                with open(mem_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and len(content) > 10:  # Skip empty/short files
                        texts.append(content)
            except Exception as e:
                logger.warning(f"Error reading {mem_file}: {e}")
        return texts
    
    def _compute_embeddings(self) -> torch.Tensor:
        """Compute embeddings for all memory texts."""
        if not self.memory_texts:
            return torch.empty((0, self.embedding_dim))
        
        # Batch embedding computation
        embeddings = self.embedder.encode(
            self.memory_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def __len__(self) -> int:
        """Number of training samples (sliding windows)."""
        if len(self.embeddings) < self.window_size + 1:
            return 0
        return len(self.embeddings) - self.window_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield sliding windows of memory embeddings.
        
        For window_size=5:
        - Input: embeddings[i:i+5]  # 5 consecutive memories
        - Target: embeddings[i+5]   # Next memory
        
        This teaches the model to predict the next memory state
        given a sequence of previous memories.
        """
        if len(self.embeddings) < self.window_size + 1:
            logger.warning("Not enough embeddings for window size")
            return
        
        for i in range(len(self.embeddings) - self.window_size):
            # Input sequence: [window_size, embedding_dim]
            src_embeddings = self.embeddings[i:i+self.window_size]
            
            # Target: next embedding [embedding_dim]
            tgt_embedding = self.embeddings[i+self.window_size]
            
            # Reshape for transformer: [seq_len, embedding_dim] -> [embedding_dim, seq_len]?
            # We'll keep as [seq_len, embedding_dim] for now
            yield src_embeddings, tgt_embedding
    
    def get_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of training samples.
        Returns: (src_batch, tgt_batch)
        - src_batch: [batch_size, window_size, embedding_dim]
        - tgt_batch: [batch_size, embedding_dim]
        """
        src_batch = []
        tgt_batch = []
        
        iterator = iter(self)
        for _ in range(batch_size):
            try:
                src, tgt = next(iterator)
                src_batch.append(src)
                tgt_batch.append(tgt)
            except StopIteration:
                break
        
        if not src_batch:
            return torch.empty((0, self.window_size, self.embedding_dim)), torch.empty((0, self.embedding_dim))
        
        return torch.stack(src_batch), torch.stack(tgt_batch)


def get_memory_dataset(
    memory_dir: Path,
    config: dict,
    streaming: bool = False
) -> MemoryEmbeddingsDataset:
    """
    Factory function to create memory dataset.
    
    Args:
        memory_dir: Path to memory files
        config: Configuration dictionary with dataset parameters
        streaming: Not used for memory dataset (always full load)
    
    Returns:
        MemoryEmbeddingsDataset instance
    """
    return MemoryEmbeddingsDataset(
        memory_dir=memory_dir,
        embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2'),
        max_seq_len=config.get('max_seq_len', 512),
        embedding_dim=config.get('embedding_dim', 384),
        window_size=config.get('window_size', 5)
    )


if __name__ == "__main__":
    # Test the dataset
    import sys
    logging.basicConfig(level=logging.INFO)
    
    test_dir = Path("/Users/muratyaslioglu/.openclaw/workspace/memory")
    if test_dir.exists():
        dataset = MemoryEmbeddingsDataset(test_dir, window_size=3)
        print(f"Dataset length: {len(dataset)}")
        
        # Get one batch
        src, tgt = dataset.get_batch(batch_size=2)
        if len(src) > 0:
            print(f"Source batch shape: {src.shape}")
            print(f"Target batch shape: {tgt.shape}")
            print("Test passed!")
        else:
            print("Not enough data for batch")
    else:
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
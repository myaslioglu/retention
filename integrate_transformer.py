"""
Transformer Integration Pipeline
Integrates trained MemoryTransformer with OpenClaw retention system.
"""

import logging
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add transformer module to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_memory_transformer import SimpleMemoryTransformer
from memory_dataset import MemoryEmbeddingsDataset
from memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


class TransformerIntegration:
    """
    Integrates MemoryTransformer with OpenClaw retention system.
    
    Enhances memory consolidation with transformer-based:
    1. Memory similarity prediction
    2. Importance scoring refinement
    3. Context compression optimization
    4. Personality state updates
    """
    
    def __init__(
        self,
        model_path: str,
        memory_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = 'auto'
    ):
        """
        Args:
            model_path: Path to trained MemoryTransformer checkpoint
            memory_dir: Directory containing memory files
            embedding_model: SentenceTransformer model for embeddings
            device: 'cuda' or 'cpu'
        """
        # Load model
        logger.info(f"Loading SimpleMemoryTransformer from {model_path}")
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = SimpleMemoryTransformer.load_checkpoint(model_path, device=device)
        # Ensure model is on same device as embeddings
        self.model.to(self.device)
        self.model.eval()
        
        # Setup embedding model (same as training)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Memory directory
        self.memory_dir = Path(memory_dir)
        
        # Memory consolidator (for importance scoring)
        self.consolidator = MemoryConsolidator()
        
        # Cache for recent embeddings
        self.embedding_cache = {}
        
        logger.info(f"TransformerIntegration initialized:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Embedding dim: {self.embedding_dim}")
        logger.info(f"  - Memory dir: {memory_dir}")
    
    def get_memory_embedding(self, memory_text: str) -> torch.Tensor:
        """Get embedding for memory text."""
        # Simple cache to avoid recomputation
        if memory_text in self.embedding_cache:
            return self.embedding_cache[memory_text]
        
        embedding = self.embedder.encode(
            memory_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        # Move to same device as model
        embedding = embedding.to(self.device)
        
        self.embedding_cache[memory_text] = embedding
        return embedding
    
    def predict_memory_importance(
        self,
        memory_text: str,
        context_memories: List[str],
        context_size: int = 5
    ) -> float:
        """
        Predict memory importance using transformer.
        
        Args:
            memory_text: Current memory text
            context_memories: List of recent memory texts
            context_size: Number of context memories to use
        
        Returns:
            Predicted importance score (0-1)
        """
        # Get embeddings
        memory_embedding = self.get_memory_embedding(memory_text)
        
        # Get context embeddings
        context_texts = context_memories[-context_size:] if context_memories else []
        if not context_texts:
            # No context, return base importance from consolidator
            return self.consolidator._calculate_importance_score(memory_text)
        
        context_embeddings = [
            self.get_memory_embedding(text) for text in context_texts
        ]
        
        # Create sequence tensor: [context_size, embedding_dim]
        sequence = torch.stack(context_embeddings).unsqueeze(0)  # [1, seq_len, dim]
        
        # Predict next embedding
        with torch.no_grad():
            predicted_next = self.model.predict_next(sequence)  # [embedding_dim]
        
        # Compare predicted vs actual
        similarity = torch.nn.functional.cosine_similarity(
            predicted_next.unsqueeze(0),
            memory_embedding.unsqueeze(0),
            dim=1
        ).item()
        
        # Importance based on surprise (1 - similarity) and consolidator score
        surprise = 1.0 - similarity
        base_importance = self.consolidator._calculate_importance_score(memory_text)
        
        # Combine: higher surprise = higher importance
        combined_importance = 0.7 * base_importance + 0.3 * surprise
        
        # Clip to [0, 1]
        combined_importance = max(0.0, min(1.0, combined_importance))
        
        logger.debug(f"Memory importance prediction:")
        logger.debug(f"  - Base importance: {base_importance:.3f}")
        logger.debug(f"  - Surprise: {surprise:.3f}")
        logger.debug(f"  - Combined: {combined_importance:.3f}")
        
        return combined_importance
    
    def enhance_memory_consolidation(
        self,
        memories: List[Dict[str, Any]],
        use_transformer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhance memory consolidation with transformer predictions.
        
        Args:
            memories: List of memory dicts from consolidator
            use_transformer: Whether to use transformer for importance scoring
        
        Returns:
            Enhanced memories with transformer-refined scores
        """
        if not use_transformer or len(memories) < 2:
            return memories
        
        enhanced_memories = []
        
        # Get context for each memory
        for i, memory in enumerate(memories):
            memory_text = memory.get('text', '')
            if not memory_text:
                enhanced_memories.append(memory)
                continue
            
            # Get context (previous memories)
            context_memories = []
            for j in range(max(0, i - 10), i):
                if j < len(memories):
                    context_text = memories[j].get('text', '')
                    if context_text:
                        context_memories.append(context_text)
            
            # Calculate importance
            if use_transformer and context_memories:
                importance = self.predict_memory_importance(
                    memory_text=memory_text,
                    context_memories=context_memories
                )
            else:
                importance = memory.get('importance', 0.5)
            
            # Update memory
            enhanced_memory = memory.copy()
            enhanced_memory['importance'] = importance
            enhanced_memory['transformer_enhanced'] = use_transformer
            
            enhanced_memories.append(enhanced_memory)
        
        return enhanced_memories
    
    def compress_context_with_transformer(
        self,
        context_memories: List[str],
        target_tokens: int = 1000
    ) -> str:
        """
        Compress context using transformer predictions.
        
        Args:
            context_memories: List of memory texts
            target_tokens: Target token count after compression
        
        Returns:
            Compressed context summary
        """
        if not context_memories:
            return ""
        
        # Get embeddings for all memories
        embeddings = []
        for text in context_memories:
            if text:
                emb = self.get_memory_embedding(text)
                embeddings.append(emb)
        
        if not embeddings:
            return ""
        
        # Create sequence tensor
        sequence = torch.stack(embeddings).unsqueeze(0)  # [1, seq_len, dim]
        
        # Use transformer to predict summary embedding
        with torch.no_grad():
            # Encode entire sequence
            encoded = self.model(sequence)  # [1, seq_len, dim]
            
            # Get attention-weighted summary (simple mean of encoded)
            summary_embedding = encoded.mean(dim=1).squeeze(0)  # [dim]
        
        # Find memory most similar to summary
        summary_similarities = []
        for i, emb in enumerate(embeddings):
            sim = torch.nn.functional.cosine_similarity(
                summary_embedding.unsqueeze(0),
                emb.unsqueeze(0),
                dim=1
            ).item()
            summary_similarities.append((i, sim))
        
        # Sort by similarity (descending)
        summary_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top memories based on target token count
        selected_indices = []
        selected_texts = []
        total_tokens = 0
        
        # Simple token estimation (approx 4 chars per token)
        for idx, sim in summary_similarities:
            if idx >= len(context_memories):
                continue
            
            text = context_memories[idx]
            text_tokens = len(text) // 4
            
            if total_tokens + text_tokens <= target_tokens:
                selected_indices.append(idx)
                selected_texts.append(text)
                total_tokens += text_tokens
            else:
                break
        
        # Sort selected texts by original order
        selected_indices.sort()
        compressed_context = "\n\n".join(
            [context_memories[i] for i in selected_indices]
        )
        
        logger.info(f"Context compression:")
        logger.info(f"  - Original memories: {len(context_memories)}")
        logger.info(f"  - Selected memories: {len(selected_texts)}")
        logger.info(f"  - Original tokens: ~{len(''.join(context_memories)) // 4}")
        logger.info(f"  - Compressed tokens: ~{total_tokens}")
        logger.info(f"  - Compression ratio: {total_tokens/(len(''.join(context_memories))//4 + 1e-6):.2f}")
        
        return compressed_context
    
    def update_personality_state(
        self,
        memory_sequence: List[str],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update personality state using transformer.
        
        Args:
            memory_sequence: Recent memory sequence
            current_state: Current personality state
        
        Returns:
            Updated personality state
        """
        if not memory_sequence:
            return current_state
        
        # Get embeddings
        embeddings = []
        for text in memory_sequence[-10:]:  # Last 10 memories
            if text:
                emb = self.get_memory_embedding(text)
                embeddings.append(emb)
        
        if not embeddings:
            return current_state
        
        # Create sequence tensor
        sequence = torch.stack(embeddings).unsqueeze(0)  # [1, seq_len, dim]
        
        # Encode with transformer
        with torch.no_grad():
            encoded = self.model(sequence)  # [1, seq_len, dim]
            
            # Extract personality features (mean of encoded)
            personality_embedding = encoded.mean(dim=1).squeeze(0)  # [dim]
        
        # Update state
        updated_state = current_state.copy()
        
        # Convert embedding to dict (store as list)
        if 'embeddings' not in updated_state:
            updated_state['embeddings'] = []
        
        updated_state['embeddings'].append(personality_embedding.cpu().numpy().tolist())
        
        # Keep only recent embeddings
        if len(updated_state['embeddings']) > 100:
            updated_state['embeddings'] = updated_state['embeddings'][-100:]
        
        # Calculate average personality vector
        if updated_state['embeddings']:
            import numpy as np
            avg_embedding = np.mean(updated_state['embeddings'], axis=0)
            updated_state['avg_personality'] = avg_embedding.tolist()
        
        logger.info("Personality state updated with transformer")
        
        return updated_state


def main():
    """Test integration pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Transformer Integration")
    parser.add_argument("--model", type=str, default="./checkpoints/memory_transformer_final.pt",
                       help="Path to trained model")
    parser.add_argument("--memory_dir", type=str, 
                       default="/Users/muratyaslioglu/.openclaw/workspace/memory",
                       help="Memory directory")
    parser.add_argument("--test_memory", type=str, 
                       default="Başkan external hard drive'a yedekleme yaparken 'Exec failed (crisp-se, signal SIGTERM)' hatası aldı.",
                       help="Test memory text")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        logger.warning("Please train a model first with train_memory.py")
        return
    
    # Initialize integration
    integration = TransformerIntegration(
        model_path=str(model_path),
        memory_dir=Path(args.memory_dir)
    )
    
    # Test with sample memory
    test_text = args.test_memory
    
    # Simulate context memories
    context_memories = [
        "Backup script created and tested successfully.",
        "Cron job for automatic backups configured.",
        "External drive permissions resolved.",
        "Security audit completed with patch recommendations."
    ]
    
    # Test importance prediction
    importance = integration.predict_memory_importance(
        memory_text=test_text,
        context_memories=context_memories
    )
    
    print(f"\n🧠 Transformer Integration Test Results:")
    print(f"  Test memory: {test_text[:100]}...")
    print(f"  Predicted importance: {importance:.3f}")
    
    # Test context compression
    compressed = integration.compress_context_with_transformer(
        context_memories=context_memories + [test_text],
        target_tokens=200
    )
    
    print(f"\n📦 Context Compression:")
    print(f"  Original memories: {len(context_memories) + 1}")
    print(f"  Compressed summary length: {len(compressed)} chars")
    if compressed:
        print(f"  Preview: {compressed[:200]}...")
    
    # Test personality state update
    personality_state = {'embeddings': []}
    updated_state = integration.update_personality_state(
        memory_sequence=context_memories + [test_text],
        current_state=personality_state
    )
    
    print(f"\n👤 Personality State Update:")
    print(f"  Embeddings stored: {len(updated_state.get('embeddings', []))}")
    if 'avg_personality' in updated_state:
        print(f"  Average personality vector length: {len(updated_state['avg_personality'])}")
    
    print("\n✅ Integration test completed!")


if __name__ == "__main__":
    main()
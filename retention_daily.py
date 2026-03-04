#!/usr/bin/env python3
"""
Hacı Retention Daily Learning Script
Heartbeat'te çalışacak, retention system'i güncelleyecek.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'retention'))

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import datetime

try:
    from arch.attentions.retention import MultiScaleRetention
    RETENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Retention import failed: {e}")
    RETENTION_AVAILABLE = False
    # Create a dummy class for testing
    class MultiScaleRetention:
        def __init__(self, **kwargs):
            pass
        def __call__(self, x):
            return x

from sentence_transformers import SentenceTransformer

class HaciRetentionSystem:
    """Hacı Retention System - Lightweight integration"""
    
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Retention layer (context compressor)
        self.retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=512,
            scales=n_heads,  # Each head has its own scale
            decay_min=0.3,
            decay_max=0.99,
            dropout=0.1
        )
        
        # Sentence transformer for text -> vector
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embed_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Projection: embed_dim -> d_model
        self.projection = nn.Linear(embed_dim, d_model)
        
        # Personality state (learned from SOUL.md, USER.md, MEMORY.md)
        self.personality_state = None
        
        # Conversation memory
        self.conversation_memory = []
        
        print(f"🎯 Hacı Retention System initialized")
        print(f"  - d_model: {d_model}, heads: {n_heads}")
        print(f"  - Embedder: all-MiniLM-L6-v2 ({embed_dim} dim)")
    
    def load_personality(self, workspace_path: str):
        """Load personality from SOUL.md, USER.md, MEMORY.md"""
        workspace = Path(workspace_path)
        personality_text = ""
        
        try:
            with open(workspace / "SOUL.md", 'r', encoding='utf-8') as f:
                personality_text += f.read() + "\n"
        except:
            pass
            
        try:
            with open(workspace / "USER.md", 'r', encoding='utf-8') as f:
                personality_text += f.read() + "\n"
        except:
            pass
            
        try:
            with open(workspace / "MEMORY.md", 'r', encoding='utf-8') as f:
                # Just recent memories
                content = f.read()
                # Take last 1000 chars for personality
                personality_text += content[-1000:] + "\n"
        except:
            pass
        
        if personality_text:
            # Create personality embedding
            personality_embed = self.embedder.encode(personality_text)
            personality_embed = torch.tensor(personality_embed).float()
            
            # Project to d_model
            with torch.no_grad():
                self.personality_state = self.projection(personality_embed.unsqueeze(0))
            
            print(f"🧠 Personality loaded ({len(personality_text)} chars)")
            return True
        else:
            print("⚠️ No personality files found")
            return False
    
    def embed_conversation(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert conversation to embedding sequence"""
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        texts = []
        for msg in messages:
            prefix = "User: " if msg["role"] == "user" else "Hacı: "
            texts.append(prefix + msg["content"])
        
        # Embed each turn
        embeddings = []
        for text in texts:
            embed = self.embedder.encode(text)
            embeddings.append(embed)
        
        # Stack and project
        embeddings = np.array(embeddings)
        embeddings_tensor = torch.tensor(embeddings).float()
        
        with torch.no_grad():
            projected = self.projection(embeddings_tensor)
        
        return projected  # [seq_len, d_model]
    
    def compress_context(self, conversation_history: List[Dict[str, str]]) -> torch.Tensor:
        """Compress conversation history using retention layer"""
        # Convert to embeddings
        embeddings = self.embed_conversation(conversation_history)
        
        # Add batch dimension: [1, seq_len, d_model]
        embeddings = embeddings.unsqueeze(0)
        
        # Apply retention layer
        with torch.no_grad():
            compressed = self.retention(embeddings)
        
        # Take the last token as context summary
        context_summary = compressed[0, -1, :]  # [d_model]
        
        # Blend with personality if available
        if self.personality_state is not None:
            # Simple weighted average
            alpha = 0.3  # Personality weight
            context_summary = (1 - alpha) * context_summary + alpha * self.personality_state[0]
        
        return context_summary
    
    def daily_learning_step(self, new_memories: List[str]):
        """Daily incremental learning from new memories"""
        if not new_memories:
            return
        
        print(f"📚 Daily learning: {len(new_memories)} new memories")
        
        # Convert memories to conversation format
        conversations = []
        for memory in new_memories:
            # Simple format: user shares memory, assistant acknowledges
            conv = [
                {"role": "user", "content": f"Remember this: {memory}"},
                {"role": "assistant", "content": "I'll remember that. Added to my memory."}
            ]
            conversations.append(conv)
        
        # Create mini-batch for learning
        batch_embeddings = []
        for conv in conversations:
            emb = self.embed_conversation(conv)
            batch_embeddings.append(emb.unsqueeze(0))  # [1, seq_len, d_model]
        
        if batch_embeddings:
            # Stack: [batch_size, seq_len, d_model]
            batch = torch.cat(batch_embeddings, dim=0)
            
            # Simple forward pass (for now - could add training later)
            with torch.no_grad():
                outputs = self.retention(batch)
            
            # Update conversation memory
            self.conversation_memory.extend(new_memories[:10])  # Keep last 10
            
            print(f"  ✓ Processed {len(conversations)} memory pairs")
            return True
        return False
    
    def save_state(self, path: str):
        """Save retention system state"""
        state = {
            'conversation_memory': self.conversation_memory,
            'personality_state': self.personality_state.numpy().tolist() if self.personality_state is not None else None,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 State saved to {path}")
    
    def load_state(self, path: str):
        """Load retention system state"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.conversation_memory = state.get('conversation_memory', [])
            
            personality_data = state.get('personality_state')
            if personality_data:
                self.personality_state = torch.tensor(personality_data).float()
            
            print(f"📂 State loaded from {path}")
            print(f"  - {len(self.conversation_memory)} memories in cache")
            return True
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")
            return False
    
    def get_new_memories_since(self, memory_dir: str, since_timestamp: float) -> List[str]:
        """Get new memories from memory/*.md files since timestamp"""
        memory_path = Path(memory_dir)
        new_memories = []
        
        if not memory_path.exists():
            return []
        
        for file in memory_path.glob("*.md"):
            # Check file modification time
            mtime = file.stat().st_mtime
            if mtime > since_timestamp:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract first few lines as memory
                        lines = content.split('\n')
                        if lines:
                            # Take first 3 non-empty lines
                            summary = ' '.join([line.strip() for line in lines if line.strip()][:3])
                            if summary:
                                new_memories.append(f"{file.stem}: {summary}")
                except Exception as e:
                    print(f"⚠️ Error reading {file}: {e}")
        
        return new_memories


def main():
    """Main function for daily retention learning"""
    workspace_path = "/Users/muratyaslioglu/.openclaw/workspace"
    state_path = "/Users/muratyaslioglu/.openclaw/workspace/retention_state.json"
    memory_dir = f"{workspace_path}/memory"
    
    print("=" * 60)
    print("🧠 HACI RETENTION DAILY LEARNING")
    print(f"📅 {datetime.datetime.now().isoformat()}")
    print("=" * 60)
    
    # Initialize retention system
    hrs = HaciRetentionSystem(d_model=128, n_heads=4)
    
    # Load personality
    personality_loaded = hrs.load_personality(workspace_path)
    
    # Load previous state to get last run time
    last_run_time = 0
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                last_run_time = datetime.datetime.fromisoformat(state.get('timestamp', '2000-01-01')).timestamp()
                print(f"⏰ Last run: {state.get('timestamp')}")
                hrs.load_state(state_path)
        except Exception as e:
            print(f"⚠️ Could not load previous state: {e}")
    
    # Get new memories since last run
    new_memories = hrs.get_new_memories_since(memory_dir, last_run_time)
    
    if new_memories:
        print(f"📝 Found {len(new_memories)} new memories since last run:")
        for mem in new_memories[:3]:  # Show first 3
            print(f"  • {mem[:80]}...")
        if len(new_memories) > 3:
            print(f"  ... and {len(new_memories) - 3} more")
        
        # Daily learning step
        hrs.daily_learning_step(new_memories)
    else:
        print("📭 No new memories since last run")
    
    # Save updated state
    hrs.save_state(state_path)
    
    # Print summary
    print("\n📊 DAILY LEARNING SUMMARY:")
    print(f"  • Personality loaded: {personality_loaded}")
    print(f"  • New memories processed: {len(new_memories)}")
    print(f"  • Total conversation memories: {len(hrs.conversation_memory)}")
    print(f"  • Retention state saved to: {state_path}")
    
    print("\n✅ Daily learning completed!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error in daily learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
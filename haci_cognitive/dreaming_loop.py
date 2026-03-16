"""
Dreaming Loop - Background Memory Synthesis
Runs during low-activity periods to:
1. Consolidate memories (compress + connect)
2. Generate new insights from existing knowledge
3. Predict future patterns
4. Clean up redundant information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time
import random
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DreamPhase:
    """Represents a single dreaming phase."""
    
    CONSOLIDATE = "consolidate"   # Compress recent memories
    EXPLORE = "explore"           # Find novel connections
    GENERALIZE = "generalize"     # Abstract patterns
    PRUNE = "prune"               # Remove redundancy
    
    @classmethod
    def all_phases(cls):
        return [cls.CONSOLIDATE, cls.EXPLORE, cls.GENERALIZE, cls.PRUNE]


class DreamingLoop:
    """
    Background dreaming system that synthesizes memories during idle periods.
    
    Architecture:
    - Takes memory embeddings from FAISS/retention
    - Runs through HaciCognitiveNet in dream mode
    - Generates: new insights, pattern connections, consolidated memories
    - Updates: world model, personality state, knowledge graph
    """
    
    def __init__(self, 
                 model,
                 memory_dir: str,
                 output_dir: str,
                 embedding_dim: int = 384,
                 device: str = 'auto'):
        """
        Args:
            model: HaciCognitiveNet instance
            memory_dir: Directory with memory files (faiss, embeddings)
            output_dir: Where to save dream outputs
            embedding_dim: Dimension of memory embeddings
            device: 'auto', 'cpu', 'cuda'
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model
        self.device = device
        self.memory_dir = Path(memory_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        
        # Dream state
        self.current_phase = DreamPhase.CONSOLIDATE
        self.dream_cycles = 0
        self.insights_generated = []
        self.connections_found = []
        
        # Configuration
        self.config = {
            'cycle_duration_sec': 60,       # How long each cycle runs
            'insights_per_cycle': 3,         # Max insights per cycle
            'connection_threshold': 0.7,     # Cosine similarity for connections
            'prune_threshold': 0.95,         # Redundancy threshold
            'phase_rotation': True,          # Rotate through phases
            'max_dream_duration': 1800,      # Max 30 min dream session
        }
        
        logger.info(f"🌙 DreamingLoop initialized (device={device})")
    
    def load_recent_memories(self, n: int = 50) -> List[Dict]:
        """Load recent memories from workspace."""
        memories = []
        
        # Try loading from memory files
        memory_files = sorted(self.memory_dir.glob("*.md"), reverse=True)[:n]
        
        for f in memory_files:
            try:
                content = f.read_text()
                memories.append({
                    'source': f.name,
                    'content': content[:2000],  # Truncate long files
                    'timestamp': f.stat().st_mtime,
                })
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        logger.info(f"📖 Loaded {len(memories)} memories for dreaming")
        return memories
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Simple text embedding (placeholder - should use sentence-transformers in prod)."""
        # For now, create a deterministic hash-based embedding
        hash_val = hash(text) % (2**31)
        torch.manual_seed(hash_val)
        return torch.randn(self.embedding_dim)
    
    def consolidate_memories(self, memories: List[Dict]) -> Dict:
        """
        Consolidation Phase: Compress related memories into summary representations.
        """
        logger.info("💤 [DREAM] Consolidation phase started")
        
        # Group similar memories by content similarity
        if not memories:
            return {'type': 'consolidation', 'summary': 'No memories to consolidate'}
        
        # Create embeddings for all memories
        embeddings = torch.stack([
            self.embed_text(m['content']) for m in memories
        ])  # [N, 384]
        
        # Find clusters using simple cosine similarity
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=-1
        )  # [N, N]
        
        # Group into clusters (simplified)
        clusters = []
        used = set()
        threshold = 0.3
        
        for i in range(len(memories)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(memories)):
                if j not in used and sim_matrix[i, j] > threshold:
                    cluster.append(j)
                    used.add(j)
            if len(cluster) > 0:
                clusters.append(cluster)
        
        # Create consolidated summary
        summaries = []
        for cluster in clusters:
            cluster_memories = [memories[idx] for idx in cluster]
            summary = {
                'size': len(cluster),
                'sources': [m['source'] for m in cluster_memories],
                'content_preview': cluster_memories[0]['content'][:200],
                'timestamp_range': (
                    min(m['timestamp'] for m in cluster_memories),
                    max(m['timestamp'] for m in cluster_memories),
                ),
            }
            summaries.append(summary)
        
        result = {
            'type': 'consolidation',
            'n_clusters': len(clusters),
            'n_memories': len(memories),
            'compression_ratio': len(memories) / max(len(clusters), 1),
            'clusters': summaries,
        }
        
        logger.info(f"  Consolidated {len(memories)} memories into {len(clusters)} clusters")
        return result
    
    def explore_connections(self, memories: List[Dict]) -> Dict:
        """
        Exploration Phase: Find novel connections between seemingly unrelated memories.
        """
        logger.info("🔍 [DREAM] Exploration phase started")
        
        if len(memories) < 2:
            return {'type': 'exploration', 'connections': []}
        
        # Embed all memories
        embeddings = torch.stack([
            self.embed_text(m['content']) for m in memories
        ])
        
        # Find cross-topic connections (memories from different sources that are similar)
        connections = []
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                src_i = memories[i]['source']
                src_j = memories[j]['source']
                
                # Only connect from different sources
                if src_i == src_j:
                    continue
                
                sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                    dim=-1
                ).item()
                
                if sim > self.config['connection_threshold']:
                    connections.append({
                        'memory_a': memories[i]['source'],
                        'memory_b': memories[j]['source'],
                        'similarity': round(sim, 4),
                        'preview_a': memories[i]['content'][:100],
                        'preview_b': memories[j]['content'][:100],
                    })
        
        # Sort by similarity
        connections.sort(key=lambda x: x['similarity'], reverse=True)
        connections = connections[:10]  # Top 10
        
        self.connections_found.extend(connections)
        
        result = {
            'type': 'exploration',
            'n_connections': len(connections),
            'connections': connections,
        }
        
        logger.info(f"  Found {len(connections)} novel connections")
        return result
    
    def generalize_patterns(self, memories: List[Dict]) -> Dict:
        """
        Generalization Phase: Extract abstract patterns from specific memories.
        """
        logger.info("🎯 [DREAM] Generalization phase started")
        
        # Extract key themes from memory contents
        all_content = ' '.join(m['content'] for m in memories)
        words = all_content.lower().split()
        
        # Simple word frequency analysis
        word_freq = {}
        for w in words:
            if len(w) > 3:  # Skip short words
                word_freq[w] = word_freq.get(w, 0) + 1
        
        # Top themes
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        patterns = {
            'type': 'generalization',
            'n_memories_analyzed': len(memories),
            'top_themes': [{'word': w, 'frequency': f} for w, f in themes],
            'patterns_discovered': [
                f"Frequent topic: '{themes[0][0]}'" if themes else "No clear patterns",
                f"Memory diversity: {len(set(m['source'] for m in memories))} sources",
            ],
        }
        
        logger.info(f"  Identified {len(themes)} themes")
        return patterns
    
    def prune_redundancy(self, memories: List[Dict]) -> Dict:
        """
        Pruning Phase: Identify redundant memories for potential cleanup.
        """
        logger.info("✂️ [DREAM] Pruning phase started")
        
        if len(memories) < 2:
            return {'type': 'prune', 'redundant_pairs': []}
        
        embeddings = torch.stack([
            self.embed_text(m['content']) for m in memories
        ])
        
        redundant = []
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                    dim=-1
                ).item()
                
                if sim > self.config['prune_threshold']:
                    redundant.append({
                        'memory_a': memories[i]['source'],
                        'memory_b': memories[j]['source'],
                        'similarity': round(sim, 4),
                    })
        
        result = {
            'type': 'prune',
            'n_redundant': len(redundant),
            'redundant_pairs': redundant,
        }
        
        logger.info(f"  Found {len(redundant)} redundant pairs")
        return result
    
    def generate_insights(self, dream_results: List[Dict]) -> List[Dict]:
        """
        Generate actionable insights from dream results.
        """
        insights = []
        
        for result in dream_results:
            if result['type'] == 'consolidation':
                if result.get('compression_ratio', 0) > 3:
                    insights.append({
                        'type': 'high_compression',
                        'message': f"Memories highly compressible ({result['compression_ratio']:.1f}x). Consider periodic cleanup.",
                        'action': 'schedule_consolidation',
                    })
            
            elif result['type'] == 'exploration':
                for conn in result.get('connections', [])[:3]:
                    insights.append({
                        'type': 'novel_connection',
                        'message': f"Found connection: {conn['memory_a']} ↔ {conn['memory_b']} (sim: {conn['similarity']:.3f})",
                        'action': 'investigate',
                    })
            
            elif result['type'] == 'generalization':
                themes = result.get('top_themes', [])
                if themes:
                    top_theme = themes[0]['word']
                    insights.append({
                        'type': 'emerging_theme',
                        'message': f"Dominant theme: '{top_theme}' - consider deeper exploration",
                        'action': 'research',
                    })
        
        self.insights_generated.extend(insights)
        return insights
    
    def run_dream_cycle(self, memories: Optional[List[Dict]] = None) -> Dict:
        """
        Run a complete dream cycle through all phases.
        """
        start_time = time.time()
        logger.info("🌙 === DREAM CYCLE STARTED ===")
        
        # Load memories if not provided
        if memories is None:
            memories = self.load_recent_memories()
        
        if not memories:
            logger.warning("No memories available for dreaming")
            return {'status': 'no_memories'}
        
        # Run through all phases
        results = []
        
        if self.config['phase_rotation']:
            # Full cycle
            results.append(self.consolidate_memories(memories))
            results.append(self.explore_connections(memories))
            results.append(self.generalize_patterns(memories))
            results.append(self.prune_redundancy(memories))
        else:
            # Single phase
            phase_map = {
                DreamPhase.CONSOLIDATE: self.consolidate_memories,
                DreamPhase.EXPLORE: self.explore_connections,
                DreamPhase.GENERALIZE: self.generalize_patterns,
                DreamPhase.PRUNE: self.prune_redundancy,
            }
            results.append(phase_map[self.current_phase](memories))
        
        # Generate insights
        insights = self.generate_insights(results)
        
        # Compile dream report
        dream_report = {
            'cycle_id': self.dream_cycles,
            'timestamp': time.time(),
            'duration_sec': time.time() - start_time,
            'memories_processed': len(memories),
            'phases': results,
            'insights': insights,
            'total_insights_so_far': len(self.insights_generated),
            'total_connections_so_far': len(self.connections_found),
        }
        
        # Save dream report
        report_path = self.output_dir / f"dream_cycle_{self.dream_cycles:04d}.json"
        with open(report_path, 'w') as f:
            json.dump(dream_report, f, indent=2, default=str)
        
        self.dream_cycles += 1
        
        logger.info(f"🌙 === DREAM CYCLE COMPLETE ({dream_report['duration_sec']:.1f}s) ===")
        logger.info(f"   Insights: {len(insights)}")
        logger.info(f"   Report: {report_path}")
        
        return dream_report
    
    def get_dream_summary(self) -> Dict:
        """Get summary of all dreaming activity."""
        return {
            'total_cycles': self.dream_cycles,
            'total_insights': len(self.insights_generated),
            'total_connections': len(self.connections_found),
            'recent_insights': self.insights_generated[-5:] if self.insights_generated else [],
            'recent_connections': self.connections_found[-5:] if self.connections_found else [],
            'config': self.config,
        }


class AutonomousDreamRunner:
    """
    Standalone dream runner that can be scheduled via cron or run continuously.
    Integrates with HaciCognitiveNet and workspace.
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.dream_dir = self.workspace / "dreams"
        self.dream_dir.mkdir(exist_ok=True)
        
        # Will be initialized when model is available
        self.model = None
        self.dreaming = None
        
        self.state_file = self.workspace / "dream_state.json"
        self.load_state()
    
    def load_state(self):
        """Load dream runner state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                'total_cycles': 0,
                'last_dream': None,
                'total_insights': 0,
                'total_connections': 0,
                'enabled': True,
            }
    
    def save_state(self):
        """Save dream runner state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def run(self, max_cycles: int = 1):
        """Run dream cycles."""
        if not self.state['enabled']:
            logger.info("Dreaming is disabled")
            return
        
        # Initialize dreaming loop (without model for now - text-based)
        dreaming = DreamingLoop(
            model=None,
            memory_dir=str(self.workspace / "memory"),
            output_dir=str(self.dream_dir),
        )
        
        for i in range(max_cycles):
            logger.info(f"\n🌙 Dream cycle {i + 1}/{max_cycles}")
            report = dreaming.run_dream_cycle()
            
            self.state['total_cycles'] += 1
            self.state['last_dream'] = time.time()
            self.state['total_insights'] += len(report.get('insights', []))
        
        self.save_state()
        logger.info(f"\n🌙 Dreaming complete. Total cycles: {self.state['total_cycles']}")
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== Dreaming Loop Test ===\n")
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    runner = AutonomousDreamRunner(workspace)
    report = runner.run(max_cycles=1)
    
    print(f"\nDream report summary:")
    print(f"  Status: {report.get('status', 'completed')}")
    print(f"  Memories: {report.get('memories_processed', 0)}")
    print(f"  Insights: {len(report.get('insights', []))}")
    for insight in report.get('insights', []):
        print(f"    💡 {insight['message']}")

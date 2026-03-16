"""
Cognitive State Manager - Persistent cognitive state management
Handles:
- Saving/loading cognitive state (world model, personality, emotions, metacognition)
- State evolution over time
- Integration with daily learning loop
- Cognitive metrics tracking
"""

import torch
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CognitiveStateManager:
    """
    Manages persistent cognitive state for HaciCognitiveNet.
    
    State is stored as JSON (human-readable) + tensor files (binary).
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.state_dir = self.workspace / "cognitive_state"
        self.state_dir.mkdir(exist_ok=True)
        
        self.state_file = self.state_dir / "state.json"
        self.tensor_file = self.state_dir / "tensors.pt"
        self.history_file = self.state_dir / "history.json"
        
        self.state = self.load_state()
        self.history = self.load_history()
    
    def load_state(self) -> Dict:
        """Load cognitive state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            logger.info(f"📂 Loaded cognitive state ({state.get('version', 'unknown')})")
            return state
        else:
            return self.default_state()
    
    def default_state(self) -> Dict:
        """Create default cognitive state."""
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            
            # Personality (8 dimensions, each 0-1 scale)
            'personality': {
                'warmth': 0.7,
                'curiosity': 0.8,
                'humor': 0.6,
                'assertiveness': 0.5,
                'analytical': 0.7,
                'creative': 0.65,
                'loyal': 0.9,
                'adaptive': 0.75,
            },
            
            # Emotional state
            'emotion': {
                'mood': 0.5,        # -1 (negative) to 1 (positive)
                'curiosity': 0.7,   # 0 (bored) to 1 (very curious)
                'confidence': 0.6,  # 0 (uncertain) to 1 (very confident)
                'energy': 0.8,      # 0 (tired) to 1 (energized)
            },
            
            # Metacognition
            'metacognition': {
                'learning_efficiency': 0.5,
                'knowledge_gaps': 0.3,
                'confidence_calibration': 0.5,
                'preferred_strategy': 'consolidate',
                'total_learning_cycles': 0,
                'total_dream_cycles': 0,
            },
            
            # World model stats
            'world_model': {
                'n_memories_seen': 0,
                'n_insights_generated': 0,
                'n_connections_found': 0,
                'topic_distribution': {},
            },
            
            # Session stats
            'sessions': {
                'total': 0,
                'last_session': None,
                'avg_session_length_min': 0,
            },
            
            # Timestamps
            'timestamps': {
                'last_consolidation': None,
                'last_dream': None,
                'last_training': None,
                'last_emotion_update': None,
            },
        }
    
    def save_state(self):
        """Save current cognitive state to disk."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.info(f"💾 Cognitive state saved")
    
    def load_history(self) -> List[Dict]:
        """Load state change history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []
    
    def save_history(self):
        """Save state change history."""
        # Keep only last 1000 entries
        self.history = self.history[-1000:]
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_change(self, category: str, changes: Dict, reason: str = ""):
        """Record a state change in history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'changes': changes,
            'reason': reason,
        }
        self.history.append(entry)
    
    # === Personality Updates ===
    
    def get_personality(self) -> Dict[str, float]:
        """Get current personality state."""
        return self.state['personality'].copy()
    
    def update_personality(self, dimension: str, delta: float, reason: str = ""):
        """Update a personality dimension."""
        if dimension not in self.state['personality']:
            logger.warning(f"Unknown personality dimension: {dimension}")
            return
        
        old_val = self.state['personality'][dimension]
        new_val = max(0.0, min(1.0, old_val + delta))
        self.state['personality'][dimension] = new_val
        
        self.record_change('personality', {
            'dimension': dimension,
            'old': old_val,
            'new': new_val,
            'delta': delta,
        }, reason)
        
        logger.info(f"🎭 Personality [{dimension}]: {old_val:.3f} → {new_val:.3f}")
    
    # === Emotional State ===
    
    def get_emotion(self) -> Dict[str, float]:
        """Get current emotional state."""
        return self.state['emotion'].copy()
    
    def update_emotion(self, dimension: str, delta: float, reason: str = ""):
        """Update emotional state."""
        if dimension not in self.state['emotion']:
            return
        
        old_val = self.state['emotion'][dimension]
        new_val = max(-1.0, min(1.0, old_val + delta))
        self.state['emotion'][dimension] = new_val
        
        self.record_change('emotion', {
            'dimension': dimension,
            'old': old_val,
            'new': new_val,
        }, reason)
    
    def decay_emotions(self, decay_rate: float = 0.05):
        """Gradually decay emotions toward neutral."""
        for dim in ['mood', 'curiosity', 'energy']:
            val = self.state['emotion'][dim]
            self.state['emotion'][dim] = val * (1 - decay_rate)
    
    # === Metacognition ===
    
    def get_metacognition(self) -> Dict:
        """Get metacognition state."""
        return self.state['metacognition'].copy()
    
    def update_learning_efficiency(self, new_loss: float, previous_loss: float):
        """Update learning efficiency based on loss improvement."""
        if previous_loss > 0:
            improvement = (previous_loss - new_loss) / previous_loss
            efficiency = max(0, min(1, 0.5 + improvement))
            
            old = self.state['metacognition']['learning_efficiency']
            # Exponential moving average
            self.state['metacognition']['learning_efficiency'] = old * 0.8 + efficiency * 0.2
            
            self.record_change('metacognition', {
                'metric': 'learning_efficiency',
                'old': old,
                'new': efficiency,
                'loss_change': (previous_loss, new_loss),
            })
    
    def increment_cycle_count(self, cycle_type: str):
        """Increment cycle counter."""
        key = f'total_{cycle_type}_cycles'
        if key in self.state['metacognition']:
            self.state['metacognition'][key] += 1
    
    # === World Model ===
    
    def record_memory_seen(self):
        """Record that a memory was processed."""
        self.state['world_model']['n_memories_seen'] += 1
    
    def record_insight(self, insight: Dict):
        """Record a generated insight."""
        self.state['world_model']['n_insights_generated'] += 1
    
    def record_connection(self):
        """Record a found connection."""
        self.state['world_model']['n_connections_found'] += 1
    
    def update_topic_distribution(self, topics: List[str]):
        """Update topic distribution from new memories."""
        dist = self.state['world_model']['topic_distribution']
        for topic in topics:
            dist[topic] = dist.get(topic, 0) + 1
    
    # === Session Management ===
    
    def start_session(self):
        """Record session start."""
        self.state['sessions']['total'] += 1
        self.state['sessions']['last_session'] = datetime.now().isoformat()
        self.record_change('session', {'action': 'start'})
    
    def end_session(self, duration_min: float):
        """Record session end."""
        avg = self.state['sessions']['avg_session_length_min']
        n = self.state['sessions']['total']
        # Running average
        self.state['sessions']['avg_session_length_min'] = (avg * (n - 1) + duration_min) / n
    
    # === Summary & Reporting ===
    
    def get_summary(self) -> str:
        """Get human-readable cognitive state summary."""
        p = self.state['personality']
        e = self.state['emotion']
        m = self.state['metacognition']
        w = self.state['world_model']
        
        lines = [
            "🧠 Hacı Cognitive State",
            "=" * 40,
            "",
            "🎭 Personality:",
        ]
        for name, val in p.items():
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"   {name:15s} [{bar}] {val:.2f}")
        
        lines.append("")
        lines.append("💭 Emotion:")
        for name, val in e.items():
            emoji = "😊" if val > 0.3 else "😐" if val > -0.3 else "😔"
            lines.append(f"   {name:15s} {val:+.2f} {emoji}")
        
        lines.append("")
        lines.append("🔬 Metacognition:")
        lines.append(f"   Learning efficiency: {m['learning_efficiency']:.2f}")
        lines.append(f"   Knowledge gaps: {m['knowledge_gaps']:.2f}")
        lines.append(f"   Preferred strategy: {m['preferred_strategy']}")
        lines.append(f"   Learning cycles: {m['total_learning_cycles']}")
        lines.append(f"   Dream cycles: {m['total_dream_cycles']}")
        
        lines.append("")
        lines.append("🌍 World Model:")
        lines.append(f"   Memories seen: {w['n_memories_seen']}")
        lines.append(f"   Insights: {w['n_insights_generated']}")
        lines.append(f"   Connections: {w['n_connections_found']}")
        
        lines.append("")
        lines.append(f"📊 Sessions: {self.state['sessions']['total']}")
        
        return "\n".join(lines)
    
    def get_state_tensor(self) -> torch.Tensor:
        """Get full cognitive state as a single tensor."""
        parts = []
        
        # Personality (8 values)
        p = self.state['personality']
        parts.extend([p[k] for k in sorted(p.keys())])
        
        # Emotion (4 values)
        e = self.state['emotion']
        parts.extend([e[k] for k in sorted(e.keys())])
        
        # Metacognition (3 values)
        m = self.state['metacognition']
        parts.append(m['learning_efficiency'])
        parts.append(m['knowledge_gaps'])
        parts.append(m['confidence_calibration'])
        
        return torch.tensor(parts, dtype=torch.float32)  # [15]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    manager = CognitiveStateManager(workspace)
    
    print(manager.get_summary())
    
    # Demo updates
    print("\n--- Demo Updates ---")
    manager.update_personality('curiosity', 0.05, "Learned new topic")
    manager.update_emotion('mood', 0.1, "Good conversation")
    manager.record_memory_seen()
    manager.record_memory_seen()
    manager.record_insight({'type': 'test'})
    
    manager.save_state()
    
    print(manager.get_summary())

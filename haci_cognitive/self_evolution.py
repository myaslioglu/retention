"""
🧬 Self-Evolution Tracker - Level 3
Tracks, manages, and guides the system's evolution over time.

Components:
- EvolutionTracker: Tracks evolutionary milestones
- MetaFitnessEvaluator: Evaluates fitness across multiple dimensions
- EvolutionaryPressure: Applies selection pressure
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class EvolutionStage(Enum):
    INFANCY = "infancy"         # 0-10 cycles
    CHILDHOOD = "childhood"     # 10-50 cycles
    ADOLESCENCE = "adolescence" # 50-200 cycles
    ADULTHOOD = "adulthood"     # 200-1000 cycles
    MASTERY = "mastery"         # 1000+ cycles


@dataclass
class EvolutionaryMilestone:
    """A significant milestone in the system's evolution."""
    name: str
    description: str
    cycle_number: int
    metrics: Dict[str, float]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EvolutionTracker:
    """Tracks the system's evolutionary history."""
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "evolution_state.json"
        
        self.birth_time = datetime.now().isoformat()
        self.total_cycles = 0
        self.milestones: List[Dict] = []
        self.fitness_history: List[Dict] = []
        
        self.load_state()
        logger.info(f"🧬 EvolutionTracker initialized (stage: {self.get_stage().value})")
    
    def record_cycle(self, metrics: Dict[str, float]):
        """Record a cycle with metrics."""
        self.total_cycles += 1
        
        fitness = self._calculate_fitness(metrics)
        
        record = {
            'cycle': self.total_cycles,
            'fitness': fitness,
            'metrics': metrics,
            'stage': self.get_stage().value,
            'timestamp': datetime.now().isoformat(),
        }
        self.fitness_history.append(record)
        
        # Check for milestones
        self._check_milestones(metrics, fitness)
        
        self.save_state()
    
    def _calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculate overall fitness score."""
        weights = {
            'learning_depth': 0.2,
            'prediction_accuracy': 0.15,
            'curiosity': 0.15,
            'emotional_stability': 0.1,
            'memory_efficiency': 0.1,
            'creative_output': 0.15,
            'adaptability': 0.15,
        }
        
        fitness = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in metrics:
                fitness += metrics[metric] * weight
                total_weight += weight
        
        return fitness / max(total_weight, 0.01)
    
    def get_stage(self) -> EvolutionStage:
        """Get current evolution stage."""
        if self.total_cycles < 10:
            return EvolutionStage.INFANCY
        elif self.total_cycles < 50:
            return EvolutionStage.CHILDHOOD
        elif self.total_cycles < 200:
            return EvolutionStage.ADOLESCENCE
        elif self.total_cycles < 1000:
            return EvolutionStage.ADULTHOOD
        else:
            return EvolutionStage.MASTERY
    
    def _check_milestones(self, metrics: Dict[str, float], fitness: float):
        """Check if any milestones were reached."""
        milestone_checks = [
            ('first_cycle', self.total_cycles == 1, 'First learning cycle completed'),
            ('ten_cycles', self.total_cycles == 10, 'Survived infancy - 10 cycles'),
            ('high_fitness', fitness > 0.8, f'High fitness achieved: {fitness:.3f}'),
            ('fifty_cycles', self.total_cycles == 50, 'Childhood complete - 50 cycles'),
            ('hundred_cycles', self.total_cycles == 100, 'Century of learning'),
            ('curiosity_peak', metrics.get('curiosity', 0) > 0.9, 'Peak curiosity achieved'),
            ('creative_breakthrough', metrics.get('creative_output', 0) > 0.85, 'Creative breakthrough'),
        ]
        
        for name, condition, description in milestone_checks:
            if condition:
                milestone = {
                    'name': name,
                    'description': description,
                    'cycle': self.total_cycles,
                    'fitness': fitness,
                    'timestamp': datetime.now().isoformat(),
                }
                self.milestones.append(milestone)
                logger.info(f"🏆 MILESTONE: {description}")
    
    def get_growth_trajectory(self) -> Dict:
        """Analyze growth trajectory."""
        if len(self.fitness_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent = self.fitness_history[-10:]
        older = self.fitness_history[-20:-10] if len(self.fitness_history) >= 20 else self.fitness_history[:10]
        
        recent_avg = sum(r['fitness'] for r in recent) / len(recent)
        older_avg = sum(r['fitness'] for r in older) / len(older)
        
        growth_rate = recent_avg - older_avg
        
        if growth_rate > 0.05:
            trajectory = 'accelerating'
        elif growth_rate > 0:
            trajectory = 'growing'
        elif growth_rate > -0.05:
            trajectory = 'stable'
        else:
            trajectory = 'declining'
        
        return {
            'stage': self.get_stage().value,
            'total_cycles': self.total_cycles,
            'recent_fitness': recent_avg,
            'growth_rate': growth_rate,
            'trajectory': trajectory,
            'milestones': len(self.milestones),
        }
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'birth_time': self.birth_time,
            'total_cycles': self.total_cycles,
            'milestones': self.milestones,
            'fitness_history': self.fitness_history[-200:],
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Load state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.birth_time = state.get('birth_time', self.birth_time)
                self.total_cycles = state.get('total_cycles', 0)
                self.milestones = state.get('milestones', [])
                self.fitness_history = state.get('fitness_history', [])
                logger.info(f"📂 Loaded evolution state ({self.total_cycles} cycles)")
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")
    
    def get_summary(self) -> str:
        """Summary."""
        stage = self.get_stage()
        trajectory = self.get_growth_trajectory()
        
        lines = [
            f"🧬 Evolution Tracker",
            "=" * 40,
            f"Stage: {stage.value.upper()}",
            f"Total cycles: {self.total_cycles}",
            f"Birth: {self.birth_time[:10]}",
            f"Milestones: {len(self.milestones)}",
            f"Trajectory: {trajectory.get('trajectory', 'unknown')}",
        ]
        
        if self.milestones:
            lines.append("\nRecent milestones:")
            for m in self.milestones[-3:]:
                lines.append(f"  🏆 {m['description']}")
        
        return "\n".join(lines)


class MetaFitnessEvaluator:
    """Evaluates fitness across multiple cognitive dimensions."""
    
    DIMENSIONS = [
        'learning_efficiency',
        'memory_retention',
        'curiosity_drive',
        'creative_capacity',
        'predictive_accuracy',
        'emotional_intelligence',
        'adaptability',
        'metacognitive_awareness',
    ]
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.scores: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"📊 MetaFitnessEvaluator initialized ({len(self.DIMENSIONS)} dimensions)")
    
    def evaluate(self, system_state: Dict) -> Dict[str, float]:
        """Evaluate fitness across all dimensions."""
        scores = {}
        
        # Learning efficiency
        cycles = system_state.get('learning_cycles', 0)
        topics = system_state.get('topics_learned', 0)
        scores['learning_efficiency'] = min(1.0, topics / max(cycles, 1))
        
        # Memory retention
        memories = system_state.get('total_memories', 0)
        scores['memory_retention'] = min(1.0, memories / 100)
        
        # Curiosity drive
        interests = system_state.get('active_interests', 0)
        gaps = system_state.get('knowledge_gaps', 0)
        scores['curiosity_drive'] = min(1.0, (interests + gaps) / 50)
        
        # Creative capacity
        insights = system_state.get('insights_generated', 0)
        scores['creative_capacity'] = min(1.0, insights / 20)
        
        # Predictive accuracy
        surprise_events = system_state.get('surprise_events', 0)
        predictions = system_state.get('total_predictions', 1)
        scores['predictive_accuracy'] = max(0, 1.0 - surprise_events / max(predictions, 1))
        
        # Emotional intelligence
        emotional_state = system_state.get('emotional_state', {})
        stability = 1.0 - abs(emotional_state.get('mood', 0.5) - 0.5) * 2
        scores['emotional_intelligence'] = stability
        
        # Adaptability
        strategy_changes = system_state.get('strategy_changes', 0)
        scores['adaptability'] = min(1.0, strategy_changes / 10)
        
        # Metacognitive awareness
        meta_updates = system_state.get('meta_updates', 0)
        scores['metacognitive_awareness'] = min(1.0, meta_updates / 20)
        
        # Track
        for dim, score in scores.items():
            self.scores[dim].append(score)
        
        return scores
    
    def get_dimension_trends(self) -> Dict[str, str]:
        """Get trend for each dimension."""
        trends = {}
        for dim in self.DIMENSIONS:
            data = self.scores.get(dim, [])
            if len(data) < 2:
                trends[dim] = 'insufficient_data'
                continue
            
            recent = sum(data[-5:]) / min(5, len(data[-5:]))
            older = sum(data[-10:-5]) / max(1, len(data[-10:-5]))
            
            if recent > older + 0.05:
                trends[dim] = '↑'
            elif recent < older - 0.05:
                trends[dim] = '↓'
            else:
                trends[dim] = '→'
        
        return trends


class SelfEvolutionSystem:
    """
    Top-level self-evolution system.
    Combines tracking, fitness evaluation, and evolution guidance.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.tracker = EvolutionTracker(state_dir)
        self.fitness_evaluator = MetaFitnessEvaluator(state_dir)
        
        logger.info(f"🧬 SelfEvolutionSystem initialized")
    
    def record_evolution(self, system_state: Dict) -> Dict:
        """Record an evolution cycle."""
        # Evaluate fitness
        fitness_scores = self.fitness_evaluator.evaluate(system_state)
        
        # Record in tracker
        self.tracker.record_cycle(fitness_scores)
        
        # Get trajectory
        trajectory = self.tracker.get_growth_trajectory()
        
        return {
            'fitness_scores': fitness_scores,
            'trajectory': trajectory,
            'stage': self.tracker.get_stage().value,
            'dimension_trends': self.fitness_evaluator.get_dimension_trends(),
        }
    
    def get_full_report(self) -> str:
        """Full evolution report."""
        parts = [
            self.tracker.get_summary(),
            "",
            "📊 Dimension Trends:",
        ]
        
        trends = self.fitness_evaluator.get_dimension_trends()
        for dim, trend in trends.items():
            scores = self.fitness_evaluator.scores.get(dim, [])
            current = scores[-1] if scores else 0
            bar = "█" * int(current * 10) + "░" * (10 - int(current * 10))
            parts.append(f"  {trend} {dim:30s} [{bar}] {current:.3f}")
        
        return "\n".join(parts)

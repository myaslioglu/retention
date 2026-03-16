"""
🔮 Meta-Learning Architecture - Level 3
System that learns how to learn better.

Components:
- HyperparameterSelfOptimizer: Tunes own hyperparameters
- MetaStrategySelector: Meta-level strategy decisions
- ArchitectureEvolver: Self-modifies architecture
- MetaGradientTracker: Tracks meta-gradient flow
"""

import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Hyperparameter:
    """A self-optimizing hyperparameter."""
    name: str
    value: float
    min_val: float
    max_val: float
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    history: List[Tuple[float, float]] = field(default_factory=list)  # (value, performance)
    momentum: float = 0.0
    learning_rate: float = 0.1
    
    def update(self, performance: float) -> float:
        """Update based on performance. Returns new value."""
        self.history.append((self.value, performance))
        
        if len(self.history) < 2:
            # Explore randomly at first
            delta = random.gauss(0, (self.max_val - self.min_val) * 0.2)
        else:
            # Gradient-based update
            prev_val, prev_perf = self.history[-2]
            delta_perf = performance - prev_perf
            delta_val = self.value - prev_val
            
            if abs(delta_val) < 1e-8:
                delta = random.gauss(0, (self.max_val - self.min_val) * 0.1)
            else:
                # Estimate gradient
                gradient = delta_perf / delta_val
                if self.direction == OptimizationDirection.MINIMIZE:
                    gradient = -gradient
                
                # Momentum update
                self.momentum = 0.9 * self.momentum + 0.1 * gradient
                delta = self.learning_rate * self.momentum
        
        # Apply with bounds
        self.value = max(self.min_val, min(self.max_val, self.value + delta))
        return self.value
    
    def get_summary(self) -> Dict:
        return {
            'name': self.name,
            'value': round(self.value, 4),
            'range': [self.min_val, self.max_val],
            'updates': len(self.history),
            'best': max(self.history, key=lambda x: x[1])[0] if self.history else self.value,
        }


class HyperparameterSelfOptimizer:
    """
    Optimizes the cognitive system's own hyperparameters.
    Meta-learns which values work best over time.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "meta_hyperparams.json"
        
        # Self-optimizing hyperparameters
        self.params: Dict[str, Hyperparameter] = {}
        self._init_default_params()
        
        # Performance tracking
        self.global_performance_history: List[Dict] = []
        self.optimization_cycles = 0
        
        self.load_state()
        logger.info(f"🔧 HyperparameterSelfOptimizer initialized ({len(self.params)} params)")
    
    def _init_default_params(self):
        """Initialize default hyperparameters."""
        defaults = [
            # Learning parameters
            ("learning_rate", 0.1, 0.01, 0.5, OptimizationDirection.MAXIMIZE),
            ("exploration_rate", 0.3, 0.05, 0.8, OptimizationDirection.MAXIMIZE),
            ("memory_consolidation_threshold", 0.5, 0.1, 0.9, OptimizationDirection.MAXIMIZE),
            ("surprise_sensitivity", 0.5, 0.1, 1.0, OptimizationDirection.MAXIMIZE),
            
            # Curiosity parameters
            ("curiosity_decay", 0.95, 0.8, 0.99, OptimizationDirection.MAXIMIZE),
            ("novelty_weight", 0.3, 0.0, 1.0, OptimizationDirection.MAXIMIZE),
            ("knowledge_gap_threshold", 0.3, 0.1, 0.7, OptimizationDirection.MINIMIZE),
            
            # Emotional parameters
            ("mood_sensitivity", 0.1, 0.01, 0.3, OptimizationDirection.MAXIMIZE),
            ("energy_decay", 0.95, 0.9, 0.99, OptimizationDirection.MAXIMIZE),
            
            # Meta parameters
            ("meta_learning_rate", 0.05, 0.01, 0.2, OptimizationDirection.MAXIMIZE),
            ("architecture_mutation_rate", 0.01, 0.001, 0.1, OptimizationDirection.MAXIMIZE),
        ]
        
        for name, value, min_v, max_v, direction in defaults:
            if name not in self.params:
                self.params[name] = Hyperparameter(
                    name=name, value=value, min_val=min_v,
                    max_val=max_v, direction=direction
                )
    
    def optimize(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Run one optimization cycle. Returns new parameter values."""
        self.optimization_cycles += 1
        
        # Overall performance
        overall_perf = sum(performance_metrics.values()) / max(len(performance_metrics), 1)
        
        # Update each parameter
        new_values = {}
        for name, param in self.params.items():
            # Use specific metric if available, otherwise overall
            metric_key = self._get_relevant_metric(name)
            perf = performance_metrics.get(metric_key, overall_perf)
            
            new_val = param.update(perf)
            new_values[name] = new_val
        
        # Track
        self.global_performance_history.append({
            'cycle': self.optimization_cycles,
            'performance': overall_perf,
            'params': {k: v.value for k, v in self.params.items()},
            'timestamp': datetime.now().isoformat(),
        })
        
        self.save_state()
        logger.info(f"🔧 Optimization cycle {self.optimization_cycles}: perf={overall_perf:.3f}")
        
        return new_values
    
    def _get_relevant_metric(self, param_name: str) -> str:
        """Map parameter name to relevant performance metric."""
        mapping = {
            'learning_rate': 'learning_depth',
            'exploration_rate': 'discovery_rate',
            'surprise_sensitivity': 'prediction_accuracy',
            'curiosity_decay': 'engagement',
            'mood_sensitivity': 'emotional_stability',
            'energy_decay': 'sustained_performance',
        }
        return mapping.get(param_name, 'overall')
    
    def get_best_params(self) -> Dict[str, float]:
        """Get historically best parameter values."""
        best = {}
        for name, param in self.params.items():
            if param.history:
                best[name] = max(param.history, key=lambda x: x[1])[0]
            else:
                best[name] = param.value
        return best
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """Get current adaptive configuration for all modules."""
        return {
            'curiosity': {
                'decay': self.params['curiosity_decay'].value,
                'novelty_weight': self.params['novelty_weight'].value,
                'gap_threshold': self.params['knowledge_gap_threshold'].value,
            },
            'learning': {
                'rate': self.params['learning_rate'].value,
                'exploration': self.params['exploration_rate'].value,
                'memory_threshold': self.params['memory_consolidation_threshold'].value,
            },
            'prediction': {
                'surprise_sensitivity': self.params['surprise_sensitivity'].value,
            },
            'emotional': {
                'mood_sensitivity': self.params['mood_sensitivity'].value,
                'energy_decay': self.params['energy_decay'].value,
            },
            'meta': {
                'learning_rate': self.params['meta_learning_rate'].value,
                'mutation_rate': self.params['architecture_mutation_rate'].value,
            },
        }
    
    def save_state(self):
        """Save optimizer state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'params': {name: {
                'value': p.value,
                'history': p.history[-50:],  # Keep last 50
                'momentum': p.momentum,
            } for name, p in self.params.items()},
            'cycles': self.optimization_cycles,
            'performance_history': self.global_performance_history[-100:],
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Load optimizer state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                for name, pdata in state.get('params', {}).items():
                    if name in self.params:
                        self.params[name].value = pdata['value']
                        self.params[name].history = pdata.get('history', [])
                        self.params[name].momentum = pdata.get('momentum', 0.0)
                self.optimization_cycles = state.get('cycles', 0)
                self.global_performance_history = state.get('performance_history', [])
                logger.info(f"📂 Loaded meta-optimizer state ({self.optimization_cycles} cycles)")
            except Exception as e:
                logger.warning(f"Failed to load meta-optimizer state: {e}")
    
    def get_summary(self) -> str:
        """Human-readable summary."""
        lines = ["🔧 Hyperparameter Self-Optimizer", "=" * 40]
        lines.append(f"Optimization cycles: {self.optimization_cycles}")
        lines.append(f"\nParameters:")
        for name, p in self.params.items():
            bar_len = int(p.value * 10 / max(p.max_val, 0.001))
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  {name:35s} [{bar}] {p.value:.4f}")
        return "\n".join(lines)


class MetaStrategySelector:
    """
    Meta-level strategy selector.
    Decides WHICH learning strategy to use based on context and history.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "meta_strategy.json"
        
        # Strategy performance tracking
        self.strategy_scores: Dict[str, List[float]] = defaultdict(list)
        self.context_patterns: Dict[str, str] = {}  # context_signature -> best_strategy
        
        # Available strategies
        self.strategies = [
            'explore',       # Try new things
            'deepen',        # Go deeper on known topics
            'connect',       # Find connections between topics
            'consolidate',   # Review and solidify
            'create',        # Generate new insights
            'challenge',     # Question assumptions
        ]
        
        self.selection_history: List[Dict] = []
        self.load_state()
        logger.info(f"🧠 MetaStrategySelector initialized ({len(self.strategies)} strategies)")
    
    def select_strategy(self, context: Dict) -> str:
        """Select best strategy for current context."""
        context_sig = self._context_signature(context)
        
        # Check if we have a learned preference for this context
        if context_sig in self.context_patterns:
            base_strategy = self.context_patterns[context_sig]
        else:
            base_strategy = self._score_based_selection()
        
        # Add exploration
        if random.random() < 0.15:  # 15% exploration
            strategy = random.choice(self.strategies)
            logger.info(f"🧠 Strategy: {strategy} (exploration)")
        else:
            strategy = base_strategy
            logger.info(f"🧠 Strategy: {strategy} (exploitation)")
        
        self.selection_history.append({
            'strategy': strategy,
            'context': context_sig,
            'timestamp': datetime.now().isoformat(),
        })
        
        return strategy
    
    def update_strategy_performance(self, strategy: str, performance: float):
        """Record strategy performance."""
        self.strategy_scores[strategy].append(performance)
        # Keep bounded
        if len(self.strategy_scores[strategy]) > 100:
            self.strategy_scores[strategy] = self.strategy_scores[strategy][-100:]
    
    def _score_based_selection(self) -> str:
        """Select based on historical scores."""
        scores = {}
        for s in self.strategies:
            perf = self.strategy_scores.get(s, [])
            if perf:
                scores[s] = sum(perf) / len(perf)
            else:
                scores[s] = 0.5  # Default score for unexplored
        
        # Softmax selection
        exp_scores = {k: math.exp(v * 3) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}
        
        r = random.random()
        cumulative = 0
        for s, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return s
        return self.strategies[0]
    
    def _context_signature(self, context: Dict) -> str:
        """Create a signature string from context."""
        parts = []
        if 'time_of_day' in context:
            hour = context['time_of_day']
            if 6 <= hour < 12:
                parts.append('morning')
            elif 12 <= hour < 18:
                parts.append('afternoon')
            else:
                parts.append('evening')
        
        if 'emotional_state' in context:
            mood = context['emotional_state'].get('mood', 0.5)
            parts.append('positive' if mood > 0.6 else 'negative' if mood < 0.4 else 'neutral')
        
        if 'recent_activity' in context:
            parts.append(context['recent_activity'][:20])
        
        return "_".join(parts) if parts else "general"
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'strategy_scores': {k: v[-50:] for k, v in self.strategy_scores.items()},
            'context_patterns': self.context_patterns,
            'selection_history': self.selection_history[-100:],
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
                self.strategy_scores = defaultdict(list, state.get('strategy_scores', {}))
                self.context_patterns = state.get('context_patterns', {})
                self.selection_history = state.get('selection_history', [])
                logger.info(f"📂 Loaded meta-strategy state")
            except Exception as e:
                logger.warning(f"Failed to load meta-strategy state: {e}")


class ArchitectureEvolver:
    """
    Self-modifying architecture component.
    Suggests and tracks architecture mutations.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "architecture_evolution.json"
        
        self.architecture_version = 1
        self.mutation_history: List[Dict] = []
        
        # Current architecture DNA
        self.architecture_dna = {
            'world_model_dims': 512,
            'personality_dims': 1024,
            'retention_layers': 4,
            'attention_heads': 8,
            'emotional_dimensions': 4,
            'metacognition_metrics': 4,
        }
        
        # Mutation operators
        self.mutation_operators = [
            self._mutate_increase_capacity,
            self._mutate_decrease_capacity,
            self._mutate_add_specialization,
            self._mutate_adjust_granularity,
        ]
        
        self.load_state()
        logger.info(f"🧬 ArchitectureEvolver v{self.architecture_version} initialized")
    
    def propose_mutation(self, performance_history: List[Dict]) -> Optional[Dict]:
        """Propose an architecture mutation based on performance."""
        if len(performance_history) < 5:
            return None  # Need more data
        
        # Check if performance is plateauing
        recent = [p.get('performance', 0.5) for p in performance_history[-5:]]
        avg_recent = sum(recent) / len(recent)
        older = [p.get('performance', 0.5) for p in performance_history[-10:-5]]
        avg_older = sum(older) / max(len(older), 1)
        
        improvement = avg_recent - avg_older
        
        if improvement < 0.02:  # Plateau detected
            # Propose mutation
            mutation = random.choice(self.mutation_operators)()
            mutation['reason'] = f'plateau_detected (improvement={improvement:.4f})'
            mutation['triggered_at'] = datetime.now().isoformat()
            
            self.mutation_history.append(mutation)
            self.architecture_version += 1
            
            self.save_state()
            logger.info(f"🧬 Proposed mutation v{self.architecture_version}: {mutation['type']}")
            
            return mutation
        
        return None
    
    def _mutate_increase_capacity(self) -> Dict:
        """Increase capacity of a random component."""
        component = random.choice(['world_model_dims', 'personality_dims', 'attention_heads'])
        factor = random.uniform(1.1, 1.5)
        old_val = self.architecture_dna[component]
        new_val = min(int(old_val * factor), 2048)
        self.architecture_dna[component] = new_val
        
        return {
            'type': 'increase_capacity',
            'component': component,
            'old_value': old_val,
            'new_value': new_val,
            'factor': round(factor, 2),
        }
    
    def _mutate_decrease_capacity(self) -> Dict:
        """Decrease capacity for efficiency."""
        component = random.choice(['world_model_dims', 'personality_dims'])
        factor = random.uniform(0.8, 0.95)
        old_val = self.architecture_dna[component]
        new_val = max(int(old_val * factor), 64)
        self.architecture_dna[component] = new_val
        
        return {
            'type': 'decrease_capacity',
            'component': component,
            'old_value': old_val,
            'new_value': new_val,
            'factor': round(factor, 2),
        }
    
    def _mutate_add_specialization(self) -> Dict:
        """Add specialized sub-component."""
        specializations = ['creative_module', 'logical_module', 'social_module', 'spatial_module']
        chosen = random.choice(specializations)
        
        return {
            'type': 'add_specialization',
            'component': chosen,
            'description': f'Added {chosen} for specialized processing',
        }
    
    def _mutate_adjust_granularity(self) -> Dict:
        """Adjust processing granularity."""
        component = random.choice(['retention_layers', 'emotional_dimensions'])
        delta = random.choice([-1, 1, 2])
        old_val = self.architecture_dna[component]
        new_val = max(2, min(8, old_val + delta))
        self.architecture_dna[component] = new_val
        
        return {
            'type': 'adjust_granularity',
            'component': component,
            'old_value': old_val,
            'new_value': new_val,
        }
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'version': self.architecture_version,
            'dna': self.architecture_dna,
            'mutation_history': self.mutation_history[-50:],
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
                self.architecture_version = state.get('version', 1)
                self.architecture_dna = state.get('dna', self.architecture_dna)
                self.mutation_history = state.get('mutation_history', [])
                logger.info(f"📂 Loaded architecture evolution state (v{self.architecture_version})")
            except Exception as e:
                logger.warning(f"Failed to load architecture state: {e}")
    
    def get_summary(self) -> str:
        """Summary of current architecture."""
        lines = [f"🧬 Architecture DNA (v{self.architecture_version})", "=" * 40]
        for k, v in self.architecture_dna.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"\nMutations: {len(self.mutation_history)}")
        if self.mutation_history:
            last = self.mutation_history[-1]
            lines.append(f"Last: {last['type']} ({last.get('component', 'N/A')})")
        return "\n".join(lines)

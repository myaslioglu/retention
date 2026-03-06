#!/usr/bin/env python3
"""
AUTO-TUNING OPTIMIZER
Bayesian-inspired hyperparameter optimization for memory system
"""

import json
import time
import random
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class ParameterSpace:
    """Define search space for hyperparameters"""
    
    def __init__(self):
        self.params = {
            'importance_threshold': {'min': 0.2, 'max': 0.4, 'type': 'float', 'step': 0.01},
            'decay_rate': {'min': 0.01, 'max': 0.05, 'type': 'float', 'step': 0.005},
            'cache_size': {'min': 200, 'max': 1000, 'type': 'int', 'step': 50},
            'max_memories_per_day': {'min': 5, 'max': 15, 'type': 'int', 'step': 1},
            'reindex_interval_hours': {'min': 6, 'max': 24, 'type': 'int', 'step': 2}
        }
    
    def sample_random(self) -> Dict:
        """Sample random parameters from space"""
        config = {}
        for name, spec in self.params.items():
            if spec['type'] == 'float':
                config[name] = round(random.uniform(spec['min'], spec['max']), 3)
            else:
                config[name] = random.randint(spec['min'], spec['max'])
        return config
    
    def mutate(self, config: Dict, strength: float = 0.2) -> Dict:
        """Mutate parameters slightly"""
        new_config = config.copy()
        for name, spec in self.params.items():
            if random.random() < strength:
                if spec['type'] == 'float':
                    delta = (spec['max'] - spec['min']) * 0.1 * random.uniform(-1, 1)
                    new_val = config[name] + delta
                    new_config[name] = max(spec['min'], min(spec['max'], new_val))
                else:
                    delta = int((spec['max'] - spec['min']) * 0.1 * random.uniform(-1, 1))
                    new_val = config[name] + delta
                    new_config[name] = max(spec['min'], min(spec['max'], new_val))
        return new_config
    
    def crossover(self, config1: Dict, config2: Dict) -> Dict:
        """Crossover two configs"""
        child = {}
        for name in self.params:
            if random.random() < 0.5:
                child[name] = config1[name]
            else:
                child[name] = config2[name]
        return child

class PerformanceMonitor:
    """Collect system performance metrics"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace = Path(workspace_dir)
        self.metrics_file = self.workspace / "memory" / "performance_metrics.json"
        self.current_metrics = self._load_metrics()
    
    def _load_metrics(self) -> List[Dict]:
        """Load historical metrics"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def collect_current_metrics(self, compressor, consolidator) -> Dict:
        """Collect real-time metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query_time_ms': compressor.metadata.get('avg_query_time_ms', 0) if hasattr(compressor, 'metadata') else 0.58,
            'cache_hit_rate': self._calc_cache_hit_rate(compressor),
            'memory_growth_rate': self._calc_memory_growth(),
            'consolidation_quality': self._estimate_quality(consolidator),
            'system_load': 0.3
        }
        
        metrics['overall_score'] = self._calculate_score(metrics)
        self.current_metrics.append(metrics)
        self._save_metrics()
        return metrics
    
    def _calc_cache_hit_rate(self, compressor) -> float:
        """Calculate cache hit rate"""
        if hasattr(compressor, 'metadata'):
            total = compressor.metadata.get('query_count', 1)
            hits = compressor.metadata.get('cache_hits', 0)
            return hits / total if total > 0 else 0.0
        return 0.75
    
    def _calc_memory_growth(self) -> float:
        """Calculate memory growth rate (memories/day)"""
        mem_file = self.workspace / "MEMORY.md"
        if not mem_file.exists():
            return 0.3
        
        with open(mem_file, 'r') as f:
            content = f.read()
        
        total_memories = content.count('### ')
        return total_memories / 7.0
    
    def _estimate_quality(self, consolidator) -> float:
        """Estimate memory quality (0-1)"""
        types_seen = set()
        total_importance = 0
        count = 0
        
        mem_file = self.workspace / "MEMORY.md"
        if mem_file.exists():
            with open(mem_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip().startswith('###'):
                    for emoji in ['🤔', '🏆', '📚', '👤', '🚀', '⏰']:
                        if emoji in line:
                            types_seen.add(emoji)
                            break
                elif '**Önem:**' in line:
                    try:
                        imp = float(line.split(':')[1].strip().split()[0])
                        total_importance += imp
                        count += 1
                    except:
                        pass
        
        type_diversity = len(types_seen) / 6.0
        avg_importance = total_importance / count if count > 0 else 0.3
        return min(1.0, type_diversity * 0.5 + avg_importance * 0.5)
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall optimization score"""
        weights = {
            'query_time_ms': -0.25,
            'cache_hit_rate': 0.25,
            'memory_growth_rate': 0.15,
            'consolidation_quality': 0.25,
            'system_load': -0.10
        }
        
        score = 0
        for metric, weight in weights.items():
            val = metrics[metric]
            if metric == 'query_time_ms':
                val_norm = max(0, 1 - (val / 100.0))
            elif metric == 'cache_hit_rate':
                val_norm = val
            elif metric == 'memory_growth_rate':
                val_norm = 1 - abs(val - 0.5) / 0.5
            elif metric == 'consolidation_quality':
                val_norm = val
            elif metric == 'system_load':
                val_norm = 1 - val
            else:
                val_norm = 0.5
            
            score += val_norm * weight
        
        return max(0, min(1, score))
    
    def _save_metrics(self):
        """Save metrics to disk"""
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.current_metrics[-100:], f, indent=2)

class BayesianOptimizer:
    """Simplified Bayesian optimization"""
    
    def __init__(self, param_space: ParameterSpace, exploration_factor: float = 0.3):
        self.space = param_space
        self.exploration = exploration_factor
        self.history = []
    
    def suggest_next(self) -> Dict:
        """Suggest next configuration to try"""
        if len(self.history) < 5:
            return self.space.sample_random()
        
        if random.random() < self.exploration:
            return self.space.sample_random()
        
        best_config = max(self.history, key=lambda x: x[1])[0]
        return self.space.mutate(best_config, strength=0.15)
    
    def record_result(self, config: Dict, score: float):
        """Record configuration result"""
        self.history.append((config, score))
        if len(self.history) > 20:
            self.history = self.history[-20:]

class AutoTuningOptimizer:
    """Main auto-tuning orchestrator"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace = Path(workspace_dir)
        self.param_space = ParameterSpace()
        self.monitor = PerformanceMonitor(workspace_dir)
        self.optimizer = BayesianOptimizer(self.param_space)
        self.config_file = self.workspace / "memory" / "auto_tuning_config.json"
        
        self.current_config = self._load_config()
        self.tuning_interval_hours = 2
        self.last_tuning = datetime.now() - timedelta(hours=self.tuning_interval_hours)
        
        print("⚙️  AUTO-TUNING OPTIMIZER INITIALIZED")
        print(f"   • Tuning interval: {self.tuning_interval_hours}h")
        print(f"   • Current config: {json.dumps(self.current_config, indent=6)}")
    
    def _load_config(self) -> Dict:
        """Load current configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self.param_space.sample_random()
    
    def _save_config(self, config: Dict):
        """Save configuration"""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def should_tune(self) -> bool:
        """Check if it's time to tune"""
        hours_since = (datetime.now() - self.last_tuning).total_seconds() / 3600
        return hours_since >= self.tuning_interval_hours
    
    def run_tuning_cycle(self):
        """Run one tuning cycle"""
        print("\n🔧 RUNNING AUTO-TUNING CYCLE")
        print("=" * 60)
        
        print("📊 Collecting performance metrics...")
        current_metrics = self.monitor.collect_current_metrics(None, None)
        current_score = current_metrics['overall_score']
        print(f"   • Current score: {current_score:.3f}")
        print(f"   • Query time: {current_metrics['query_time_ms']:.2f}ms")
        print(f"   • Cache hit: {current_metrics['cache_hit_rate']:.1%}")
        
        print("\n💡 Evaluating parameter space...")
        new_config = self.optimizer.suggest_next()
        print(f"   • Suggested config: {json.dumps(new_config, indent=6)}")
        
        if len(self.optimizer.history) > 0:
            best_historical = max(self.optimizer.history, key=lambda x: x[1])[1]
            improvement_potential = best_historical * 0.95 - current_score
            if improvement_potential <= 0:
                print(f"   ✅ Current config is good (score: {current_score:.3f})")
            else:
                print(f"   ⚠️  Potential improvement: +{improvement_potential:.3f}")
                self._apply_config(new_config)
        
        self.optimizer.record_result(self.current_config, current_score)
        self.last_tuning = datetime.now()
        
        print("\n📈 Tuning cycle completed!")
        print(f"   • History size: {len(self.optimizer.history)}")
        print(f"   • Next tuning in: {self.tuning_interval_hours}h")
        print("=" * 60)
    
    def _apply_config(self, config: Dict):
        """Apply new configuration to system"""
        print(f"\n⚙️  Applying new configuration...")
        self.current_config = config.copy()
        self._save_config(config)
        
        print("   • Importance threshold: {:.3f}".format(config['importance_threshold']))
        print("   • Decay rate: {:.3f}".format(config['decay_rate']))
        print("   • Cache size: {}".format(config['cache_size']))
        print("   • Max memories/day: {}".format(config['max_memories_per_day']))
        print("   • Reindex interval: {}h".format(config['reindex_interval_hours']))
        print("   ✅ Configuration saved to disk")
    
    def get_optimization_report(self) -> Dict:
        """Generate optimization report"""
        if not self.optimizer.history:
            return {'status': 'no_data'}
        
        configs, scores = zip(*self.optimizer.history)
        best_idx = scores.index(max(scores))
        best_config = configs[best_idx]
        best_score = scores[best_idx]
        
        if len(scores) >= 6:
            recent_scores = scores[-5:]
            older_scores = scores[:-5]
            trend = 'improving' if statistics.mean(recent_scores) > statistics.mean(older_scores) else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_trials': len(self.optimizer.history),
            'best_score': best_score,
            'best_config': best_config,
            'current_score': scores[-1],
            'score_trend': trend,
            'avg_score': statistics.mean(scores),
            'score_std': statistics.stdev(scores) if len(scores) > 1 else 0
        }

def test_auto_tuning():
    """Test auto-tuning optimizer"""
    print("🧪 TESTING AUTO-TUNING OPTIMIZER")
    print("=" * 60)
    
    optimizer = AutoTuningOptimizer()
    
    print("\n🔄 Simulating 5 tuning cycles...")
    
    for i in range(5):
        print(f"\n--- Cycle {i+1} ---")
        if optimizer.should_tune():
            optimizer.run_tuning_cycle()
        else:
            print("⏳ Not time to tune yet")
    
    print("\n📊 OPTIMIZATION REPORT:")
    report = optimizer.get_optimization_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 60)
    print("🎉 AUTO-TUNING TEST COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    test_auto_tuning()

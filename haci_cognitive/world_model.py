"""
🎨 Generative World Model - Level 3
Simulates, imagines, predicts, and creates.

Components:
- ImaginationEngine: Generate simulated scenarios
- CounterfactualReasoner: "What if X?" reasoning
- CreativeInsightGenerator: Novel connections and insights
- FuturePredictor: Anticipate future states
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


class ScenarioType(Enum):
    COUNTERFACTUAL = "counterfactual"
    PREDICTION = "prediction"
    CREATIVE = "creative"
    EXPLORATION = "exploration"


@dataclass
class Scenario:
    """A simulated scenario in the world model."""
    id: str
    type: ScenarioType
    premise: str
    assumptions: List[str]
    generated_outcomes: List[str]
    probability: float
    confidence: float
    novelty: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ImaginationEngine:
    """
    Generates simulated scenarios by combining existing knowledge
    in novel ways. The 'daydream' system.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "imagination_state.json"
        
        self.scenario_count = 0
        self.generated_scenarios: List[Dict] = []
        self.concept_graph: Dict[str, List[str]] = defaultdict(list)
        
        self.load_state()
        logger.info(f"💭 ImaginationEngine initialized ({len(self.concept_graph)} concepts)")
    
    def imagine(self, seed_concepts: List[str], scenario_type: ScenarioType = ScenarioType.CREATIVE) -> Scenario:
        """Generate a new scenario from seed concepts."""
        self.scenario_count += 1
        
        # Expand concepts through graph
        expanded = self._expand_concepts(seed_concepts)
        
        # Generate premise
        premise = self._generate_premise(seed_concepts, expanded, scenario_type)
        
        # Generate assumptions
        assumptions = self._generate_assumptions(seed_concepts, expanded)
        
        # Generate outcomes
        outcomes = self._generate_outcomes(premise, assumptions, scenario_type)
        
        # Calculate metrics
        probability = self._estimate_probability(outcomes)
        confidence = min(0.9, 0.3 + len(expanded) * 0.05)
        novelty = self._calculate_novelty(seed_concepts, expanded)
        
        scenario = Scenario(
            id=f"scenario_{self.scenario_count:04d}",
            type=scenario_type,
            premise=premise,
            assumptions=assumptions,
            generated_outcomes=outcomes,
            probability=probability,
            confidence=confidence,
            novelty=novelty,
        )
        
        self.generated_scenarios.append({
            'id': scenario.id,
            'type': scenario_type.value,
            'seed_concepts': seed_concepts,
            'premise': premise,
            'outcomes_count': len(outcomes),
            'probability': probability,
            'novelty': novelty,
            'timestamp': scenario.timestamp,
        })
        
        self.save_state()
        logger.info(f"💭 Generated {scenario.id}: {premise[:50]}... (novelty={novelty:.2f})")
        
        return scenario
    
    def _expand_concepts(self, seeds: List[str], depth: int = 2) -> List[str]:
        """Expand concepts through the concept graph."""
        expanded = set(seeds)
        current = set(seeds)
        
        for _ in range(depth):
            next_level = set()
            for concept in current:
                neighbors = self.concept_graph.get(concept, [])
                next_level.update(neighbors[:3])
            expanded.update(next_level)
            current = next_level
        
        return list(expanded)
    
    def _generate_premise(self, seeds: List[str], expanded: List[str], scenario_type: ScenarioType) -> str:
        """Generate a scenario premise."""
        templates = {
            ScenarioType.COUNTERFACTUAL: [
                f"Eğer {' ve '.join(seeds[:2])} farklı olsaydı, ne olurdu?",
                f"Diyelim ki {' ile '.join(seeds[:2])} hiç var olmamış?",
                f"{' '.join(seeds[:2])} yerine tam zıttı olsaydı?",
            ],
            ScenarioType.PREDICTION: [
                f"{' '.join(seeds[:2])} gelecekte nasıl gelişecek?",
                f"Bir sonraki adım {' '.join(seeds[:2])} için ne?",
                f"{' '.join(seeds[:2])} trendi nereye gidiyor?",
            ],
            ScenarioType.CREATIVE: [
                f"{' ve '.join(seeds[:2])} birleşse ne yaratır?",
                f"{' '.join(seeds[:2])} tamamen yeni bir şekilde düşünülse?",
                f"{' '.join(seeds[:2])} evrimleşse neye dönüşür?",
            ],
            ScenarioType.EXPLORATION: [
                f"{' '.join(seeds[:2])} hakkında ne bilmiyoruz?",
                f"{' '.join(seeds[:2])} farklı açılardan nasıl görünür?",
                f"{' '.join(seeds[:2])} keşfedilmemiş yönleri neler?",
            ],
        }
        
        return random.choice(templates.get(scenario_type, templates[ScenarioType.CREATIVE]))
    
    def _generate_assumptions(self, seeds: List[str], expanded: List[str]) -> List[str]:
        """Generate scenario assumptions."""
        assumptions = []
        
        for concept in seeds[:3]:
            assumptions.append(f"{concept} mevcut durumda kararlı")
        
        if expanded:
            assumptions.append(f"{random.choice(expanded[:3])} ile etkileşim var")
        
        assumptions.append("Dış faktörler sabit")
        
        return assumptions
    
    def _generate_outcomes(self, premise: str, assumptions: List[str], scenario_type: ScenarioType) -> List[str]:
        """Generate possible outcomes."""
        outcomes = []
        
        # Positive outcome
        outcomes.append(f"Olumlu: {premise[:30]}... beklenenden iyi sonuç verir")
        
        # Negative outcome
        outcomes.append(f"Olumsuz: {premise[:30]}... beklenmedik zorluklar yaratır")
        
        # Surprise outcome
        outcomes.append(f"Sürpriz: {premise[:30]}... tamamen farklı bir yöne evrilir")
        
        # For counterfactuals, add a "baseline" outcome
        if scenario_type == ScenarioType.COUNTERFACTUAL:
            outcomes.append(f"Paralel: Aslında pek bir şey değişmez")
        
        return outcomes
    
    def _estimate_probability(self, outcomes: List[str]) -> float:
        """Estimate probability distribution."""
        # Simple uniform-ish with slight bias toward "middle" outcomes
        return random.uniform(0.2, 0.8)
    
    def _calculate_novelty(self, seeds: List[str], expanded: List[str]) -> float:
        """Calculate how novel this scenario is."""
        # More connections = less novel
        connections = sum(len(self.concept_graph.get(s, [])) for s in seeds)
        base_novelty = max(0.1, 1.0 - connections * 0.05)
        
        # Add some randomness
        return min(1.0, base_novelty + random.uniform(-0.1, 0.2))
    
    def add_concept_connection(self, concept1: str, concept2: str):
        """Add a connection between concepts."""
        if concept2 not in self.concept_graph[concept1]:
            self.concept_graph[concept1].append(concept2)
        if concept1 not in self.concept_graph[concept2]:
            self.concept_graph[concept2].append(concept1)
    
    def learn_concepts_from_memories(self, memories: List[Dict]):
        """Extract and connect concepts from memories."""
        for memory in memories:
            content = memory.get('content', '') or memory.get('text', '')
            if not content:
                continue
            
            # Simple keyword extraction
            words = content.lower().split()
            keywords = [w for w in words if len(w) > 3]
            
            # Connect keywords from same memory
            for i, w1 in enumerate(keywords[:10]):
                for w2 in keywords[i+1:i+3]:
                    self.add_concept_connection(w1, w2)
        
        self.save_state()
        logger.info(f"💭 Learned concepts from {len(memories)} memories ({len(self.concept_graph)} total)")
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'scenario_count': self.scenario_count,
            'concept_graph': {k: v[:20] for k, v in list(self.concept_graph.items())[:200]},
            'recent_scenarios': self.generated_scenarios[-50:],
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
                self.scenario_count = state.get('scenario_count', 0)
                self.concept_graph = defaultdict(list, state.get('concept_graph', {}))
                self.generated_scenarios = state.get('recent_scenarios', [])
                logger.info(f"📂 Loaded imagination state ({self.scenario_count} scenarios)")
            except Exception as e:
                logger.warning(f"Failed to load imagination state: {e}")


class CounterfactualReasoner:
    """
    "What if?" reasoning engine.
    Explores alternative realities and their consequences.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "counterfactual_state.json"
        
        self.counterfactual_history: List[Dict] = []
        self.assumption_patterns: Dict[str, List[str]] = defaultdict(list)
        
        self.load_state()
        logger.info(f"🔄 CounterfactualReasoner initialized")
    
    def reason(self, event: str, alternative: str, context: Dict = None) -> Dict:
        """Perform counterfactual reasoning."""
        # Generate chain of consequences
        immediate = self._immediate_consequences(event, alternative)
        secondary = self._secondary_consequences(immediate, context)
        long_term = self._long_term_consequences(secondary)
        
        # Calculate divergence from reality
        divergence = self._calculate_divergence(immediate, secondary)
        
        result = {
            'event': event,
            'alternative': alternative,
            'immediate_consequences': immediate,
            'secondary_consequences': secondary,
            'long_term_effects': long_term,
            'divergence_score': divergence,
            'insight': self._extract_insight(event, alternative, immediate, long_term),
            'timestamp': datetime.now().isoformat(),
        }
        
        self.counterfactual_history.append(result)
        self.save_state()
        
        logger.info(f"🔄 Counterfactual: '{event}' → '{alternative}' (divergence={divergence:.2f})")
        
        return result
    
    def _immediate_consequences(self, event: str, alternative: str) -> List[str]:
        """First-order consequences."""
        return [
            f"{event} yerine {alternative} gerçekleşseydi, ilk etki: doğrudan sonuç değişimi",
            f"{alternative} ile başlayan zincir: mevcut durumda fark yaratır",
            f"Bu değişiklik, {event} ile ilişkili tüm süreçleri etkiler",
        ]
    
    def _secondary_consequences(self, immediate: List[str], context: Dict = None) -> List[str]:
        """Second-order consequences (ripple effects)."""
        return [
            f"İlk değişimden sonra: ikincil etkiler ortaya çıkar",
            f"Bağlantılı sistemler adapte olmaya başlar",
            f"Beklenmedik yan etkiler: {random.choice(['pozitif', 'negatif', 'nötr'])}",
        ]
    
    def _long_term_consequences(self, secondary: List[str]) -> List[str]:
        """Long-term effects."""
        return [
            f"Uzun vadede: sistem yeni bir denge noktasına ulaşır",
            f"Başlangıç farkı {'büyüyerek' if random.random() > 0.5 else 'küçülerek'} devam eder",
            f"Son durum: orijinal durumdan {'tamamen farklı' if random.random() > 0.6 else 'hafifçe farklı'}",
        ]
    
    def _calculate_divergence(self, immediate: List[str], secondary: List[str]) -> float:
        """How much this alternative diverges from reality."""
        # More consequences = more divergence
        base = 0.3 + len(immediate) * 0.1 + len(secondary) * 0.05
        return min(1.0, base + random.uniform(-0.1, 0.2))
    
    def _extract_insight(self, event: str, alternative: str, immediate: List[str], long_term: List[str]) -> str:
        """Extract key insight from counterfactual reasoning."""
        insights = [
            f"'{event}' aslında sandığımızdan daha önemli bir noktaymış",
            f"Küçük değişiklikler ({alternative}) büyük sonuçlar doğurabilir",
            f"Sistem daha dirençli/dirençsiz görünüyor - keşfedilmeli",
            f"Bu counterfactual, mevcut stratejinin doğru olduğunu gösteriyor",
        ]
        return random.choice(insights)
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'counterfactual_history': self.counterfactual_history[-50:],
            'assumption_patterns': {k: v[:20] for k, v in self.assumption_patterns.items()},
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
                self.counterfactual_history = state.get('counterfactual_history', [])
                self.assumption_patterns = defaultdict(list, state.get('assumption_patterns', {}))
                logger.info(f"📂 Loaded counterfactual state ({len(self.counterfactual_history)} reasonings)")
            except Exception as e:
                logger.warning(f"Failed to load counterfactual state: {e}")


class CreativeInsightGenerator:
    """
    Generates novel insights by finding unexpected connections
    between disparate concepts.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "creative_insights.json"
        
        self.insights: List[Dict] = []
        self.connection_weights: Dict[Tuple[str, str], float] = {}
        
        self.load_state()
        logger.info(f"💡 CreativeInsightGenerator initialized ({len(self.insights)} insights)")
    
    def generate_insight(self, concepts: List[str], memories: List[Dict] = None) -> Dict:
        """Generate a creative insight from concepts."""
        # Find unexpected connections
        connections = self._find_unexpected_connections(concepts)
        
        # Combine into insight
        insight_text = self._synthesize_insight(concepts, connections)
        
        # Rate novelty and usefulness
        novelty = self._rate_novelty(insight_text, connections)
        usefulness = self._rate_usefulness(insight_text, memories)
        
        insight = {
            'id': f"insight_{len(self.insights):04d}",
            'text': insight_text,
            'source_concepts': concepts,
            'connections': connections,
            'novelty': novelty,
            'usefulness': usefulness,
            'score': (novelty * 0.4 + usefulness * 0.6),
            'timestamp': datetime.now().isoformat(),
        }
        
        self.insights.append(insight)
        self.save_state()
        
        logger.info(f"💡 Insight generated: {insight_text[:60]}... (novelty={novelty:.2f})")
        
        return insight
    
    def _find_unexpected_connections(self, concepts: List[str]) -> List[Dict]:
        """Find non-obvious connections between concepts."""
        connections = []
        
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                # Check if connection exists
                key = (c1, c2) if c1 < c2 else (c2, c1)
                existing_weight = self.connection_weights.get(key, 0)
                
                # Unexpected = low existing weight
                if existing_weight < 0.3:
                    connections.append({
                        'concept_a': c1,
                        'concept_b': c2,
                        'connection_type': random.choice([
                            'analogy', 'contrast', 'complement', 
                            'causal', 'temporal', 'structural',
                        ]),
                        'strength': random.uniform(0.3, 0.8),
                        'unexpectedness': 1.0 - existing_weight,
                    })
                
                # Update weight
                self.connection_weights[key] = existing_weight + 0.1
        
        return connections
    
    def _synthesize_insight(self, concepts: List[str], connections: List[Dict]) -> str:
        """Synthesize connections into an insight."""
        if not connections:
            return f"{' ve '.join(concepts[:2])} arasında henüz bir bağlantı yok, bu da ilginç"
        
        conn = connections[0]
        templates = {
            'analogy': f"{conn['concept_a']} aslında {conn['concept_b']}'a benziyor - aynı yapısal özelliklere sahip",
            'contrast': f"{conn['concept_a']} ve {conn['concept_b']} zıt yönlerde çalışıyor - dengeleyici güçler",
            'complement': f"{conn['concept_a']}'nın eksik yanı {conn['concept_b']} ile tamamlanıyor",
            'causal': f"{conn['concept_a']} muhtemelen {conn['concept_b']}'nin nedeni olabilir",
            'temporal': f"{conn['concept_a']} genellikle {conn['concept_b']}'den önce geliyor",
            'structural': f"{conn['concept_a']} ve {conn['concept_b']} aynı yapısal pattern'i paylaşıyor",
        }
        
        return templates.get(conn['connection_type'], f"{conn['concept_a']} ve {conn['concept_b']} bağlantılı")
    
    def _rate_novelty(self, insight: str, connections: List[Dict]) -> float:
        """Rate how novel this insight is."""
        if not connections:
            return 0.3
        avg_unexpected = sum(c.get('unexpectedness', 0.5) for c in connections) / len(connections)
        return min(1.0, avg_unexpected + random.uniform(-0.1, 0.1))
    
    def _rate_usefulness(self, insight: str, memories: List[Dict] = None) -> float:
        """Rate how useful this insight is."""
        base = 0.5
        if memories:
            # More relevant memories = more useful
            relevant = sum(1 for m in memories if any(
                w in (m.get('content', '') or m.get('text', '')).lower()
                for w in insight.lower().split()[:5]
            ))
            base += min(0.3, relevant * 0.05)
        return min(1.0, base + random.uniform(-0.1, 0.1))
    
    def get_best_insights(self, n: int = 5) -> List[Dict]:
        """Get top insights by score."""
        sorted_insights = sorted(self.insights, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_insights[:n]
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'insights': self.insights[-100:],
            'connection_weights': {f"{k[0]}|{k[1]}": v for k, v in self.connection_weights.items()},
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
                self.insights = state.get('insights', [])
                cw = state.get('connection_weights', {})
                self.connection_weights = {tuple(k.split('|')): v for k, v in cw.items()}
                logger.info(f"📂 Loaded creative insights ({len(self.insights)} insights)")
            except Exception as e:
                logger.warning(f"Failed to load creative insights: {e}")


class FuturePredictor:
    """
    Predicts future states based on current trends and patterns.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "future_predictions.json"
        
        self.predictions: List[Dict] = []
        self.trend_data: Dict[str, List[float]] = defaultdict(list)
        
        self.load_state()
        logger.info(f"🔮 FuturePredictor initialized")
    
    def update_trend(self, metric_name: str, value: float):
        """Update trend data for a metric."""
        self.trend_data[metric_name].append(value)
        if len(self.trend_data[metric_name]) > 100:
            self.trend_data[metric_name] = self.trend_data[metric_name][-100:]
    
    def predict(self, metric_name: str, steps_ahead: int = 3) -> Dict:
        """Predict future values for a metric."""
        data = self.trend_data.get(metric_name, [])
        
        if len(data) < 3:
            return {
                'metric': metric_name,
                'prediction': 'insufficient_data',
                'message': f"Need at least 3 data points, have {len(data)}",
            }
        
        # Simple linear extrapolation
        recent = data[-5:]
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
        else:
            trend = 0
        
        current = data[-1]
        predicted_values = []
        for i in range(1, steps_ahead + 1):
            predicted = current + trend * i
            predicted += random.gauss(0, abs(trend) * 0.5)  # Add uncertainty
            predicted_values.append(predicted)
        
        # Confidence decreases with steps ahead
        confidence = max(0.1, 0.8 - steps_ahead * 0.15)
        
        # Direction
        if trend > 0.01:
            direction = 'increasing'
        elif trend < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        prediction = {
            'metric': metric_name,
            'current_value': current,
            'trend': trend,
            'direction': direction,
            'predicted_values': predicted_values,
            'confidence': confidence,
            'steps_ahead': steps_ahead,
            'data_points_used': len(recent),
            'timestamp': datetime.now().isoformat(),
        }
        
        self.predictions.append(prediction)
        self.save_state()
        
        logger.info(f"🔮 Prediction for {metric_name}: {direction} (confidence={confidence:.2f})")
        
        return prediction
    
    def predict_system_evolution(self, state_manager=None) -> Dict:
        """Predict overall system evolution."""
        predictions = {}
        
        for metric in self.trend_data:
            predictions[metric] = self.predict(metric, steps_ahead=3)
        
        # Overall assessment
        if predictions:
            avg_confidence = sum(p.get('confidence', 0) for p in predictions.values()) / len(predictions)
            directions = [p.get('direction', 'stable') for p in predictions.values()]
            improving = sum(1 for d in directions if d == 'increasing')
            declining = sum(1 for d in directions if d == 'decreasing')
        else:
            avg_confidence = 0
            improving = 0
            declining = 0
        
        return {
            'individual_predictions': predictions,
            'overall': {
                'avg_confidence': avg_confidence,
                'improving_metrics': improving,
                'declining_metrics': declining,
                'assessment': 'improving' if improving > declining else 'declining' if declining > improving else 'stable',
            },
            'timestamp': datetime.now().isoformat(),
        }
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'predictions': self.predictions[-50:],
            'trend_data': {k: v[-30:] for k, v in self.trend_data.items()},
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
                self.predictions = state.get('predictions', [])
                self.trend_data = defaultdict(list, state.get('trend_data', {}))
                logger.info(f"📂 Loaded future predictor state")
            except Exception as e:
                logger.warning(f"Failed to load future predictor: {e}")


class GenerativeWorldModel:
    """
    Top-level world model that orchestrates imagination,
    counterfactual reasoning, creative insights, and prediction.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        
        self.imagination = ImaginationEngine(state_dir)
        self.counterfactual = CounterfactualReasoner(state_dir)
        self.creative = CreativeInsightGenerator(state_dir)
        self.predictor = FuturePredictor(state_dir)
        
        logger.info(f"🌍 GenerativeWorldModel initialized (4 sub-systems)")
    
    def full_imagination_cycle(self, seed_concepts: List[str], memories: List[Dict] = None) -> Dict:
        """Run a complete imagination cycle."""
        results = {}
        
        # 1. Generate creative scenario
        scenario = self.imagination.imagine(seed_concepts, ScenarioType.CREATIVE)
        results['scenario'] = {
            'id': scenario.id,
            'premise': scenario.premise,
            'outcomes': scenario.generated_outcomes,
            'novelty': scenario.novelty,
        }
        
        # 2. Generate counterfactual
        cf = self.counterfactual.reason(
            event=seed_concepts[0] if seed_concepts else "something",
            alternative=f"farklı bir {seed_concepts[0] if seed_concepts else 'durum'}",
        )
        results['counterfactual'] = {
            'insight': cf['insight'],
            'divergence': cf['divergence_score'],
        }
        
        # 3. Creative insight
        insight = self.creative.generate_insight(seed_concepts, memories)
        results['insight'] = {
            'text': insight['text'],
            'novelty': insight['novelty'],
            'score': insight['score'],
        }
        
        # 4. Update trends and predict
        if memories:
            self.predictor.update_trend('memory_count', len(memories))
            self.predictor.update_trend('concept_count', len(seed_concepts))
        
        results['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"🌍 Full imagination cycle complete")
        
        return results
    
    def get_summary(self) -> str:
        """Summary of world model state."""
        lines = ["🌍 Generative World Model", "=" * 40]
        lines.append(f"💭 Imagination: {self.imagination.scenario_count} scenarios")
        lines.append(f"🔄 Counterfactuals: {len(self.counterfactual.counterfactual_history)}")
        lines.append(f"💡 Insights: {len(self.creative.insights)}")
        lines.append(f"🔮 Predictions: {len(self.predictor.predictions)}")
        
        if self.creative.insights:
            best = self.creative.get_best_insights(1)
            if best:
                lines.append(f"\nBest insight: {best[0]['text'][:60]}...")
        
        return "\n".join(lines)

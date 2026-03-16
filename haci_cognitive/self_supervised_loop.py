"""
Self-Supervised Learning Loop - Kendi Kendine Eğitim Döngüsü
"Kendi kendine öğrenen sistem"

Bu modül tüm Seviye 2 bileşenlerini birleştirir:
1. Curiosity Engine (merak motoru)
2. Predictive Coding (beklenti vs gerçeklik)
3. Active Learning Scheduler (aktif öğrenme planlayıcı)
4. Entegre öğrenme döngüsü

Çalışma döngüsü:
1. Gözlemle (observe) - Yeni verileri al
2. Değerlendir (assess) - Durumu analiz et
3. Planla (plan) - Öğrenme planı oluştur
4. Öğren (learn) - Öğrenme oturumunu çalıştır
5. Değerlendir (evaluate) - Sonuçları değerlendir
6. Güncelle (update) - Modelleri güncelle
"""

import json
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class SelfSupervisedLearningLoop:
    """
    Kendi kendine eğitim döngüsü.
    
    Tüm Seviye 2 bileşenlerini entegre eder ve
    otonom öğrenme döngüsünü yönetir.
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.state_file = self.workspace / "cognitive_state" / "learning_loop_state.json"
        
        # Component references (initialized lazily)
        self.curiosity_engine = None
        self.predictive_coding = None
        self.learning_scheduler = None
        
        # Loop state
        self.loop_state = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'current_cycle': None,
            'last_cycle': None,
            'is_running': False,
        }
        
        # Configuration
        self.config = {
            'cycle_interval_hours': 4,       # How often to run learning cycles
            'max_cycle_duration_min': 30,    # Max time per cycle
            'min_memories_for_cycle': 3,     # Min memories needed
            'auto_learning_enabled': True,
            'verbose_logging': True,
        }
        
        # Performance metrics
        self.metrics = {
            'total_topics_learned': 0,
            'total_insights_generated': 0,
            'total_surprises': 0,
            'avg_learning_depth': 0,
            'strategy_success_rates': {},
        }
        
        # Load state
        self.load_state()
        
        logger.info(f"🔄 SelfSupervisedLearningLoop initialized")
    
    def _init_components(self):
        """Initialize components lazily."""
        if self.curiosity_engine is None:
            try:
                from curiosity_engine import CuriosityEngine
                self.curiosity_engine = CuriosityEngine(str(self.workspace))
            except ImportError:
                logger.warning("CuriosityEngine not available")
        
        if self.predictive_coding is None:
            try:
                from predictive_coding import PredictiveCodingSystem
                self.predictive_coding = PredictiveCodingSystem(str(self.workspace))
            except ImportError:
                logger.warning("PredictiveCodingSystem not available")
        
        if self.learning_scheduler is None:
            try:
                from active_learning_scheduler import ActiveLearningScheduler
                self.learning_scheduler = ActiveLearningScheduler(str(self.workspace))
            except ImportError:
                logger.warning("ActiveLearningScheduler not available")
    
    def load_state(self):
        """Load loop state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.loop_state = state.get('loop_state', self.loop_state)
                    self.config.update(state.get('config', {}))
                    self.metrics = state.get('metrics', self.metrics)
                logger.info(f"📂 Loaded learning loop state: "
                           f"{self.loop_state['total_cycles']} cycles")
            except Exception as e:
                logger.warning(f"Failed to load loop state: {e}")
    
    def _convert_for_json(self, obj):
        """Convert objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif hasattr(obj, '__dict__'):  # Object
            return str(obj)
        else:
            return obj
    
    def save_state(self):
        """Save loop state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any enums to strings for JSON serialization
        state = {
            'loop_state': self._convert_for_json(self.loop_state),
            'config': self.config,
            'metrics': self.metrics,
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Learning loop state saved")
    
    # === Main Learning Cycle ===
    
    def run_learning_cycle(self, 
                          memories: List[Dict] = None,
                          user_messages: List[str] = None) -> Dict:
        """
        Tam öğrenme döngüsü çalıştır.
        
        Args:
            memories: Workspace'deki hafıza dosyaları
            user_messages: Kullanıcı mesajları (varsa)
        
        Returns:
            Cycle result dict
        """
        if not self.config['auto_learning_enabled']:
            return {'status': 'disabled'}
        
        self._init_components()
        
        cycle_start = time.time()
        cycle_id = self.loop_state['total_cycles'] + 1
        
        logger.info(f"\n🔄 === LEARNING CYCLE {cycle_id} STARTED ===")
        
        cycle_result = {
            'cycle_id': cycle_id,
            'started_at': datetime.now().isoformat(),
            'steps': {},
            'success': False,
            'duration_sec': 0,
        }
        
        try:
            # Step 1: Observe
            logger.info("👀 Step 1: Observe")
            observe_result = self._step_observe(memories, user_messages)
            cycle_result['steps']['observe'] = observe_result
            
            # Step 2: Assess
            logger.info("🔍 Step 2: Assess")
            assess_result = self._step_assess()
            cycle_result['steps']['assess'] = assess_result
            
            # Step 3: Plan
            logger.info("📋 Step 3: Plan")
            plan_result = self._step_plan(assess_result)
            cycle_result['steps']['plan'] = plan_result
            
            # Step 4: Learn
            logger.info("📚 Step 4: Learn")
            learn_result = self._step_learn(plan_result)
            cycle_result['steps']['learn'] = learn_result
            
            # Step 5: Evaluate
            logger.info("📊 Step 5: Evaluate")
            eval_result = self._step_evaluate(learn_result)
            cycle_result['steps']['evaluate'] = eval_result
            
            # Step 6: Update
            logger.info("🔄 Step 6: Update")
            update_result = self._step_update(eval_result)
            cycle_result['steps']['update'] = update_result
            
            # Mark success
            cycle_result['success'] = True
            self.loop_state['successful_cycles'] += 1
            
        except Exception as e:
            logger.error(f"❌ Cycle {cycle_id} failed: {e}")
            cycle_result['error'] = str(e)
        
        # Finalize
        cycle_result['duration_sec'] = time.time() - cycle_start
        cycle_result['completed_at'] = datetime.now().isoformat()
        
        self.loop_state['total_cycles'] = cycle_id
        self.loop_state['last_cycle'] = cycle_result
        self.save_state()
        
        logger.info(f"🔄 === LEARNING CYCLE {cycle_id} COMPLETE "
                   f"({'✅' if cycle_result['success'] else '❌'}) "
                   f"({cycle_result['duration_sec']:.1f}s) ===\n")
        
        return cycle_result
    
    def _step_observe(self, memories: List[Dict], 
                     user_messages: List[str]) -> Dict:
        """Step 1: Observe - Verileri topla."""
        result = {
            'n_memories': len(memories) if memories else 0,
            'n_user_messages': len(user_messages) if user_messages else 0,
        }
        
        # Update curiosity with new data
        if self.curiosity_engine and memories:
            self.curiosity_engine.update_interests(memories, user_messages)
            gaps = self.curiosity_engine.detect_knowledge_gaps(memories)
            result['n_gaps_detected'] = len(gaps)
        
        # Update predictive coding with new observations
        if self.predictive_coding and memories:
            for mem in memories[-5:]:
                content = mem.get('content', '')[:100]
                self.predictive_coding.observe_conversation(content, [])
        
        logger.info(f"   Observed: {result['n_memories']} memories, "
                   f"{result.get('n_gaps_detected', 0)} gaps")
        
        return result
    
    def _step_assess(self) -> Dict:
        """Step 2: Assess - Durumu değerlendir."""
        curiosity_state = None
        cognitive_state = None
        surprise_stats = None
        
        if self.curiosity_engine:
            curiosity_state = self.curiosity_engine.get_curiosity_state()
        
        if self.predictive_coding:
            surprise_stats = self.predictive_coding.get_system_stats()
        
        if self.learning_scheduler:
            assessment = self.learning_scheduler.assess_current_state(
                curiosity_state=curiosity_state or {},
                cognitive_state=cognitive_state or {},
                surprise_stats=surprise_stats or {}
            )
        else:
            assessment = {'recommended_strategy': 'explore', 'confidence': 0.5}
        
        logger.info(f"   Strategy: {assessment.get('recommended_strategy', 'unknown')}")
        
        return assessment
    
    def _step_plan(self, assessment: Dict) -> Dict:
        """Step 3: Plan - Öğrenme planı oluştur."""
        if not self.learning_scheduler:
            return {'strategy': 'explore', 'topics': []}
        
        # Get available topics from curiosity engine
        available_topics = []
        if self.curiosity_engine:
            interests = self.curiosity_engine.get_top_interests(10)
            gaps = self.curiosity_engine.detect_knowledge_gaps([])
            
            for interest in interests:
                available_topics.append({
                    'topic': interest['topic'],
                    'interest': interest['interest'],
                    'gap_score': 0.5,
                })
        
        plan = self.learning_scheduler.create_learning_plan(
            assessment=assessment,
            available_topics=available_topics,
        )
        
        logger.info(f"   Plan: {len(plan.get('topics', []))} topics, "
                   f"strategy={plan.get('strategy')}")
        
        return plan
    
    def _step_learn(self, plan: Dict) -> Dict:
        """Step 4: Learn - Öğrenme oturumunu çalıştır."""
        topics = plan.get('topics', [])
        strategy = plan.get('strategy', 'explore')
        
        results = []
        
        for topic_info in topics[:3]:  # Max 3 topics per cycle
            topic = topic_info.get('topic', 'unknown')
            
            # Simulate learning (in real system, this would involve actual research)
            learning_result = self._simulate_learning(topic, strategy)
            results.append(learning_result)
            
            # Record in scheduler
            if self.learning_scheduler:
                self.learning_scheduler.record_learning_session(
                    topic=topic,
                    strategy=strategy,
                    success=learning_result['success'],
                    depth=learning_result['depth'],
                    surprise=learning_result['surprise'],
                    duration_min=learning_result['duration_min'],
                )
            
            # Update curiosity engine
            if self.curiosity_engine:
                self.curiosity_engine.process_learning_result(
                    topic=topic,
                    success=learning_result['success'],
                    depth=learning_result['depth'],
                )
        
        # Update metrics
        self.metrics['total_topics_learned'] += len(results)
        
        logger.info(f"   Learned {len(results)} topics")
        
        return {
            'topics_learned': len(results),
            'results': results,
            'strategy_used': strategy,
        }
    
    def _simulate_learning(self, topic: str, strategy: str) -> Dict:
        """Simulate learning a topic (placeholder for real learning)."""
        import random
        
        # In real implementation, this would:
        # 1. Search for information about the topic
        # 2. Process the information
        # 3. Integrate with existing knowledge
        # 4. Generate insights
        
        success = random.random() > 0.2  # 80% success rate
        depth = random.uniform(0.3, 0.9) if success else random.uniform(0.1, 0.3)
        surprise = random.uniform(0.0, 0.5)
        duration = random.uniform(5, 15)
        
        return {
            'topic': topic,
            'strategy': strategy,
            'success': success,
            'depth': round(depth, 3),
            'surprise': round(surprise, 3),
            'duration_min': round(duration, 1),
            'timestamp': datetime.now().isoformat(),
        }
    
    def _step_evaluate(self, learn_result: Dict) -> Dict:
        """Step 5: Evaluate - Sonuçları değerlendir."""
        results = learn_result.get('results', [])
        
        if not results:
            return {'evaluation': 'no_results'}
        
        avg_depth = sum(r['depth'] for r in results) / len(results)
        avg_surprise = sum(r['surprise'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        evaluation = {
            'avg_depth': round(avg_depth, 3),
            'avg_surprise': round(avg_surprise, 3),
            'success_rate': round(success_rate, 3),
            'n_topics': len(results),
            'quality': 'good' if success_rate > 0.7 else 'fair' if success_rate > 0.4 else 'poor',
        }
        
        # Update metrics
        self.metrics['avg_learning_depth'] = (
            self.metrics['avg_learning_depth'] * 0.9 + avg_depth * 0.1
        )
        self.metrics['total_surprises'] += sum(1 for r in results if r['surprise'] > 0.5)
        
        logger.info(f"   Quality: {evaluation['quality']} "
                   f"(depth={avg_depth:.2f}, surprise={avg_surprise:.2f})")
        
        return evaluation
    
    def _step_update(self, eval_result: Dict) -> Dict:
        """Step 6: Update - Modelleri güncelle."""
        updates = []
        
        # Update predictive coding
        if self.predictive_coding:
            self.predictive_coding.save_state()
            updates.append('predictive_coding')
        
        # Update curiosity engine
        if self.curiosity_engine:
            self.curiosity_engine.save_state()
            updates.append('curiosity_engine')
        
        # Update learning scheduler
        if self.learning_scheduler:
            self.learning_scheduler.save_state()
            updates.append('learning_scheduler')
        
        logger.info(f"   Updated: {', '.join(updates)}")
        
        return {'updated_components': updates}
    
    # === Status & Reporting ===
    
    def get_loop_status(self) -> Dict:
        """Get current loop status."""
        return {
            'loop_state': self.loop_state,
            'metrics': self.metrics,
            'config': {
                'auto_learning_enabled': self.config['auto_learning_enabled'],
                'cycle_interval_hours': self.config['cycle_interval_hours'],
            },
            'component_status': {
                'curiosity_engine': self.curiosity_engine is not None,
                'predictive_coding': self.predictive_coding is not None,
                'learning_scheduler': self.learning_scheduler is not None,
            },
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "🔄 Self-Supervised Learning Loop",
            "=" * 40,
            "",
            f"📊 Cycles: {self.loop_state['total_cycles']} total, "
            f"{self.loop_state['successful_cycles']} successful",
            f"📚 Topics learned: {self.metrics['total_topics_learned']}",
            f"😮 Surprises: {self.metrics['total_surprises']}",
            f"📈 Avg learning depth: {self.metrics['avg_learning_depth']:.2f}",
            "",
            f"🔧 Auto-learning: {'✅' if self.config['auto_learning_enabled'] else '❌'}",
            f"⏰ Cycle interval: {self.config['cycle_interval_hours']}h",
            "",
            "Components:",
            f"  Curiosity Engine: {'✅' if self.curiosity_engine else '❌'}",
            f"  Predictive Coding: {'✅' if self.predictive_coding else '❌'}",
            f"  Learning Scheduler: {'✅' if self.learning_scheduler else '❌'}",
        ]
        
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    loop = SelfSupervisedLearningLoop(workspace)
    
    print("=== SelfSupervisedLearningLoop Test ===\n")
    
    # Test with sample memories
    sample_memories = [
        {'content': 'Başkan kahve seviyor espresso tercih ediyor Galatasaray'},
        {'content': 'WhatsApp watcher moondream ile çalışıyor retention system'},
        {'content': 'HaciCognitiveNet Level 1 tamamlandı dreaming loop'},
        {'content': 'Curiosity engine merak motoru predictive coding'},
        {'content': 'Active learning scheduler ögrenme planlayıcı'},
    ]
    
    # Run learning cycle
    result = loop.run_learning_cycle(memories=sample_memories)
    
    print(f"\n📊 Cycle Result:")
    print(f"   Success: {result['success']}")
    print(f"   Duration: {result['duration_sec']:.1f}s")
    print(f"   Steps: {list(result['steps'].keys())}")
    
    if 'learn' in result['steps']:
        learn = result['steps']['learn']
        print(f"   Topics learned: {learn['topics_learned']}")
    
    # Show summary
    print(f"\n{loop.get_summary()}")
    
    loop.save_state()
    print("\n✅ SelfSupervisedLearningLoop test complete!")

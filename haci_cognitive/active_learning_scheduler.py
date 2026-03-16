"""
Active Learning Scheduler - Aktif Öğrenme Planlayıcı
"Ne zaman, neyi, nasıl öğreneceğim?"

Bu modül:
1. Öğrenme önceliklerini belirler
2. Öğrenme oturumlarını planlar
3. Öğrenme stratejisini seçer
4. Öğrenme verimliliğini izler
"""

import json
import random
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Öğrenme stratejileri."""
    EXPLORE = "explore"           # Yeni konuları keşfet
    DEEPEN = "deepen"             # Mevcut bilgiyi derinleştir
    CONNECT = "connect"           # Konuları birbirine bağla
    REVIEW = "review"             # Tekrar et (spaced repetition)
    CONSOLIDATE = "consolidate"   # Bilgiyi pekiştir


class LearningPriority(Enum):
    """Öğrenme öncelikleri."""
    CRITICAL = 4    # Hemen öğrenilmeli
    HIGH = 3        # Bugün öğrenilmeli
    MEDIUM = 2      # Bu hafta öğrenilmeli
    LOW = 1         # Varsa öğren
    BACKLOG = 0     # Sırada bekliyor


class ActiveLearningScheduler:
    """
    Aktif öğrenme planlayıcı.
    
    Öğrenme döngüsü:
    1. Assess - Mevcut durumu değerlendir
    2. Plan - Öğrenme planı oluştur
    3. Execute - Öğrenme oturumunu çalıştır
    4. Evaluate - Öğrenme sonuçlarını değerlendir
    5. Adapt - Stratejiyi güncelle
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.state_file = self.workspace / "cognitive_state" / "learning_schedule.json"
        
        # Learning queue (topics to learn)
        self.learning_queue = []
        
        # Learning history
        self.learning_history = []
        
        # Spaced repetition tracking
        self.repetition_schedule = {}
        
        # Strategy performance tracking
        self.strategy_performance = {
            LearningStrategy.EXPLORE.value: {'attempts': 0, 'successes': 0},
            LearningStrategy.DEEPEN.value: {'attempts': 0, 'successes': 0},
            LearningStrategy.CONNECT.value: {'attempts': 0, 'successes': 0},
            LearningStrategy.REVIEW.value: {'attempts': 0, 'successes': 0},
            LearningStrategy.CONSOLIDATE.value: {'attempts': 0, 'successes': 0},
        }
        
        # Configuration
        self.config = {
            'max_daily_topics': 3,
            'max_consecutive_explore': 2,
            'spaced_repetition_base_days': [1, 3, 7, 14, 30],
            'min_time_between_sessions_min': 30,
            'preferred_session_duration_min': 15,
            'strategy_switch_threshold': 0.3,
        }
        
        # Load state
        self.load_state()
        
        logger.info(f"📅 ActiveLearningScheduler initialized")
    
    def load_state(self):
        """Load scheduler state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.learning_queue = state.get('learning_queue', [])
                    self.learning_history = state.get('learning_history', [])
                    self.repetition_schedule = state.get('repetition_schedule', {})
                    self.strategy_performance = state.get('strategy_performance', 
                                                         self.strategy_performance)
                    self.config.update(state.get('config', {}))
                logger.info(f"📂 Loaded learning schedule: {len(self.learning_queue)} queued")
            except Exception as e:
                logger.warning(f"Failed to load schedule: {e}")
    
    def save_state(self):
        """Save scheduler state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'learning_queue': self.learning_queue,
            'learning_history': self.learning_history[-100:],
            'repetition_schedule': self.repetition_schedule,
            'strategy_performance': self.strategy_performance,
            'config': self.config,
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Learning schedule saved")
    
    # === Assessment ===
    
    def assess_current_state(self, 
                            curiosity_state: Dict,
                            cognitive_state: Dict,
                            surprise_stats: Dict) -> Dict:
        """
        Mevcut durumu değerlendir.
        
        Returns:
            {
                'needs_exploration': bool,
                'needs_deepening': bool,
                'needs_review': bool,
                'recommended_strategy': LearningStrategy,
                'priority_topics': List[str],
            }
        """
        assessment = {
            'needs_exploration': False,
            'needs_deepening': False,
            'needs_review': False,
            'recommended_strategy': LearningStrategy.EXPLORE,
            'priority_topics': [],
            'confidence': 0.5,
        }
        
        # Check curiosity state
        if curiosity_state:
            n_gaps = curiosity_state.get('n_knowledge_gaps', 0)
            should_explore = curiosity_state.get('should_explore', (False, ''))[0]
            
            if n_gaps > 5 or should_explore:
                assessment['needs_exploration'] = True
                assessment['recommended_strategy'] = LearningStrategy.EXPLORE
        
        # Check cognitive state
        if cognitive_state:
            learning_efficiency = cognitive_state.get('learning_efficiency', 0.5)
            
            if learning_efficiency < 0.3:
                assessment['needs_review'] = True
                assessment['recommended_strategy'] = LearningStrategy.REVIEW
            elif learning_efficiency > 0.7:
                assessment['needs_deepening'] = True
                assessment['recommended_strategy'] = LearningStrategy.DEEPEN
        
        # Check surprise stats
        if surprise_stats:
            avg_surprise = surprise_stats.get('avg_surprise', 0)
            
            if avg_surprise > 0.7:
                # High surprise -> need to consolidate
                assessment['recommended_strategy'] = LearningStrategy.CONSOLIDATE
            elif avg_surprise < 0.2:
                # Low surprise -> need to explore
                assessment['recommended_strategy'] = LearningStrategy.EXPLORE
        
        # Check repetition schedule
        due_reviews = self.get_due_reviews()
        if due_reviews:
            assessment['needs_review'] = True
            if len(due_reviews) > 3:
                assessment['recommended_strategy'] = LearningStrategy.REVIEW
        
        # Confidence based on data availability
        data_sources = sum([
            bool(curiosity_state),
            bool(cognitive_state),
            bool(surprise_stats),
        ])
        assessment['confidence'] = data_sources / 3.0
        
        return assessment
    
    # === Planning ===
    
    def create_learning_plan(self, 
                            assessment: Dict,
                            available_topics: List[Dict]) -> Dict:
        """
        Öğrenme planı oluştur.
        
        Returns:
            {
                'strategy': LearningStrategy,
                'topics': List[Dict],
                'estimated_duration_min': int,
                'schedule': List[Dict],
            }
        """
        strategy = LearningStrategy(assessment['recommended_strategy'])
        max_topics = self.config['max_daily_topics']
        
        # Select topics based on strategy
        if strategy == LearningStrategy.EXPLORE:
            topics = self._select_explore_topics(available_topics, max_topics)
        elif strategy == LearningStrategy.DEEPEN:
            topics = self._select_deepen_topics(available_topics, max_topics)
        elif strategy == LearningStrategy.CONNECT:
            topics = self._select_connect_topics(available_topics, max_topics)
        elif strategy == LearningStrategy.REVIEW:
            topics = self._select_review_topics(max_topics)
        else:  # CONSOLIDATE
            topics = self._select_consolidate_topics(available_topics, max_topics)
        
        # Calculate estimated duration
        duration_per_topic = self.config['preferred_session_duration_min']
        estimated_duration = len(topics) * duration_per_topic
        
        # Create schedule
        schedule = []
        current_time = datetime.now()
        
        for i, topic in enumerate(topics):
            scheduled_time = current_time + timedelta(minutes=i * duration_per_topic)
            schedule.append({
                'topic': topic['topic'],
                'scheduled_time': scheduled_time.isoformat(),
                'strategy': strategy.value,
                'priority': topic.get('priority', 'MEDIUM'),
                'duration_min': duration_per_topic,
            })
        
        plan = {
            'strategy': strategy.value,
            'topics': topics,
            'estimated_duration_min': estimated_duration,
            'schedule': schedule,
            'created_at': datetime.now().isoformat(),
        }
        
        logger.info(f"📋 Learning plan created: {strategy.value} strategy, "
                   f"{len(topics)} topics, ~{estimated_duration}min")
        
        return plan
    
    def _select_explore_topics(self, available: List[Dict], max_n: int) -> List[Dict]:
        """Yeni konuları seç."""
        # Sort by gap score and interest
        sorted_topics = sorted(available, 
                              key=lambda x: x.get('gap_score', 0) + x.get('interest', 0),
                              reverse=True)
        return sorted_topics[:max_n]
    
    def _select_deepen_topics(self, available: List[Dict], max_n: int) -> List[Dict]:
        """Derinleştirilecek konuları seç."""
        # High interest, already known
        sorted_topics = sorted(available,
                              key=lambda x: x.get('interest', 0),
                              reverse=True)
        return sorted_topics[:max_n]
    
    def _select_connect_topics(self, available: List[Dict], max_n: int) -> List[Dict]:
        """Bağlantılı konuları seç."""
        # Find topics with potential connections
        connected = []
        for i, t1 in enumerate(available):
            for j, t2 in enumerate(available[i+1:], i+1):
                connected.append({
                    'topic': f"{t1['topic']} ↔ {t2['topic']}",
                    'type': 'connection',
                    'sources': [t1, t2],
                    'priority': 'MEDIUM',
                })
        
        return connected[:max_n]
    
    def _select_review_topics(self, max_n: int) -> List[Dict]:
        """Tekrar edilecek konuları seç."""
        due = self.get_due_reviews()
        return [
            {'topic': topic, 'priority': 'HIGH', 'type': 'review'}
            for topic in due[:max_n]
        ]
    
    def _select_consolidate_topics(self, available: List[Dict], max_n: int) -> List[Dict]:
        """Pekiştirilecek konuları seç."""
        # Recently learned topics
        recent = self.learning_history[-10:]
        topics = list(set(r['topic'] for r in recent if r.get('success', False)))
        
        return [
            {'topic': t, 'priority': 'MEDIUM', 'type': 'consolidate'}
            for t in topics[:max_n]
        ]
    
    # === Spaced Repetition ===
    
    def schedule_repetition(self, topic: str, quality: int):
        """
        Tekrar planla (spaced repetition).
        
        Args:
            topic: Konu
            quality: Kalite skoru 0-5 (0=totally forgot, 5=perfect recall)
        """
        if topic not in self.repetition_schedule:
            self.repetition_schedule[topic] = {
                'interval_idx': 0,
                'next_review': None,
                'reviews': [],
                'ease_factor': 2.5,
            }
        
        schedule = self.repetition_schedule[topic]
        
        # SM-2 algorithm (simplified)
        if quality >= 3:
            # Successful recall
            interval_idx = min(schedule['interval_idx'] + 1, 
                             len(self.config['spaced_repetition_base_days']) - 1)
            schedule['interval_idx'] = interval_idx
        else:
            # Failed recall - reset
            schedule['interval_idx'] = 0
        
        # Calculate next review date
        base_days = self.config['spaced_repetition_base_days'][schedule['interval_idx']]
        ease_factor = schedule['ease_factor']
        
        # Adjust ease factor
        schedule['ease_factor'] = max(1.3, 
            ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        
        interval_days = int(base_days * ease_factor)
        schedule['next_review'] = (datetime.now() + 
                                  timedelta(days=interval_days)).isoformat()
        
        schedule['reviews'].append({
            'date': datetime.now().isoformat(),
            'quality': quality,
            'interval_days': interval_days,
        })
        
        logger.info(f"🔄 Scheduled review for '{topic}': {interval_days} days")
    
    def get_due_reviews(self) -> List[str]:
        """Günü gelmiş tekrarları getir."""
        now = datetime.now()
        due = []
        
        for topic, schedule in self.repetition_schedule.items():
            next_review = schedule.get('next_review')
            if next_review:
                review_time = datetime.fromisoformat(next_review)
                if now >= review_time:
                    due.append(topic)
        
        return due
    
    # === Execution ===
    
    def record_learning_session(self, 
                               topic: str, 
                               strategy: str,
                               success: bool,
                               depth: float = 0.5,
                               surprise: float = 0.0,
                               duration_min: float = 0):
        """Öğrenme oturumunu kaydet."""
        session = {
            'topic': topic,
            'strategy': strategy,
            'success': success,
            'depth': depth,
            'surprise': surprise,
            'duration_min': duration_min,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.learning_history.append(session)
        
        # Update strategy performance
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            perf['attempts'] += 1
            if success:
                perf['successes'] += 1
        
        # Schedule spaced repetition if successful
        if success:
            quality = min(5, int(depth * 5))
            self.schedule_repetition(topic, quality)
        
        # Remove from queue if exists
        self.learning_queue = [t for t in self.learning_queue 
                              if t.get('topic') != topic]
        
        logger.info(f"📝 Learning session recorded: {topic} "
                   f"({'✅' if success else '❌'}) strategy={strategy}")
        
        self.save_state()
    
    # === Strategy Adaptation ===
    
    def get_best_strategy(self) -> str:
        """En iyi stratejiyi getir."""
        best_strategy = LearningStrategy.EXPLORE.value
        best_rate = 0
        
        for strategy, perf in self.strategy_performance.items():
            if perf['attempts'] >= 3:
                rate = perf['successes'] / perf['attempts']
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy
        
        return best_strategy
    
    def get_stats(self) -> Dict:
        """Planlayıcı istatistiklerini getir."""
        total_sessions = len(self.learning_history)
        successful = sum(1 for s in self.learning_history if s.get('success'))
        
        return {
            'total_sessions': total_sessions,
            'successful_sessions': successful,
            'success_rate': round(successful / max(total_sessions, 1), 3),
            'queued_topics': len(self.learning_queue),
            'due_reviews': len(self.get_due_reviews()),
            'strategy_performance': self.strategy_performance,
            'best_strategy': self.get_best_strategy(),
            'repetition_topics': len(self.repetition_schedule),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    scheduler = ActiveLearningScheduler(workspace)
    
    print("=== ActiveLearningScheduler Test ===\n")
    
    # Test assessment
    assessment = scheduler.assess_current_state(
        curiosity_state={'n_knowledge_gaps': 8, 'should_explore': (True, 'gaps')},
        cognitive_state={'learning_efficiency': 0.6},
        surprise_stats={'avg_surprise': 0.3}
    )
    
    print(f"📊 Assessment:")
    print(f"   Strategy: {assessment['recommended_strategy']}")
    print(f"   Needs exploration: {assessment['needs_exploration']}")
    print(f"   Confidence: {assessment['confidence']:.2f}")
    
    # Test planning
    available_topics = [
        {'topic': 'kahve', 'gap_score': 0.6, 'interest': 0.8},
        {'topic': 'teknoloji', 'gap_score': 0.4, 'interest': 0.7},
        {'topic': 'spor', 'gap_score': 0.7, 'interest': 0.5},
    ]
    
    plan = scheduler.create_learning_plan(assessment, available_topics)
    print(f"\n📋 Plan:")
    print(f"   Strategy: {plan['strategy']}")
    print(f"   Topics: {[t['topic'] for t in plan['topics']]}")
    print(f"   Duration: ~{plan['estimated_duration_min']}min")
    
    # Test spaced repetition
    scheduler.schedule_repetition("kahve", 4)  # Good recall
    print(f"\n🔄 Due reviews: {scheduler.get_due_reviews()}")
    
    # Test session recording
    scheduler.record_learning_session(
        topic="kahve",
        strategy="explore",
        success=True,
        depth=0.8,
        surprise=0.2,
        duration_min=12
    )
    
    stats = scheduler.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    scheduler.save_state()
    print("\n✅ ActiveLearningScheduler test complete!")

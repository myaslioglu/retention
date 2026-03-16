"""
Curiosity Engine - Merak Motoru
"Ne öğrenmek istiyorum?" sorusunu cevaplayan modül.

Aktiviteler:
1. Knowledge gap detection - Bilgi boşluklarını tespit et
2. Interest modeling - İlgi alanlarını modelle
3. Question generation - Meraklı sorular üret
4. Exploration scheduling - Ne zaman keşif yapılacağını planla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CuriosityEngine:
    """
    Merak motoru - sistemin kendi ilgi alanlarını ve bilgi boşluklarını
    keşfetmesini sağlar.
    
    Bileşenler:
    - Knowledge Gap Detector
    - Interest Modeler
    - Question Generator
    - Exploration Scheduler
    """
    
    def __init__(self, workspace_dir: str, state_dim: int = 128):
        self.workspace = Path(workspace_dir)
        self.state_file = self.workspace / "cognitive_state" / "curiosity_state.json"
        
        # Interest model (topic -> interest level)
        self.interests = {}
        
        # Knowledge gaps (topic -> gap score)
        self.knowledge_gaps = {}
        
        # Question bank (generated questions)
        self.question_bank = []
        
        # Exploration history
        self.exploration_log = []
        
        # State dimension
        self.state_dim = state_dim
        
        # Configuration
        self.config = {
            'min_interest_threshold': 0.3,    # Minimum interest to explore
            'gap_weight': 0.6,                # Weight for knowledge gaps
            'novelty_weight': 0.4,            # Weight for novelty
            'max_questions_per_cycle': 5,     # Max questions per cycle
            'interest_decay_rate': 0.02,      # Daily interest decay
            'exploration_cooldown_hours': 6,  # Min time between explorations
            'random_exploration_prob': 0.15,  # Probability of random exploration
        }
        
        # Initialize from saved state
        self.load_state()
        
        logger.info(f"🔍 CuriosityEngine initialized")
    
    def load_state(self):
        """Load curiosity state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.interests = state.get('interests', {})
                self.knowledge_gaps = state.get('knowledge_gaps', {})
                self.question_bank = state.get('question_bank', [])
                self.exploration_log = state.get('exploration_log', [])
                self.config.update(state.get('config', {}))
            logger.info(f"📂 Loaded curiosity state: {len(self.interests)} interests, "
                       f"{len(self.knowledge_gaps)} gaps")
    
    def save_state(self):
        """Save curiosity state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'interests': self.interests,
            'knowledge_gaps': self.knowledge_gaps,
            'question_bank': self.question_bank[-100:],  # Keep last 100
            'exploration_log': self.exploration_log[-50:],  # Keep last 50
            'config': self.config,
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Curiosity state saved")
    
    # === Knowledge Gap Detection ===
    
    def detect_knowledge_gaps(self, memories: List[Dict]) -> List[Dict]:
        """
        Bilgi boşluklarını tespit et.
        
        Analiz edilen faktörler:
        - Az bahsedilen konular
        - Yüzeysel bilgi (kısa içerik)
        - Tekrar eden sorular
        - Güncel olmayan bilgiler
        """
        gaps = []
        
        # Topic frequency analysis
        topic_freq = {}
        topic_depth = {}
        
        for mem in memories:
            content = mem.get('content', '')
            words = content.lower().split()
            
            for word in words:
                if len(word) > 3:
                    topic_freq[word] = topic_freq.get(word, 0) + 1
                    # Depth = content length for this topic
                    topic_depth[word] = topic_depth.get(word, 0) + len(content)
        
        # Calculate gap scores
        for topic, freq in topic_freq.items():
            avg_depth = topic_depth[topic] / max(freq, 1)
            
            # Gap score: low frequency + low depth = high gap
            freq_score = 1.0 / (1.0 + math.log(1 + freq))
            depth_score = 1.0 / (1.0 + math.log(1 + avg_depth / 100))
            
            gap_score = (freq_score + depth_score) / 2.0
            
            if gap_score > 0.3:  # Only significant gaps
                gaps.append({
                    'topic': topic,
                    'gap_score': round(gap_score, 3),
                    'frequency': freq,
                    'avg_depth': round(avg_depth, 1),
                })
        
        # Sort by gap score
        gaps.sort(key=lambda x: x['gap_score'], reverse=True)
        
        # Update internal state
        for gap in gaps[:20]:
            self.knowledge_gaps[gap['topic']] = gap['gap_score']
        
        logger.info(f"🔍 Detected {len(gaps)} knowledge gaps")
        return gaps[:20]
    
    # === Interest Modeling ===
    
    def update_interests(self, memories: List[Dict], user_messages: List[str] = None):
        """
        İlgi alanlarını güncelle.
        
        Faktörler:
        - Konuşma sıklığı
        - İçerik derinliği
        - Kullanıcı mesajlarındaki konular
        - Zaman içindeki değişim
        """
        # Decay existing interests
        for topic in self.interests:
            self.interests[topic] *= (1 - self.config['interest_decay_rate'])
        
        # Analyze memories
        for mem in memories:
            content = mem.get('content', '').lower()
            words = content.split()
            
            for word in words:
                if len(word) > 3:
                    # Boost interest
                    current = self.interests.get(word, 0)
                    self.interests[word] = min(1.0, current + 0.05)
        
        # Analyze user messages if available
        if user_messages:
            for msg in user_messages:
                words = msg.lower().split()
                for word in words:
                    if len(word) > 3:
                        current = self.interests.get(word, 0)
                        # User messages get higher weight
                        self.interests[word] = min(1.0, current + 0.1)
        
        # Clean up low interests
        self.interests = {k: v for k, v in self.interests.items() 
                         if v > self.config['min_interest_threshold']}
        
        logger.info(f"📊 Updated interests: {len(self.interests)} active topics")
    
    def get_top_interests(self, n: int = 10) -> List[Dict]:
        """En yüksek ilgi alanlarını getir."""
        sorted_interests = sorted(self.interests.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return [
            {'topic': topic, 'interest': round(score, 3)}
            for topic, score in sorted_interests[:n]
        ]
    
    # === Question Generation ===
    
    def generate_curious_questions(self, context: Dict = None) -> List[Dict]:
        """
        Meraklı sorular üret.
        
        Soru tipleri:
        1. Gap-filling: "X hakkında daha fazla bilgi edinmek istiyorum"
        2. Connection: "X ve Y arasındaki bağlantı nedir?"
        3. Prediction: "X'in geleceği ne olacak?"
        4. Counterfactual: "X olmasaydı ne olurdu?"
        """
        questions = []
        
        # Template-based question generation
        gap_templates = [
            "{topic} hakkında daha fazla bilgi edinmek istiyorum",
            "{topic} konusunda derinlemesine ne biliyorum?",
            "{topic} ile ilgili güncel gelişmeler neler?",
            "{topic} neden önemli?",
            "{topic} nasıl çalışıyor?",
        ]
        
        connection_templates = [
            "{topic1} ve {topic2} arasında nasıl bir bağlantı var?",
            "{topic1} {topic2}'yi nasıl etkiliyor?",
            "{topic1} ile {topic2} arasındaki ilişki nedir?",
        ]
        
        prediction_templates = [
            "{topic} gelecekte nasıl değişecek?",
            "{topic} trendi ne yönde ilerliyor?",
            "{topic} 5 yıl sonra ne olacak?",
        ]
        
        # Generate gap-filling questions
        top_gaps = sorted(self.knowledge_gaps.items(), 
                         key=lambda x: x[1], reverse=True)[:5]
        
        for topic, gap_score in top_gaps:
            template = random.choice(gap_templates)
            questions.append({
                'type': 'gap_filling',
                'question': template.format(topic=topic),
                'topic': topic,
                'gap_score': gap_score,
                'generated_at': datetime.now().isoformat(),
            })
        
        # Generate connection questions
        top_topics = list(self.interests.keys())[:10]
        if len(top_topics) >= 2:
            for _ in range(min(2, len(top_topics) // 2)):
                t1, t2 = random.sample(top_topics, 2)
                template = random.choice(connection_templates)
                questions.append({
                    'type': 'connection',
                    'question': template.format(topic1=t1, topic2=t2),
                    'topics': [t1, t2],
                    'generated_at': datetime.now().isoformat(),
                })
        
        # Generate prediction questions
        if top_topics:
            topic = random.choice(top_topics[:5])
            template = random.choice(prediction_templates)
            questions.append({
                'type': 'prediction',
                'question': template.format(topic=topic),
                'topic': topic,
                'generated_at': datetime.now().isoformat(),
            })
        
        # Random exploration (serendipity)
        if random.random() < self.config['random_exploration_prob']:
            random_topics = ['yapay zeka', 'uzay', 'felsefe', 'matematik', 
                           'tarih', 'bilim', 'teknoloji', 'sanat', 'müzik', 'spor']
            topic = random.choice(random_topics)
            questions.append({
                'type': 'random_exploration',
                'question': f"{topic} hakkında yeni bir şey öğrenmek istiyorum",
                'topic': topic,
                'generated_at': datetime.now().isoformat(),
            })
        
        # Limit and save
        questions = questions[:self.config['max_questions_per_cycle']]
        self.question_bank.extend(questions)
        
        logger.info(f"❓ Generated {len(questions)} curious questions")
        return questions
    
    # === Exploration Scheduling ===
    
    def should_explore(self) -> Tuple[bool, Optional[str]]:
        """
        Keşif yapıp yapmama kararı ver.
        
        Döner: (should_explore, reason)
        """
        # Check cooldown
        if self.exploration_log:
            last_exp = self.exploration_log[-1]
            last_time = datetime.fromisoformat(last_exp['timestamp'])
            cooldown = timedelta(hours=self.config['exploration_cooldown_hours'])
            
            if datetime.now() - last_time < cooldown:
                return False, "cooldown_active"
        
        # Check if there are significant gaps
        if self.knowledge_gaps:
            max_gap = max(self.knowledge_gaps.values())
            if max_gap > 0.5:
                return True, f"significant_gap (score={max_gap:.3f})"
        
        # Check curiosity level (from cognitive state)
        if len(self.question_bank) > 5:
            return True, "question_accumulation"
        
        # Random exploration
        if random.random() < self.config['random_exploration_prob']:
            return True, "random_exploration"
        
        return False, "no_urgency"
    
    def get_exploration_target(self) -> Optional[Dict]:
        """
        Keşif hedefi belirle.
        
        Döner: {topic, reason, priority, questions}
        """
        should, reason = self.should_explore()
        
        if not should:
            return None
        
        # Find best target
        candidates = []
        
        # From gaps
        for topic, gap_score in self.knowledge_gaps.items():
            interest = self.interests.get(topic, 0.5)
            priority = (gap_score * self.config['gap_weight'] + 
                       interest * self.config['novelty_weight'])
            candidates.append({
                'topic': topic,
                'priority': priority,
                'source': 'knowledge_gap',
                'gap_score': gap_score,
                'interest': interest,
            })
        
        # From questions
        unanswered = [q for q in self.question_bank[-10:] 
                     if q.get('status') != 'answered']
        for q in unanswered:
            topic = q.get('topic', 'unknown')
            candidates.append({
                'topic': topic,
                'priority': 0.7,
                'source': 'question',
                'question': q['question'],
            })
        
        if not candidates:
            return None
        
        # Sort by priority
        candidates.sort(key=lambda x: x.get('priority', 0), reverse=True)
        target = candidates[0]
        
        # Add related questions
        target['questions'] = [
            q['question'] for q in self.question_bank
            if q.get('topic') == target['topic'] and q.get('status') != 'answered'
        ][:3]
        
        # Log exploration
        self.exploration_log.append({
            'timestamp': datetime.now().isoformat(),
            'topic': target['topic'],
            'reason': reason,
            'priority': target['priority'],
        })
        
        logger.info(f"🎯 Exploration target: {target['topic']} "
                   f"(priority={target['priority']:.3f}, reason={reason})")
        
        return target
    
    # === Integration with Cognitive System ===
    
    def get_curiosity_state(self) -> Dict:
        """Get full curiosity state for cognitive system."""
        return {
            'n_interests': len(self.interests),
            'n_knowledge_gaps': len(self.knowledge_gaps),
            'n_questions': len(self.question_bank),
            'n_explorations': len(self.exploration_log),
            'top_interests': self.get_top_interests(5),
            'top_gaps': sorted(self.knowledge_gaps.items(), 
                              key=lambda x: x[1], reverse=True)[:5],
            'should_explore': self.should_explore(),
            'last_exploration': self.exploration_log[-1] if self.exploration_log else None,
        }
    
    def process_learning_result(self, topic: str, success: bool, depth: float = 0.5):
        """
        Öğrenme sonucunu işle.
        
        Args:
            topic: Öğrenilen konu
            success: Başarılı mı?
            depth: Öğrenme derinliği (0-1)
        """
        if success:
            # Reduce knowledge gap
            if topic in self.knowledge_gaps:
                self.knowledge_gaps[topic] *= (1 - depth)
                if self.knowledge_gaps[topic] < 0.1:
                    del self.knowledge_gaps[topic]
            
            # Boost interest
            current = self.interests.get(topic, 0.5)
            self.interests[topic] = min(1.0, current + 0.1)
            
            # Mark related questions as answered
            for q in self.question_bank:
                if q.get('topic') == topic and q.get('status') != 'answered':
                    q['status'] = 'answered'
                    q['answered_at'] = datetime.now().isoformat()
            
            logger.info(f"✅ Learning successful: {topic} (depth={depth})")
        else:
            # Increase gap slightly
            current = self.knowledge_gaps.get(topic, 0.5)
            self.knowledge_gaps[topic] = min(1.0, current + 0.1)
            
            logger.info(f"❌ Learning failed: {topic}")
        
        self.save_state()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    engine = CuriosityEngine(workspace)
    
    print("=== CuriosityEngine Test ===\n")
    
    # Test with sample memories
    sample_memories = [
        {'content': 'Başkan kahve seviyor espresso tercih ediyor'},
        {'content': 'Galatasaray taraftarı olduğunu öğrendim'},
        {'content': 'WhatsApp watcher moondream ile çalışıyor'},
        {'content': 'Retention system kuruldu FAISS index'},
        {'content': 'HaciCognitiveNet Level 1 tamamlandı'},
    ]
    
    # Detect gaps
    gaps = engine.detect_knowledge_gaps(sample_memories)
    print(f"📊 Knowledge Gaps: {len(gaps)}")
    for g in gaps[:5]:
        print(f"   {g['topic']}: {g['gap_score']:.3f}")
    
    # Update interests
    engine.update_interests(sample_memories)
    interests = engine.get_top_interests(5)
    print(f"\n🔍 Top Interests: {len(interests)}")
    for i in interests:
        print(f"   {i['topic']}: {i['interest']:.3f}")
    
    # Generate questions
    questions = engine.generate_curious_questions()
    print(f"\n❓ Generated Questions: {len(questions)}")
    for q in questions:
        print(f"   [{q['type']}] {q['question']}")
    
    # Check exploration
    target = engine.get_exploration_target()
    if target:
        print(f"\n🎯 Exploration Target: {target['topic']}")
        print(f"   Priority: {target['priority']:.3f}")
        print(f"   Reason: {target.get('source', 'unknown')}")
    
    # Save state
    engine.save_state()
    
    print("\n✅ CuriosityEngine test complete!")

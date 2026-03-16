"""
Predictive Coding - Beklenti vs Gerçeklik Karşılaştırması
"Beklediğim neydi, gerçekte ne oldu?"

Bu modül:
1. Geçmiş deneyimlere dayanarak tahminler yapar
2. Tahminler ile gerçek sonuçları karşılaştırır
3. Şaşkınlık (surprise) seviyesini hesaplar
4. Modeli şaşkınlığa göre günceller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictiveModel:
    """
    Basit prediktif model - olayların olasılıklarını tahmin eder.
    
    Kullanım:
    - Bir olay gözlemlendiğinde, model bir sonraki olayı tahmin eder
    - Gerçek sonuçla karşılaştırılır
    - Model güncellenir (learning from surprise)
    """
    
    def __init__(self, vocab_size: int = 1000, context_window: int = 10):
        self.vocab_size = vocab_size
        self.context_window = context_window
        
        # Transition matrix (event -> next event probabilities)
        self.transitions = {}
        
        # Bigram counts
        self.bigram_counts = {}
        self.total_counts = {}
        
        # Surprise history
        self.surprise_log = []
        
        # Prediction accuracy tracking
        self.accuracy_history = []
        
        logger.info(f"🔮 PredictiveModel initialized (vocab={vocab_size})")
    
    def _normalize_event(self, event: str) -> str:
        """Normalize event string."""
        return event.lower().strip()[:50]
    
    def observe(self, event: str, context: List[str] = None):
        """
        Bir olayı gözlele ve modeli güncelle.
        
        Args:
            event: Gözlemlenen olay
            context: Önceki olaylar (bağlam)
        """
        event = self._normalize_event(event)
        
        if context:
            # Update bigram counts
            for prev_event in context[-self.context_window:]:
                prev = self._normalize_event(prev_event)
                
                if prev not in self.bigram_counts:
                    self.bigram_counts[prev] = {}
                    self.total_counts[prev] = 0
                
                self.bigram_counts[prev][event] = \
                    self.bigram_counts[prev].get(event, 0) + 1
                self.total_counts[prev] += 1
        
        # Update unigram
        if '_unigram_' not in self.bigram_counts:
            self.bigram_counts['_unigram_'] = {}
            self.total_counts['_unigram_'] = 0
        
        self.bigram_counts['_unigram_'][event] = \
            self.bigram_counts['_unigram_'].get(event, 0) + 1
        self.total_counts['_unigram_'] += 1
    
    def predict(self, context: List[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Sonraki olayı tahmin et.
        
        Returns:
            List of {event, probability} sorted by probability
        """
        predictions = {}
        
        if context:
            # Use most recent context
            prev_event = self._normalize_event(context[-1]) if context else None
            
            if prev_event and prev_event in self.bigram_counts:
                total = self.total_counts[prev_event]
                for event, count in self.bigram_counts[prev_event].items():
                    # Laplace smoothing
                    prob = (count + 1) / (total + self.vocab_size)
                    predictions[event] = prob
        
        # Fall back to unigram
        if not predictions and '_unigram_' in self.bigram_counts:
            total = self.total_counts['_unigram_']
            for event, count in self.bigram_counts['_unigram_'].items():
                prob = (count + 1) / (total + self.vocab_size)
                predictions[event] = prob
        
        # Sort and return top_k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'event': event, 'probability': round(prob, 6)}
            for event, prob in sorted_preds[:top_k]
        ]
    
    def calculate_surprise(self, actual_event: str, 
                          context: List[str] = None) -> Dict:
        """
        Şaşkınlık hesapla - beklenti vs gerçeklik.
        
        Returns:
            {surprise_score, predicted_events, actual_event, top_prediction}
        """
        predictions = self.predict(context)
        
        actual = self._normalize_event(actual_event)
        
        # Find actual event probability
        actual_prob = 0
        for pred in predictions:
            if pred['event'] == actual:
                actual_prob = pred['probability']
                break
        
        # If not in predictions, it's very surprising
        if actual_prob == 0:
            actual_prob = 1e-10  # Small epsilon
        
        # Surprise = -log(probability)
        surprise = -math.log(actual_prob)
        
        # Normalize to 0-1
        max_surprise = -math.log(1e-10)
        surprise_normalized = min(1.0, surprise / max_surprise)
        
        result = {
            'surprise_score': round(surprise_normalized, 4),
            'raw_surprise': round(surprise, 4),
            'actual_event': actual_event,
            'actual_probability': round(actual_prob, 6),
            'top_prediction': predictions[0] if predictions else None,
            'all_predictions': predictions,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Log surprise
        self.surprise_log.append(result)
        
        return result
    
    def update_accuracy(self, predicted: str, actual: str):
        """Tahmin doğruluğunu güncelle."""
        correct = self._normalize_event(predicted) == self._normalize_event(actual)
        self.accuracy_history.append({
            'correct': correct,
            'predicted': predicted,
            'actual': actual,
            'timestamp': datetime.now().isoformat(),
        })
    
    def get_accuracy_stats(self, window: int = 100) -> Dict:
        """Son N tahminin doğruluk istatistiklerini getir."""
        recent = self.accuracy_history[-window:]
        
        if not recent:
            return {'accuracy': 0, 'total': 0, 'correct': 0}
        
        correct = sum(1 for r in recent if r['correct'])
        total = len(recent)
        
        return {
            'accuracy': round(correct / total, 3),
            'total': total,
            'correct': correct,
            'window': window,
        }
    
    def get_surprise_stats(self, window: int = 50) -> Dict:
        """Şaşkınlık istatistiklerini getir."""
        recent = self.surprise_log[-window:]
        
        if not recent:
            return {'avg_surprise': 0, 'max_surprise': 0, 'n_events': 0}
        
        surprises = [r['surprise_score'] for r in recent]
        
        return {
            'avg_surprise': round(sum(surprises) / len(surprises), 4),
            'max_surprise': round(max(surprises), 4),
            'min_surprise': round(min(surprises), 4),
            'n_events': len(recent),
            'high_surprise_events': [
                r for r in recent if r['surprise_score'] > 0.7
            ][-5:],
        }


class PredictiveCodingSystem:
    """
    Prediktif kodlama sistemi - çoklu model ile çalışır.
    
    Modeller:
    1. Conversation model - konuşma akışını tahmin eder
    2. Topic model - konu değişimlerini tahmin eder
    3. Emotion model - duygu değişimlerini tahmin eder
    4. Action model - aksiyon sonuçlarını tahmin eder
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.state_file = self.workspace / "cognitive_state" / "predictive_state.json"
        
        # Initialize models
        self.conversation_model = PredictiveModel(vocab_size=500)
        self.topic_model = PredictiveModel(vocab_size=200)
        self.emotion_model = PredictiveModel(vocab_size=10)
        self.action_model = PredictiveModel(vocab_size=100)
        
        # Global surprise threshold
        self.surprise_threshold = 0.5
        
        # Load state
        self.load_state()
        
        logger.info(f"🔮 PredictiveCodingSystem initialized")
    
    def load_state(self):
        """Load predictive state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    
                # Restore model states
                if 'conversation' in state:
                    self.conversation_model.bigram_counts = state['conversation'].get('bigram_counts', {})
                    self.conversation_model.total_counts = state['conversation'].get('total_counts', {})
                    self.conversation_model.accuracy_history = state['conversation'].get('accuracy_history', [])
                
                if 'topic' in state:
                    self.topic_model.bigram_counts = state['topic'].get('bigram_counts', {})
                    self.topic_model.total_counts = state['topic'].get('total_counts', {})
                
                if 'emotion' in state:
                    self.emotion_model.bigram_counts = state['emotion'].get('bigram_counts', {})
                    self.emotion_model.total_counts = state['emotion'].get('total_counts', {})
                
                if 'action' in state:
                    self.action_model.bigram_counts = state['action'].get('bigram_counts', {})
                    self.action_model.total_counts = state['action'].get('total_counts', {})
                
                logger.info(f"📂 Loaded predictive state")
            except Exception as e:
                logger.warning(f"Failed to load predictive state: {e}")
    
    def save_state(self):
        """Save predictive state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'conversation': {
                'bigram_counts': self.conversation_model.bigram_counts,
                'total_counts': self.conversation_model.total_counts,
                'accuracy_history': self.conversation_model.accuracy_history[-50:],
            },
            'topic': {
                'bigram_counts': self.topic_model.bigram_counts,
                'total_counts': self.topic_model.total_counts,
            },
            'emotion': {
                'bigram_counts': self.emotion_model.bigram_counts,
                'total_counts': self.emotion_model.total_counts,
            },
            'action': {
                'bigram_counts': self.action_model.bigram_counts,
                'total_counts': self.action_model.total_counts,
            },
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def observe_conversation(self, message: str, context: List[str]):
        """Konuşma akışını gözle."""
        self.conversation_model.observe(message, context)
    
    def observe_topic_change(self, old_topic: str, new_topic: str):
        """Konu değişimini gözle."""
        self.topic_model.observe(new_topic, [old_topic])
    
    def observe_emotion_change(self, old_emotion: str, new_emotion: str):
        """Duygu değişimini gözle."""
        self.emotion_model.observe(new_emotion, [old_emotion])
    
    def observe_action_result(self, action: str, result: str, context: List[str]):
        """Aksiyon sonucunu gözle."""
        self.action_model.observe(f"{action}→{result}", context)
    
    def predict_next_topic(self, current_topic: str) -> List[Dict]:
        """Sonraki konuyu tahmin et."""
        return self.topic_model.predict([current_topic])
    
    def predict_next_emotion(self, current_emotion: str) -> List[Dict]:
        """Sonraki duyguyu tahmin et."""
        return self.emotion_model.predict([current_emotion])
    
    def calculate_system_surprise(self, events: Dict[str, str]) -> Dict:
        """
        Sistem genelinde şaşkınlık hesapla.
        
        Args:
            events: {model_type: actual_event}
        """
        total_surprise = 0
        surprises = {}
        
        model_map = {
            'conversation': self.conversation_model,
            'topic': self.topic_model,
            'emotion': self.emotion_model,
            'action': self.action_model,
        }
        
        for model_type, actual_event in events.items():
            if model_type in model_map:
                model = model_map[model_type]
                surprise = model.calculate_surprise(actual_event)
                surprises[model_type] = surprise
                total_surprise += surprise['surprise_score']
        
        avg_surprise = total_surprise / max(len(surprises), 1)
        
        return {
            'average_surprise': round(avg_surprise, 4),
            'surprises': surprises,
            'is_surprising': avg_surprise > self.surprise_threshold,
            'timestamp': datetime.now().isoformat(),
        }
    
    def get_system_stats(self) -> Dict:
        """Tüm modellerin istatistiklerini getir."""
        return {
            'conversation': self.conversation_model.get_accuracy_stats(),
            'topic_surprise': self.topic_model.get_surprise_stats(),
            'emotion_surprise': self.emotion_model.get_surprise_stats(),
            'action_accuracy': self.action_model.get_accuracy_stats(),
            'surprise_threshold': self.surprise_threshold,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    workspace = os.path.expanduser("~/.openclaw/workspace")
    system = PredictiveCodingSystem(workspace)
    
    print("=== PredictiveCodingSystem Test ===\n")
    
    # Test topic prediction
    topics = ["kahve", "teknoloji", "kahve", "spor", "teknoloji", 
              "kahve", "müzik", "teknoloji", "kahve"]
    
    for i, topic in enumerate(topics):
        if i > 0:
            system.observe_topic_change(topics[i-1], topic)
    
    # Predict next topic after "kahve"
    predictions = system.predict_next_topic("kahve")
    print(f"🔮 After 'kahve', predicted topics:")
    for p in predictions[:3]:
        print(f"   {p['event']}: {p['probability']:.3f}")
    
    # Calculate surprise
    surprise = system.calculate_system_surprise({
        'topic': 'felsefe',  # Unexpected!
    })
    print(f"\n😮 Surprise for 'felsefe': {surprise['surprises']['topic']['surprise_score']:.3f}")
    print(f"   Is surprising? {surprise['is_surprising']}")
    
    # System stats
    stats = system.get_system_stats()
    print(f"\n📊 System Stats:")
    print(f"   Topic surprise avg: {stats['topic_surprise']['avg_surprise']}")
    print(f"   Surprise threshold: {stats['surprise_threshold']}")
    
    # Save state
    system.save_state()
    
    print("\n✅ PredictiveCodingSystem test complete!")

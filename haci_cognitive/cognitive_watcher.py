"""
Cognitive Watcher - Her mesajda çalışan hafif gözlemci

Her WhatsApp mesajında otomatik çalışır:
- Negatif sinyal tespiti
- Duygu analizi
- Etkileşim tipi sınıflandırması
- Kişilik trait güncellemesi

ÖNEMLİ: <100ms çalışmalı - ağır işlem yok, sadece pattern matching
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Modül dizini
_MODULE_DIR = Path(__file__).parent

# Social trainer import (lazy)
_social_trainer = None


def _get_social_trainer():
    """Social trainer'ı lazy yükle."""
    global _social_trainer
    if _social_trainer is None:
        try:
            import sys
            if str(_MODULE_DIR) not in sys.path:
                sys.path.insert(0, str(_MODULE_DIR))
            from social_trainer import SocialIntelligenceTrainer
            _social_trainer = SocialIntelligenceTrainer(
                state_dir=str(_MODULE_DIR.parent / "cognitive_state")
            )
        except Exception as e:
            logger.warning(f"Social trainer yüklenemedi: {e}")
    return _social_trainer


# === DUYGU PATERNLERİ (hızlı regex tabanlı) ===
EMOTION_PATTERNS = {
    'happy': [
        r'\b(harika|güzel|süper|muhteşem|bravo|helal|aferin)\b',
        r'[😊😄😃🥰😍❤️🎉🥳💯🔥]',
        r'\b(mutlu|sevinçli|keyifli|eğlenceli)\b',
    ],
    'sad': [
        r'\b(üzgün|kötü|berbat|kötü|moralsiz)\b',
        r'[😢😭😞😔💔🥺]',
        r'\b(üzüldüm|canım sıkkın|moralim bozuk)\b',
    ],
    'angry': [
        r'\b(kızgın|sinirli|öfkeli|yeter|tamam|peki)\b',
        r'[😡😤🤬👎💢]',
        r'\b(sinir oldum|çileden çıktım|delirdim)\b',
    ],
    'surprised': [
        r'\b(şaşırdım|inanamıyorum|vay be|yok artık)\b',
        r'[😮🤯😲😳]',
        r'\b(şaşırtıcı|beklenmedik|inanılmaz)\b',
    ],
    'anxious': [
        r'\b(endişeli|kaygılı|stresli|gergin)\b',
        r'[😰😬😟]',
        r'\b(korkuyorum|endişeleniyorum|stres altındayım)\b',
    ],
    'neutral': [],
}

# === ETKİLEŞİM TİPİ PATERNLERİ ===
INTERACTION_PATTERNS = {
    'question': [
        r'\?$',
        r'\b(nasıl|neden|nerede|ne zaman|kim|hangi|kaç|mı|mi|mu|mü)\b.*\?',
        r'\b(söyle|anlat|açıkla|öğrenmek istiyorum)\b',
    ],
    'request': [
        r'\b(yap|getir|gönder|bul|ara|kontrol et|bak)\b',
        r'\b(rica ederim|lütfen|yapabilir misin)\b',
        r'\b(ihtiyacım var|lazım|gerek)\b',
    ],
    'compliment': [
        r'\b(teşekkür|sağol|harikasın|süpersin|bravo|helal|aferin)\b',
        r'\b(müthiş|bayıldım|çok güzel|perfect)\b',
        r'[👏🙌❤️🔥]',
    ],
    'complaint': [
        r'\b(şikayet|kızgın|sinir|rahatsız|kötü)\b',
        r'\b(olmuyor|çalışmıyor|hata|yanlış)\b',
        r'[😡😤👎]',
    ],
    'joke': [
        r'\b(haha|hehe|lol|komik|espri|şaka)\b',
        r'[😂🤣😆]',
        r'\b(güldüm|kahkaha|çok komik)\b',
    ],
    'statement': [
        r'\b(düşünüyorum|sanırım|bence|göre)\b',
        r'\b(oldu|yaptım|gittim|gördüm)\b',
    ],
}


def _detect_emotion(text: str) -> str:
    """Hızlı duygu tespiti - regex tabanlı."""
    text_lower = text.lower()
    
    scores = {}
    for emotion, patterns in EMOTION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        scores[emotion] = score
    
    # En yüksek skor
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return 'neutral'
    return best


def _detect_interaction_type(text: str) -> str:
    """Hızlı etkileşim tipi tespiti."""
    text_lower = text.lower()
    
    scores = {}
    for itype, patterns in INTERACTION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        scores[itype] = score
    
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return 'statement'
    return best


def _detect_situation(emotion: str, interaction_type: str) -> str:
    """Duygu + etkileşim tipinden situation çıkar."""
    mapping = {
        ('happy', 'compliment'): 'shared_joy',
        ('happy', 'joke'): 'humor_moment',
        ('happy', 'statement'): 'shared_joy',
        ('sad', 'statement'): 'shared_stress',
        ('sad', 'request'): 'shared_stress',
        ('angry', 'complaint'): 'conflict_moment',
        ('angry', 'statement'): 'conflict_moment',
        ('neutral', 'question'): 'trust_moment',
        ('neutral', 'statement'): 'boring_moment',
        ('surprised', 'statement'): 'shared_joy',
        ('anxious', 'statement'): 'shared_stress',
        ('anxious', 'request'): 'shared_stress',
    }
    return mapping.get((emotion, interaction_type), 'trust_moment')


def _update_personality_trait(interaction_type: str, emotion: str) -> Dict[str, float]:
    """
    Hızlı trait güncellemesi - pattern matching ile.
    Ağır hesaplama yok, sadece küçük delta güncellemeleri.
    
    Returns: Güncellenen trait'ler ve değişim miktarları
    """
    # Etkileşim tipine göre trait ağırlıkları
    trait_deltas = {
        'question': {'curiosity': 0.01, 'analytical': 0.005},
        'statement': {'warmth': 0.005, 'adaptive': 0.005},
        'request': {'warmth': 0.01, 'loyal': 0.005},
        'compliment': {'warmth': 0.015, 'humor': 0.01},
        'complaint': {'assertiveness': 0.01, 'analytical': 0.01},
        'joke': {'humor': 0.015, 'warmth': 0.01},
    }
    
    # Duygu modifikatörü
    emotion_modifier = {
        'happy': 1.2,
        'sad': 0.8,
        'angry': 0.6,
        'surprised': 1.0,
        'anxious': 0.7,
        'neutral': 1.0,
    }
    
    deltas = trait_deltas.get(interaction_type, {'warmth': 0.005})
    modifier = emotion_modifier.get(emotion, 1.0)
    
    result = {}
    for trait, delta in deltas.items():
        result[trait] = round(delta * modifier, 4)
    
    return result


# === ANA FONKSİYON ===

def on_message(text: str, sender_id: str = "unknown") -> Dict:
    """
    Her gelen mesajda çağrılır. HIZLI olmalı (<100ms).
    
    Args:
        text: Mesaj içeriği
        sender_id: Gönderen kimliği
        
    Returns:
        Gözlem özeti dict
    """
    start_time = datetime.now()
    
    # 1. Duygu tespiti
    emotion = _detect_emotion(text)
    
    # 2. Etkileşim tipi tespiti
    interaction_type = _detect_interaction_type(text)
    
    # 3. Situation çıkarımı
    situation = _detect_situation(emotion, interaction_type)
    
    # 4. Negatif sinyal tespiti (social trainer ile)
    negative_signal = None
    trainer = _get_social_trainer()
    if trainer:
        try:
            signal = trainer.detect_and_learn_from_message(text)
            if signal:
                negative_signal = {
                    'signal': signal.get('signal', 'unknown'),
                    'severity': signal.get('severity', 0),
                    'trigger': signal.get('trigger', ''),
                }
        except Exception as e:
            logger.debug(f"Negative signal detection error: {e}")
    
    # 5. Trait güncellemeleri
    trait_updates = _update_personality_trait(interaction_type, emotion)
    
    # Sonuç
    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'sender_id': sender_id,
        'text_preview': text[:80] + ('...' if len(text) > 80 else ''),
        'emotion': emotion,
        'interaction_type': interaction_type,
        'situation': situation,
        'negative_signal': negative_signal,
        'trait_updates': trait_updates,
        'processing_ms': round(elapsed_ms, 1),
    }
    
    # 6. Log'a kaydet (circular, son 100)
    _save_to_log(result)
    
    return result


# === LOG YÖNETİMİ ===

_WATCHER_LOG_FILE = _MODULE_DIR.parent / "cognitive_state" / "watcher_log.json"
_MAX_LOG_ENTRIES = 100


def _save_to_log(entry: Dict):
    """Gözlem sonucunu log'a kaydet (circular buffer)."""
    try:
        _WATCHER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        log = []
        if _WATCHER_LOG_FILE.exists():
            try:
                with open(_WATCHER_LOG_FILE) as f:
                    log = json.load(f)
            except (json.JSONDecodeError, Exception):
                log = []
        
        log.append(entry)
        
        # Circular: son 100
        if len(log) > _MAX_LOG_ENTRIES:
            log = log[-_MAX_LOG_ENTRIES:]
        
        with open(_WATCHER_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Watcher log kaydetme hatası: {e}")


def get_watcher_stats() -> Dict:
    """
    Watcher istatistiklerini döndür.
    
    Returns:
        İstatistik dict
    """
    if not _WATCHER_LOG_FILE.exists():
        return {
            'total_messages': 0,
            'emotions': {},
            'interaction_types': {},
            'negative_signals': 0,
            'avg_processing_ms': 0,
        }
    
    try:
        with open(_WATCHER_LOG_FILE) as f:
            log = json.load(f)
    except Exception:
        return {'total_messages': 0, 'error': 'log_read_failed'}
    
    if not log:
        return {'total_messages': 0}
    
    # İstatistikleri hesapla
    emotions = {}
    interaction_types = {}
    negative_count = 0
    total_ms = 0
    situations = {}
    
    for entry in log:
        # Duygu dağılımı
        emo = entry.get('emotion', 'unknown')
        emotions[emo] = emotions.get(emo, 0) + 1
        
        # Etkileşim tipi dağılımı
        itype = entry.get('interaction_type', 'unknown')
        interaction_types[itype] = interaction_types.get(itype, 0) + 1
        
        # Situation dağılımı
        sit = entry.get('situation', 'unknown')
        situations[sit] = situations.get(sit, 0) + 1
        
        # Negatif sinyal sayısı
        if entry.get('negative_signal'):
            negative_count += 1
        
        # İşlem süresi
        total_ms += entry.get('processing_ms', 0)
    
    return {
        'total_messages': len(log),
        'emotions': emotions,
        'interaction_types': interaction_types,
        'situations': situations,
        'negative_signals': negative_count,
        'negative_signal_rate': round(negative_count / len(log), 3) if log else 0,
        'avg_processing_ms': round(total_ms / len(log), 1) if log else 0,
        'first_entry': log[0].get('timestamp', 'unknown') if log else None,
        'last_entry': log[-1].get('timestamp', 'unknown') if log else None,
    }


# === CLI TEST ===

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if len(sys.argv) > 1:
        # Tek mesaj test
        text = ' '.join(sys.argv[1:])
        result = on_message(text, sender_id="test")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Batch test
        test_messages = [
            "Merhaba Hacı, nasılsın?",
            "Harika bir iş çıkardın! Tebrikler! 👏",
            "Bu ne saçma şey ya, yeter artık 😡",
            "Haha çok komik 😂",
            "Yarın hava nasıl olacak?",
            "Lütfen şu dosyayı gönder",
            "Üzgünüm, bugün kötü bir gün geçirdim",
            "Vay be, inanamıyorum! 🤯",
        ]
        
        print("=== Cognitive Watcher Test ===\n")
        for msg in test_messages:
            result = on_message(msg, sender_id="test")
            print(f"📨 \"{result['text_preview']}\"")
            print(f"   Duygu: {result['emotion']} | Tip: {result['interaction_type']} | Durum: {result['situation']}")
            if result['negative_signal']:
                print(f"   ⚠️ Negatif sinyal: {result['negative_signal']['signal']}")
            print(f"   ⚡ {result['processing_ms']}ms\n")
        
        print("\n📊 İstatistikler:")
        stats = get_watcher_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

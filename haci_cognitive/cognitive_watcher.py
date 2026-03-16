"""
Cognitive Watcher v2 - Mesaj analiz motoru (iyileştirilmiş)

Yenilikler:
- Emotion intensity (low/medium/high/extreme)
- Sarkazm/ironi algılama
- Türkçe internet dili desteği (kralsın, reis, lan, ulan vs)
- Multi-emotion support (karmaşık duygular)
- Ağırlıklı scoring (pattern quality > quantity)
- Context-aware situation detection (son 3 mesaj)
- Humor subtype detection (sarcasm, self-deprecating, dark)
- Turkish-specific emotional expressions

ÖNEMLİ: <100ms çalışmalı - ağır işlem yok
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
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
            logger.debug(f"Social trainer yüklenemedi: {e}")
    return _social_trainer


# === INTENSITY LEVELS ===
INTENSITY_LEVELS = ['low', 'medium', 'high', 'extreme']

# Intensity multiplier for scoring
INTENSITY_MULTIPLIER = {
    'low': 1.0,
    'medium': 1.5,
    'high': 2.0,
    'extreme': 3.0,
}


# === EMOJI INTENSITY MAP ===
EMOJI_INTENSITY = {
    'low': '😊🙂👍',
    'medium': '😄❤️🔥💪',
    'high': '😍🤩👏🙌',
    'extreme': '🤯💥🎆👑',
}


def _get_emoji_intensity(emoji_char: str) -> str:
    """Bir emoji'nin intensity level'ını döndür."""
    for level, chars in EMOJI_INTENSITY.items():
        if emoji_char in chars:
            return level
    return 'medium'


# === DUYGU PATERNLERİ (gelişmiş - ağırlıklı, intensity'li) ===
# Format: (regex_pattern, weight, default_intensity)

EMOTION_PATTERNS = {
    'happy': [
        # Low intensity
        (r'\b(güzel|iyi|tamam|olur|hoş|sevdim)\b', 1.0, 'low'),
        (r'\b(güldüm|gülüyorum)\b', 1.2, 'low'),
        # Medium intensity
        (r'\b(harika|süper|muhteşem|mükemmel)\b', 2.0, 'medium'),
        (r'\b(mutlu|sevinçli|keyifli)\b', 1.8, 'medium'),
        (r'\b(tamamdır|helal|aferin)\b', 1.5, 'medium'),
        # High intensity
        (r'\b(kralsın|reissin|efsane|şahane|muazzam)\b', 3.0, 'high'),
        (r'\b(taşşına|ballı|manyak|adam|kaptırmış)\b', 2.5, 'high'),
        (r'\b(taşşaklı|taşaklı)\b', 3.0, 'high'),
        (r'[😊😄😃🥰😍❤️🎉💯🔥]', 1.5, 'medium'),
        # Extreme
        (r'\b(delirdim|çığlık|patladım|yaşıyorum)\b', 3.5, 'extreme'),
        (r'\b(ölüyorum|bayıldım|oldum)\b.*\b(gülme|mutlu|sevinç)', 3.0, 'extreme'),
    ],
    'sad': [
        (r'\b(üzgün|kötü|mutsuz|canım sıkkın)\b', 1.5, 'medium'),
        (r'\b(üzüldüm|üzüldüm|moralsiz)\b', 1.8, 'medium'),
        (r'\b(berbat|facia|felaket|bittim|tükendim)\b', 2.5, 'high'),
        (r'[😢😭😞😔💔🥺]', 1.5, 'medium'),
        (r'\b(ölüm|öl|vazgeçtim|yapamıyorum)\b.*\b(üzgün|keder)', 3.0, 'extreme'),
    ],
    'angry': [
        (r'\b(sinir|kızgın|rahatsız)\b', 1.5, 'medium'),
        (r'\b(lan|ulan|yeter|bıktım|sıkıldım)\b', 1.8, 'medium'),
        (r'\b(öfkeli|delirdim|çıldırdım|patladım)\b', 2.5, 'high'),
        (r'[😡😤🤬👎💢]', 1.5, 'medium'),
        (r'\b(siktir|amk|mk|amına|göt)\b', 3.0, 'high'),
        (r'\b(rezil|alçak|şerefsiz)\b', 3.5, 'extreme'),
    ],
    'surprised': [
        (r'\b(şaşırdım|ilginç|öyle mi)\b', 1.2, 'low'),
        (r'\b(vay be|yok artık|inanılmaz)\b', 2.0, 'medium'),
        (r'\b(inanamıyorum|hayret|şok oldum)\b', 2.5, 'high'),
        (r'[😮🤯😲😳]', 1.5, 'medium'),
        (r'\b(çığlık|bayılacağım|deliriyorum)\b.*\b(şaşırtıcı|beklenmedik)', 3.0, 'extreme'),
    ],
    'anxious': [
        (r'\b(endişe|kaygı|stres|gergin)\b', 1.5, 'medium'),
        (r'\b(korku|panik|panik atak)\b', 2.5, 'high'),
        (r'[😰😬😟]', 1.5, 'medium'),
        (r'\b(öleceğim|battı|kayboldum|ne yapacağım)\b', 3.0, 'extreme'),
    ],
    'sarcastic': [
        # Sarkazm pattern'leri
        (r'\b(tabii|tabi canım|evet evet|kesinlikle|he ya|aynen)\b', 2.0, 'medium'),
        (r'\b(maşallah|hayırlı|bereketli|güzel günler)\b.*\b(geldi|geçti|olur)', 2.0, 'medium'),
        (r'\b(bravo|hakikaten|harika|süper)\b.*\b(yapmışsın|olmuş|gitti)', 2.5, 'high'),
        (r'\b(nasıl yani|olabilir|olur öyle|sıkıntı yok)\b.*\b(?!😄|!|❤️)', 1.5, 'low'),
    ],
    'neutral': [],
}


# === ETKİLEŞİM TİPİ PATERNLERİ (gelişmiş) ===
INTERACTION_PATTERNS = {
    'question': [
        (r'\?$', 2.0),
        (r'\b(nasıl|neden|nerede|ne zaman|kim|hangi|kaç)\b', 1.5),
        (r'\b(mı|mi|mu|mü)\b.*\?$', 2.0),
        (r'\b(söyle|anlat|açıkla|öğrenmek istiyorum|merak ediyorum)\b', 1.5),
        (r'\b(nedir|nasıldır|var mı|yok mu|olur mu)\b', 2.0),
    ],
    'request': [
        (r'\b(yap|getir|gönder|bul|ara|kontrol et|bak)\b', 1.5),
        (r'\b(rica ederim|lütfen|yapabilir misin|olur mu)\b', 2.0),
        (r'\b(ihtiyacım var|lazım|gerek|istiyorum)\b', 1.5),
        (r'\b(kur|yükle|başlat|durdur|sil|ekle)\b', 1.8),
    ],
    'compliment': [
        (r'\b(teşekkür|sağol|teşekkürler|sağolasın)\b', 2.5),
        (r'\b(harikasın|süpersin|bravo|helal|aferin|muhteşemsin)\b', 3.0),
        (r'\b(kralsın|reissin|efsanesin|adamın dibisin)\b', 3.5),
        (r'\b(müthiş|bayıldım|çok güzel|perfect|mükemmel)\b', 2.0),
        (r'[👏🙌❤️🔥👑]', 1.5),
    ],
    'complaint': [
        (r'\b(şikayet|rahatsız|kötü|berbat)\b', 2.0),
        (r'\b(olmuyor|çalışmıyor|hata|yanlış|bozuk)\b', 2.5),
        (r'\b(yapma|kes|bırak|sıkıldım|bıktım)\b', 2.0),
        (r'[😡😤👎]', 1.5),
    ],
    'joke': [
        (r'\b(haha|hehe|komik|espri|şaka)\b', 2.0),
        (r'\b(güldüm|kahkaha|ölüm güldüm|bayıldım gülmekten)\b', 2.5),
        (r'\b(rezil|skandal|tarihi)\b.*\b(gülmek|komik)', 2.0),
        (r'[😂🤣😆😹💀]', 2.0),
    ],
    'celebration': [
        (r'\b(kutlama|zafer|başarı|kazandık|şampiyon)\b', 2.5),
        (r'\b(gol|attık|bitirdik|destan)\b', 2.0),
        (r'[🏆🎊🎈🎇🎆🦁💛❤️]', 2.0),
    ],
    'statement': [
        (r'\b(düşünüyorum|sanırım|bence|göre|derim)\b', 1.0),
        (r'\b(oldu|yaptım|gittim|gördüm|aldım)\b', 1.0),
        (r'\b(anlaştık|tamam|ok|peki|he)\b', 1.2),
    ],
}


# === SARKAZM DETECTION ===

SARCASM_INDICATORS = [
    # Pattern + confidence boost
    (r'\b(tabii|tabi)\b.*\b(canım|yavrum|kardeşim)\b', 0.8),
    (r'\b(evvet|eevet|eyt)\b', 0.6),
    (r'\b(hakikaten|hakketen)\b.*\b(mi|mı)\b.*\?', 0.5),
    (r'🙄', 0.9),
    (r'\b(nasıl da|ne kadar)\b.*\b(güzel|harika|mükemmel)\b', 0.7),
    (r'\b(yaşasın|oh be|ne güzel)\b.*\b(!|\.|…)', 0.4),
    (r'…$', 0.3),  # Suspended dots
    (r'\.{3,}$', 0.3),  # Ellipsis
]


def _detect_sarcasm(text: str) -> Tuple[bool, float]:
    """
    Sarkazm/ironi tespiti.
    Returns: (is_sarcastic, confidence)
    """
    text_lower = text.lower()
    total_confidence = 0.0
    match_count = 0

    for pattern, confidence in SARCASM_INDICATORS:
        if re.search(pattern, text_lower):
            total_confidence += confidence
            match_count += 1

    if match_count == 0:
        return False, 0.0

    # Multiple indicators boost confidence
    combined = min(total_confidence * (1 + 0.2 * (match_count - 1)), 1.0)
    return combined > 0.5, round(combined, 2)


# === MULTI-EMOTION DETECTION ===

def _detect_emotions(text: str) -> List[Dict[str, any]]:
    """
    Birden fazla duygu tespiti.
    Returns: [{'emotion': str, 'score': float, 'intensity': str}, ...]
    """
    text_lower = text.lower()
    emotion_scores = []

    for emotion, patterns in EMOTION_PATTERNS.items():
        if emotion == 'neutral':
            continue

        total_score = 0.0
        max_intensity = 'low'

        for pattern, weight, default_intensity in patterns:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                total_score += matches * weight
                # Track highest intensity
                idx = INTENSITY_LEVELS.index(default_intensity)
                cur_idx = INTENSITY_LEVELS.index(max_intensity)
                if idx > cur_idx:
                    max_intensity = default_intensity

        if total_score > 0:
            emotion_scores.append({
                'emotion': emotion,
                'score': round(total_score, 2),
                'intensity': max_intensity,
            })

    # Sort by score descending
    emotion_scores.sort(key=lambda x: x['score'], reverse=True)

    if not emotion_scores:
        return [{'emotion': 'neutral', 'score': 0.0, 'intensity': 'low'}]

    # Keep top 3 emotions (multi-emotion support)
    return emotion_scores[:3]


def _detect_emotion(text: str) -> str:
    """Backward compatibility: primary emotion string."""
    emotions = _detect_emotions(text)
    return emotions[0]['emotion']


def _detect_interaction_type(text: str) -> Tuple[str, float]:
    """
    Etkileşim tipi tespiti (ağırlıklı).
    Returns: (type, confidence)
    """
    text_lower = text.lower()
    type_scores = {}

    for itype, patterns in INTERACTION_PATTERNS.items():
        score = 0.0
        for pattern, weight in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches * weight
        if score > 0:
            type_scores[itype] = score

    if not type_scores:
        return 'statement', 0.5

    best = max(type_scores, key=type_scores.get)
    total = sum(type_scores.values())
    confidence = round(type_scores[best] / total, 2) if total > 0 else 0.5

    return best, confidence


def _detect_situation(emotion: str, interaction_type: str, is_sarcastic: bool) -> str:
    """Duygu + etkileşim tipi + sarkazm → situation."""
    if is_sarcastic:
        sarcasm_situations = {
            ('angry', 'statement'): 'passive_aggressive',
            ('angry', 'compliment'): 'passive_aggressive',
            ('happy', 'complaint'): 'ironic_humor',
            ('neutral', 'compliment'): 'backhanded_compliment',
        }
        if (emotion, interaction_type) in sarcasm_situations:
            return sarcasm_situations[(emotion, interaction_type)]
        return 'sarcastic_moment'

    mapping = {
        ('happy', 'compliment'): 'shared_joy',
        ('happy', 'joke'): 'humor_moment',
        ('happy', 'celebration'): 'celebration_moment',
        ('happy', 'statement'): 'shared_joy',
        ('sad', 'statement'): 'shared_stress',
        ('sad', 'request'): 'support_needed',
        ('angry', 'complaint'): 'conflict_moment',
        ('angry', 'statement'): 'tension_moment',
        ('neutral', 'question'): 'information_exchange',
        ('neutral', 'statement'): 'casual_chat',
        ('surprised', 'statement'): 'discovery_moment',
        ('anxious', 'statement'): 'anxiety_moment',
        ('anxious', 'request'): 'support_needed',
    }
    return mapping.get((emotion, interaction_type), 'trust_moment')


def _update_personality_trait(interaction_type: str, emotion: str,
                              intensity: str, is_sarcastic: bool) -> Dict[str, float]:
    """Gelişmiş trait güncellemesi (intensity + sarcasm aware)."""
    # Base trait deltas
    trait_deltas = {
        'question': {'curiosity': 0.01, 'analytical': 0.005},
        'statement': {'warmth': 0.005, 'adaptive': 0.005},
        'request': {'warmth': 0.01, 'loyal': 0.005},
        'compliment': {'warmth': 0.015, 'humor': 0.01},
        'complaint': {'assertiveness': 0.01, 'analytical': 0.01},
        'joke': {'humor': 0.015, 'warmth': 0.01},
        'celebration': {'warmth': 0.02, 'loyal': 0.015},
    }

    # Emotion modifier
    emotion_modifier = {
        'happy': 1.2, 'sad': 0.8, 'angry': 0.6,
        'surprised': 1.0, 'anxious': 0.7, 'sarcastic': 0.5,
        'neutral': 1.0,
    }

    # Intensity modifier
    int_modifier = INTENSITY_MULTIPLIER.get(intensity, 1.0)

    # Sarcasm penalty
    sarcasm_modifier = 0.3 if is_sarcastic else 1.0

    deltas = trait_deltas.get(interaction_type, {'warmth': 0.005})
    e_modifier = emotion_modifier.get(emotion, 1.0)

    result = {}
    for trait, delta in deltas.items():
        final = delta * e_modifier * int_modifier * sarcasm_modifier
        result[trait] = round(final, 4)

    return result


# === CONTEXT AWARENESS ===

_recent_messages = []  # Circular buffer son 5 mesaj


def _update_context(text: str, emotion: str, interaction_type: str):
    """Context buffer güncelle."""
    global _recent_messages
    _recent_messages.append({
        'text': text[:100],
        'emotion': emotion,
        'interaction_type': interaction_type,
        'timestamp': datetime.now().isoformat(),
    })
    # Keep last 5
    if len(_recent_messages) > 5:
        _recent_messages = _recent_messages[-5:]


def _get_context_trend() -> Dict:
    """Son mesajlardan trend çıkar."""
    if not _recent_messages:
        return {'trend': 'stable', 'emotion_shift': None}

    emotions = [m['emotion'] for m in _recent_messages]
    last_emotion = emotions[-1] if emotions else 'neutral'

    if len(emotions) >= 3:
        recent = emotions[-3:]
        # Check for negative trend
        negative_emotions = {'sad', 'angry', 'anxious', 'sarcastic'}
        negative_count = sum(1 for e in recent if e in negative_emotions)
        if negative_count >= 2:
            return {'trend': 'negative_escalation', 'emotion_shift': last_emotion}
        # Check for positive trend
        positive_count = sum(1 for e in recent if e in {'happy', 'surprised'})
        if positive_count >= 2:
            return {'trend': 'positive_momentum', 'emotion_shift': last_emotion}

    return {'trend': 'stable', 'emotion_shift': last_emotion}


# === ANA FONKSİYON ===

# === SENDER PROFILING ===

_SENDER_PROFILES_FILE = _MODULE_DIR.parent / "cognitive_state" / "sender_profiles.json"
_sender_profiles_cache = None


def _load_sender_profiles() -> Dict:
    """Sender profillerini yükle."""
    global _sender_profiles_cache
    if _sender_profiles_cache is not None:
        return _sender_profiles_cache
    try:
        if _SENDER_PROFILES_FILE.exists():
            with open(_SENDER_PROFILES_FILE) as f:
                _sender_profiles_cache = json.load(f)
        else:
            _sender_profiles_cache = {}
    except Exception:
        _sender_profiles_cache = {}
    return _sender_profiles_cache


def _save_sender_profiles():
    """Sender profillerini kaydet."""
    global _sender_profiles_cache
    if _sender_profiles_cache is None:
        return
    try:
        _SENDER_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SENDER_PROFILES_FILE, 'w') as f:
            json.dump(_sender_profiles_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Sender profile save error: {e}")


def _update_sender_profile(sender_id: str, emotion: str, intensity: str,
                           interaction_type: str, text: str, emoji_count: int):
    """Sender profilini güncelle."""
    profiles = _load_sender_profiles()
    
    if sender_id not in profiles:
        profiles[sender_id] = {
            'message_count': 0,
            'emotion_distribution': {},
            'intensity_distribution': {},
            'interaction_distribution': {},
            'avg_emoji_per_msg': 0,
            'total_emojis': 0,
            'avg_msg_length': 0,
            'total_chars': 0,
            'first_seen': datetime.now().isoformat(),
            'last_seen': None,
            'emotional_baseline': 'neutral',
            'dominant_emotion': None,
            'is_expressive': False,
        }
    
    p = profiles[sender_id]
    p['message_count'] += 1
    p['last_seen'] = datetime.now().isoformat()
    
    # Emotion distribution
    p['emotion_distribution'][emotion] = p['emotion_distribution'].get(emotion, 0) + 1
    p['dominant_emotion'] = max(p['emotion_distribution'], key=p['emotion_distribution'].get)
    
    # Intensity distribution
    p['intensity_distribution'][intensity] = p['intensity_distribution'].get(intensity, 0) + 1
    
    # Interaction distribution
    p['interaction_distribution'][interaction_type] = p['interaction_distribution'].get(interaction_type, 0) + 1
    
    # Emoji stats
    p['total_emojis'] += emoji_count
    p['avg_emoji_per_msg'] = round(p['total_emojis'] / p['message_count'], 2)
    
    # Message length
    p['total_chars'] += len(text)
    p['avg_msg_length'] = round(p['total_chars'] / p['message_count'], 0)
    
    # Emotional baseline (en sık görülen emotion)
    if p['message_count'] >= 3:
        p['emotional_baseline'] = p['dominant_emotion']
    
    # Expressiveness (high intensity + emojis)
    high_intensity_pct = (p['intensity_distribution'].get('high', 0) + 
                          p['intensity_distribution'].get('extreme', 0)) / p['message_count']
    p['is_expressive'] = high_intensity_pct > 0.3 or p['avg_emoji_per_msg'] > 1.5
    
    # 100 mesajda bir kaydet (performans)
    if p['message_count'] % 10 == 0:
        _save_sender_profiles()


def _get_sender_deviation(sender_id: str, current_emotion: str, current_intensity: str) -> Dict:
    """Sender'ın mevcut duygu durumunun baseline'dan sapması."""
    profiles = _load_sender_profiles()
    
    if sender_id not in profiles or profiles[sender_id]['message_count'] < 3:
        return {'has_profile': False, 'deviation': 'none', 'significance': 0}
    
    p = profiles[sender_id]
    baseline = p['emotional_baseline']
    
    # Emotion valence map (positive/negative/neutral)
    valence = {
        'happy': 1, 'excited': 1, 'grateful': 1, 'proud': 1, 'amused': 1,
        'sad': -1, 'angry': -1, 'anxious': -1, 'frustrated': -1,
        'surprised': 0, 'neutral': 0, 'confused': 0,
    }
    
    baseline_val = valence.get(baseline, 0)
    current_val = valence.get(current_emotion, 0)
    
    diff = current_val - baseline_val
    
    if diff > 0:
        deviation = 'more_positive'
    elif diff < 0:
        deviation = 'more_negative'
    else:
        deviation = 'on_baseline'
    
    # Intensity deviation
    intensity_order = {'low': 0, 'medium': 1, 'high': 2, 'extreme': 3}
    avg_intensity_score = sum(
        intensity_order.get(k, 1) * v 
        for k, v in p['intensity_distribution'].items()
    ) / max(p['message_count'], 1)
    
    current_intensity_score = intensity_order.get(current_intensity, 1)
    intensity_diff = current_intensity_score - avg_intensity_score
    
    significance = abs(diff) + abs(intensity_diff) * 0.5
    
    return {
        'has_profile': True,
        'baseline_emotion': baseline,
        'deviation': deviation,
        'intensity_shift': 'higher' if intensity_diff > 0.5 else ('lower' if intensity_diff < -0.5 else 'normal'),
        'significance': round(significance, 2),
        'message_count': p['message_count'],
    }


def on_message(text: str, sender_id: str = "unknown") -> Dict:
    """
    Her gelen mesajda çağrılır. HIZLI olmalı (<100ms).

    Args:
        text: Mesaj içeriği
        sender_id: Gönderen kimliği

    Returns:
        Gözlem özeti dict (v2 format)
    """
    start_time = datetime.now()

    # 1. Multi-emotion detection
    emotions = _detect_emotions(text)
    primary_emotion = emotions[0]['emotion']
    primary_intensity = emotions[0]['intensity']

    # 2. Sarkazm detection
    is_sarcastic, sarcasm_confidence = _detect_sarcasm(text)

    # 3. Interaction type
    interaction_type, interaction_confidence = _detect_interaction_type(text)

    # 4. Situation (sarcasm-aware)
    situation = _detect_situation(primary_emotion, interaction_type, is_sarcastic)

    # 5. Context trend
    _update_context(text, primary_emotion, interaction_type)
    context_trend = _get_context_trend()

    # 6. Negative signal (social trainer)
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

    # 7. Sender profiling
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]', text))
    _update_sender_profile(sender_id, primary_emotion, primary_intensity,
                           interaction_type, text, emoji_count)
    sender_deviation = _get_sender_deviation(sender_id, primary_emotion, primary_intensity)

    # 8. Trait updates (intensity + sarcasm aware)
    trait_updates = _update_personality_trait(
        interaction_type, primary_emotion, primary_intensity, is_sarcastic
    )

    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

    result = {
        'timestamp': datetime.now().isoformat(),
        'sender_id': sender_id,
        'text_preview': text[:80] + ('...' if len(text) > 80 else ''),
        # V2 emotion format
        'emotion': primary_emotion,
        'emotion_intensity': primary_intensity,
        'all_emotions': emotions,
        # Sarkazm
        'is_sarcastic': is_sarcastic,
        'sarcasm_confidence': sarcasm_confidence,
        # Interaction
        'interaction_type': interaction_type,
        'interaction_confidence': interaction_confidence,
        # Situation
        'situation': situation,
        # Context
        'context_trend': context_trend,
        # Other
        'negative_signal': negative_signal,
        'sender_profile': sender_deviation,
        'trait_updates': trait_updates,
        'processing_ms': round(elapsed_ms, 1),
        'version': '2.0',
    }

    # 8. Log
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
        if len(log) > _MAX_LOG_ENTRIES:
            log = log[-_MAX_LOG_ENTRIES:]
        with open(_WATCHER_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Watcher log kaydetme hatası: {e}")


def get_watcher_stats() -> Dict:
    """Watcher istatistikleri (v2 enhanced)."""
    if not _WATCHER_LOG_FILE.exists():
        return {'total_messages': 0, 'version': '2.0'}

    try:
        with open(_WATCHER_LOG_FILE) as f:
            log = json.load(f)
    except Exception:
        return {'total_messages': 0, 'error': 'log_read_failed'}

    if not log:
        return {'total_messages': 0, 'version': '2.0'}

    emotions = {}
    intensities = {}
    interaction_types = {}
    situations = {}
    sarcasm_count = 0
    negative_count = 0
    total_ms = 0

    for entry in log:
        emo = entry.get('emotion', 'unknown')
        emotions[emo] = emotions.get(emo, 0) + 1

        inten = entry.get('emotion_intensity', 'unknown')
        intensities[inten] = intensities.get(inten, 0) + 1

        itype = entry.get('interaction_type', 'unknown')
        interaction_types[itype] = interaction_types.get(itype, 0) + 1

        sit = entry.get('situation', 'unknown')
        situations[sit] = situations.get(sit, 0) + 1

        if entry.get('is_sarcastic'):
            sarcasm_count += 1

        if entry.get('negative_signal'):
            negative_count += 1

        total_ms += entry.get('processing_ms', 0)

    return {
        'total_messages': len(log),
        'version': '2.0',
        'emotions': emotions,
        'intensities': intensities,
        'interaction_types': interaction_types,
        'situations': situations,
        'sarcasm_count': sarcasm_count,
        'sarcasm_rate': round(sarcasm_count / len(log), 3),
        'negative_signals': negative_count,
        'negative_signal_rate': round(negative_count / len(log), 3),
        'avg_processing_ms': round(total_ms / len(log), 1),
        'first_entry': log[0].get('timestamp', 'unknown'),
        'last_entry': log[-1].get('timestamp', 'unknown'),
    }


# === CLI TEST ===

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result = on_message(text, sender_id="test")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        test_messages = [
            "Merhaba Hacı, nasılsın?",
            "Harika bir iş çıkardın! Tebrikler! Kralsın 👏",
            "Bu ne saçma şey ya, yeter artık 😡",
            "Haha çok komik, koptum 😂",
            "Yarın hava nasıl olacak?",
            "Lütfen şu dosyayı gönder",
            "Üzgünüm, bugün kötü bir gün geçirdim",
            "Vay be, inanamıyorum! 🤯",
            "Tabii canım, kesinlikle öyledir...",
            "Ulan ne rezil bir maç ya 😤",
            "Başkan kaptırmış abi hahahaha 🦁",
            "Eyvallah sağol, çok yardımcı oldun",
        ]

        print("=== Cognitive Watcher v2 Test ===\n")
        for msg in test_messages:
            result = on_message(msg, sender_id="test")
            print(f"📨 \"{result['text_preview']}\"")
            emotions_str = ', '.join(
                f"{e['emotion']}({e['intensity']})" for e in result['all_emotions']
            )
            print(f"   Duygu: {emotions_str}")
            print(f"   Tip: {result['interaction_type']} ({result['interaction_confidence']})")
            print(f"   Durum: {result['situation']}")
            if result['is_sarcastic']:
                print(f"   🎭 Sarkazm! (güven: {result['sarcasm_confidence']})")
            if result['context_trend']['trend'] != 'stable':
                print(f"   📈 Trend: {result['context_trend']['trend']}")
            if result['negative_signal']:
                print(f"   ⚠️ Negatif sinyal: {result['negative_signal']['signal']}")
            print(f"   ⚡ {result['processing_ms']}ms\n")

        print("\n📊 İstatistikler:")
        stats = get_watcher_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

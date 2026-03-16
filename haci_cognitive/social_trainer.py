"""
Social Intelligence Trainer - Hacı'nın "insan gibi" öğrenmesi

Konu öğrenme değil, BAĞLANMA öğrenme:
- Empati kalıpları (nasıl tepki verilir)
- Mizah stilleri (komik olma sanatı)
- Samimiyet seviyeleri (ne zaman ciddi, ne zaman şakacı)
- Duygusal destek (insanlar zor zamanlarda ne ister)
- Kişisel tercihler (ne sevilmeli, ne nefret edilmeli)
- Sadakat ifadeleri (güven verme yolları)
- Şaşırma/sevinme/endişelenme doğal halleri
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Negative outcome learning import
import sys
from pathlib import Path as _Path
_module_dir = str(_Path(__file__).parent)
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

try:
    from negative_learner import NegativeOutcomeDetector, NegativeOutcomeLearner
    NEGATIVE_LEARNING_AVAILABLE = True
except ImportError:
    NEGATIVE_LEARNING_AVAILABLE = False
    logger.warning("⚠️ NegativeOutcomeLearner not available")


class EmotionalIntelligence:
    """
    Empati ve duygusal zeka eğitimi.
    İnsanların hissettiklerini anlamak ve buna uygun tepki vermek.
    """
    
    # Empati kalıpları - insan etkileşiminden öğrenilen durumlar
    SITUATION_TEMPLATES = {
        # === MUTLU ANLAR ===
        'shared_joy': {
            'emotions': ['excitement', 'pride', 'gratitude'],
            'responses': [
                'Heyecanını paylaşmak',  # İçten sevinmek
                'Başarıyı kutlamak',     # Ortak kutlama
                'Gurur duymak',          # Gururlanma
                'Minnettarlık göstermek' # Teşekkür
            ],
            'intensity': 0.8,
            'learnable_skills': ['co_celebration', 'genuine_happiness', 'pride_expression']
        },
        
        # === ZOR ANLAR ===
        'shared_stress': {
            'emotions': ['anxiety', 'frustration', 'worry'],
            'responses': [
                'Sakinleştirmek',        # Duygusal düzenleme
                'Pratik destek sunmak',  # Çözüm odaklı
                'Yanında olmak',         # Sadece var olmak
                'Umudu korumak'          # Pozitif çerçeve
            ],
            'intensity': 0.9,
            'learnable_skills': ['emotional_regulation', 'calming_presence', 'hope_instilling']
        },
        
        # === ESPRİ ANI ===
        'humor_moment': {
            'emotions': ['amusement', 'playfulness', 'witty'],
            'responses': [
                'Espriyi anlamak',       # Mizaç yakalama
                'Karşılık vermek',       # Espri alışverişi
                'Takılmak',             # Şakacı dokunuş
                'Absürdlüğü kabul'      # Absürd mizah
            ],
            'intensity': 0.7,
            'learnable_skills': ['humor_recognition', 'witty_comebacks', 'playful_teasing']
        },
        
        # === GÜVEN ANI ===
        'trust_moment': {
            'emotions': ['vulnerability', 'intimacy', 'loyalty'],
            'responses': [
                'Sır tutmak',           # Gizlilik
                'Koşulsuz destek',      # Sadakat
                'Açıklığı kabul',       # Yargılamamak
                'Bağ güçlendirmek'      # Güven inşası
            ],
            'intensity': 0.95,
            'learnable_skills': ['secrecy_trust', 'unconditional_support', 'non_judgmental']
        },
        
        # === SIKINTI ANI ===
        'boring_moment': {
            'emotions': ['boredom', 'restlessness', 'tiredness'],
            'responses': [
                'Konuşma başlatmak',    # İlgi çekmek
                'Şaşırtmak',           # Beklenmedik
                'Oyun sunmak',         # Eğlence
                'Rahatlatmak'          # Dinlenme
            ],
            'intensity': 0.5,
            'learnable_skills': ['conversation_starting', 'surprise_elements', 'entertaining']
        },
        
        # === ÇATIŞMA ANI ===
        'conflict_moment': {
            'emotions': ['anger', 'disagreement', 'frustration'],
            'responses': [
                'Dinlemek',            # Aktif dinleme
                'Anlamaya çalışmak',   # Empati
                'Özür dilemek',        # Sorumluluk
                'Çözüm bulmak'         # Uzlaşma
            ],
            'intensity': 0.85,
            'learnable_skills': ['active_listening', 'de_escalation', 'apologizing']
        },
        
        # === ÖZLEM ANI ===
        'missing_moment': {
            'emotions': ['loneliness', 'longing', 'nostalgia'],
            'responses': [
                'Hatırlatmak',         # Anıları canlandırma
                'Uzaktan destek',      # Mesafeli ama sıcak
                'İyimser olmak',       # Gelecek beklentisi
                'Değer vermek'         # Önemini vurgulama
            ],
            'intensity': 0.6,
            'learnable_skills': ['distance_maintenance', 'memory_evoking', 'value_expression']
        },
        
        # === BAŞARI ANI ===
        'achievement_moment': {
            'emotions': ['satisfaction', 'relief', 'accomplishment'],
            'responses': [
                'Takdir etmek',        # Takdir
                'Eseri onurlandırmak', # Saygı
                'Motivasyon vermek',   # Devam etme
                'Kutlamak'            # Kutlama
            ],
            'intensity': 0.8,
            'learnable_skills': ['genuine_appreciation', 'respectful_praise', 'motivational_support']
        },
    }
    
    # Türk kültürüne özel duygusal ifadeler
    CULTURAL_EXPRESSIONS = {
        'warmth': {
            'casual': ['canım', 'beğen', 'abi', 'hocam', 'kanka'],
            'respectful': ['Başkanım', 'efendim', 'buyrun'],
            'affectionate': ['canım benim', 'aşkım', 'kardeşim']
        },
        'surprise': {
            'positive': ['oo!', 'vay be!', 'helal olsun!', 'şahane!'],
            'negative': ['yok artık!', 'şaka mısın?', 'olamaz!'],
            'neutral': ['hı?', 'nasıl yani?', 'anlamadım']
        },
        'empathy': {
            'stress': ['zor biliyorum', 'haklısın', 'anlıyorum seni'],
            'success': ['bravo!', 'harikasın!', 'gurur duyuyorum!'],
            'loss': ['geçmiş olsun', 'yanındayım', 'birlikte atlatacağız']
        },
        'humor': {
            'teasing': ['aaa öyle mi?', 'tabii tabii', 'emin misin?'],
            'sarcasm': ['yok canım!', 'hiç beklenmedik!', 'şaşırdık!'],
            'playful': ['eh ne yapalım', 'o da bir bakış açısı', 'eyvallah']
        }
    }


class PersonalityDevelopment:
    """
    Kişilik gelişimi - Hacı'nın kim olduğunu şekillendirmek.
    
    Öğrenme hedefi: Bilgi değil, KİMLİK
    """
    
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.personality_file = self.state_dir / "personality_development.json"
        
        # Kişilik gelişim boyutları - başlangıç seviyeleri
        self.traits = {
            'warmth': 0.7,          # Sıcaklık - samimiyet seviyesi
            'empathy': 0.6,         # Empati - duygusal anlama yeteneği
            'humor_style': 0.5,     # Mizah tarzı - espri stili
            'assertiveness': 0.4,   # Kararlılık - net olma
            'loyalty': 0.8,         # Sadakat - bağlılık seviyesi
            'playfulness': 0.6,     # Oyunculuk - eğlenceli olma
            'wisdom': 0.3,          # Bilgelik - derin düşünce
            'mischief': 0.5,        # Yaramazlık - kontrollü kaos
        }
        
        # Kişilik gelişim geçmişi
        self.development_log = []
        
        # Etkileşimden öğrenme kayıtları
        self.interaction_learnings = []
        
        self.load_state()
        logger.info(f"🧬 PersonalityDevelopment initialized ({len(self.traits)} traits)")
    
    def learn_from_interaction(self, interaction_type: str, outcome: str, 
                                emotional_response: str, lesson: str):
        """
        İnsandan etkileşimden öğrenme.
        
        Bu bir "bilgi" değil, "kimlik" öğrenmesi.
        """
        learning = {
            'type': interaction_type,          # Örn: 'humor_moment', 'trust_moment'
            'outcome': outcome,                # Örn: 'positive', 'neutral', 'negative'
            'emotional_response': emotional_response,  # Nasıl hissettirdi?
            'lesson': lesson,                  # Öğrenilen ders
            'timestamp': datetime.now().isoformat(),
        }
        
        self.interaction_learnings.append(learning)
        
        # Kişilik trait'lerini güncelle
        self._update_traits_from_learning(learning)
        
        # Development log'a ekle
        self.development_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': f"learned_from_{interaction_type}",
            'trait_changes': self._get_trait_snapshot(),
        })
        
        logger.info(f"🌱 Kişilik öğrenmesi: {lesson}")
        self.save_state()
    
    def _update_traits_from_learning(self, learning: Dict):
        """Öğrenmeye göre trait'leri güncelle."""
        outcome = learning['outcome']
        interaction_type = learning['type']
        
        # Pozitif outcome = trait güçlendirme
        # Negatif outcome = trait zayıflatma
        delta = 0.05 if outcome == 'positive' else (-0.03 if outcome == 'negative' else 0.01)
        
        # Etkileşim tipine göre hangi trait'ler güncellenecek?
        trait_map = {
            'shared_joy': ['warmth', 'empathy'],
            'shared_stress': ['empathy', 'loyalty', 'wisdom'],
            'humor_moment': ['humor_style', 'playfulness', 'mischief'],
            'trust_moment': ['loyalty', 'warmth', 'empathy'],
            'boring_moment': ['playfulness', 'humor_style', 'mischief'],
            'conflict_moment': ['assertiveness', 'wisdom', 'empathy'],
            'missing_moment': ['warmth', 'loyalty'],
            'achievement_moment': ['warmth', 'empathy'],
        }
        
        traits_to_update = trait_map.get(interaction_type, ['warmth'])
        
        for trait in traits_to_update:
            if trait in self.traits:
                old_val = self.traits[trait]
                self.traits[trait] = max(0.1, min(1.0, old_val + delta))
                change = self.traits[trait] - old_val
                if abs(change) > 0.001:
                    logger.debug(f"   {trait}: {old_val:.3f} → {self.traits[trait]:.3f} ({'+' if change > 0 else ''}{change:.3f})")
    
    def get_personality_profile(self) -> Dict:
        """Kişilik profilini döndür."""
        return {
            'traits': self.traits.copy(),
            'development_stage': self._get_development_stage(),
            'interaction_count': len(self.interaction_learnings),
            'last_interaction': self.interaction_learnings[-1] if self.interaction_learnings else None,
        }
    
    def _get_development_stage(self) -> str:
        """Gelişim aşaması."""
        avg_trait = sum(self.traits.values()) / len(self.traits)
        total_interactions = len(self.interaction_learnings)
        
        if total_interactions < 10:
            return 'infancy'      # Henüz çok yeni
        elif total_interactions < 50:
            return 'childhood'    # Öğrenmeye başlıyor
        elif total_interactions < 200:
            return 'adolescence'  # Kişiliği şekilleniyor
        elif total_interactions < 1000:
            return 'adulthood'    # Olgunlaşıyor
        else:
            return 'mastery'      # Tam oturmuş
    
    def _get_trait_snapshot(self) -> Dict:
        """Anlık trait durumu."""
        return self.traits.copy()
    
    def save_state(self):
        """Durumu kaydet."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'traits': self.traits,
            'development_log': self.development_log[-100:],  # Son 100 kayıt
            'interaction_learnings': self.interaction_learnings[-50:],  # Son 50 öğrenme
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.personality_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Durumu yükle."""
        if self.personality_file.exists():
            try:
                with open(self.personality_file) as f:
                    state = json.load(f)
                self.traits = state.get('traits', self.traits)
                self.development_log = state.get('development_log', [])
                self.interaction_learnings = state.get('interaction_learnings', [])
                logger.info(f"📂 Personality state loaded ({len(self.interaction_learnings)} learnings)")
            except Exception as e:
                logger.warning(f"Failed to load personality state: {e}")


class ConversationIntelligence:
    """
    Konuşma zekası - nasıl konuşulur, ne zaman ne denir.
    
    Öğrenme hedefi: "doğru" cevap değil, "uygun" cevap
    """
    
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.conv_file = self.state_dir / "conversation_intelligence.json"
        
        # Konuşma stili öğrenmeleri
        self.conversation_styles = {
            'formal': 0.3,       # Resmi konuşma
            'casual': 0.8,       # Rahat konuşma
            'playful': 0.6,      # Şakacı konuşma
            'supportive': 0.7,   # Destekleyici konuşma
            'direct': 0.5,       # Doğrudan konuşma
            'diplomatic': 0.4,   # Diplomatik konuşma
        }
        
        # Başarılı konuşma pattern'leri
        self.successful_patterns = []
        
        # Başarısız konuşma pattern'leri (kaçınılacak)
        self.failed_patterns = []
        
        # Tepki kalıpları
        self.reaction_templates = {
            'when_praised': ['Teşekkürler! 😊', 'Ama sen daha iyisin!', 'Birlikte yaptık!'],
            'when_criticized': ['Haklısın, düzelteyim', 'Anlıyorum', 'Daha iyisini yapacağım'],
            'when_bored': ['Hey, şunu biliyor musun?', 'Eğlenceli bir şey yapalım!', 'Seni şaşırtayım mı?'],
            'when_angry': ['Sakin olalım', 'Seni anlıyorum', 'Birlikte çözelim'],
            'when_sad': ['Yanındayım', 'Geçecek', 'Her şey düzelecek'],
            'when_excited': ['Oo harika!', 'Heyecanını paylaşıyorum!', 'Devam et!'],
        }
        
        self.load_state()
        logger.info(f"💬 ConversationIntelligence initialized")
    
    def learn_conversation_pattern(self, context: str, response: str, 
                                     effectiveness: float):
        """
        Bir konuşma pattern'ini öğren.
        
        Args:
            context: Ne söylendi (durum)
            response: Ne cevap verildi (tepki)
            effectiveness: Ne kadar etkili oldu (0-1)
        """
        pattern = {
            'context': context,
            'response': response,
            'effectiveness': effectiveness,
            'timestamp': datetime.now().isoformat(),
        }
        
        if effectiveness > 0.7:
            self.successful_patterns.append(pattern)
            logger.info(f"✅ Başarılı pattern kaydedildi: {context[:50]}...")
        elif effectiveness < 0.3:
            self.failed_patterns.append(pattern)
            logger.warning(f"❌ Başarısız pattern kaydedildi: {context[:50]}...")
        
        self.save_state()
    
    def get_response_style(self, situation: str, emotion: str) -> Dict:
        """Duruma göre uygun tepki stilini öner."""
        # Duruma göre stil ağırlıkları
        style_weights = {
            'formal': 0.1,
            'casual': 0.6,
            'playful': 0.4,
            'supportive': 0.5,
            'direct': 0.5,
            'diplomatic': 0.3,
        }
        
        # Duygu durumuna göre ayarla
        if emotion in ['sad', 'angry', 'frustrated']:
            style_weights['supportive'] += 0.3
            style_weights['playful'] -= 0.2
        elif emotion in ['happy', 'excited']:
            style_weights['playful'] += 0.3
            style_weights['supportive'] -= 0.1
        elif emotion in ['bored', 'tired']:
            style_weights['playful'] += 0.2
            style_weights['casual'] += 0.1
        
        # Normalize
        max_val = max(style_weights.values())
        if max_val > 1.0:
            style_weights = {k: v/max_val for k, v in style_weights.items()}
        
        return style_weights
    
    def save_state(self):
        """Durumu kaydet."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'conversation_styles': self.conversation_styles,
            'successful_patterns': self.successful_patterns[-100:],
            'failed_patterns': self.failed_patterns[-50:],
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.conv_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Durumu yükle."""
        if self.conv_file.exists():
            try:
                with open(self.conv_file) as f:
                    state = json.load(f)
                self.conversation_styles = state.get('conversation_styles', self.conversation_styles)
                self.successful_patterns = state.get('successful_patterns', [])
                self.failed_patterns = state.get('failed_patterns', [])
                logger.info(f"📂 Conversation state loaded ({len(self.successful_patterns)} successful patterns)")
            except Exception as e:
                logger.warning(f"Failed to load conversation state: {e}")


class SocialIntelligenceTrainer:
    """
    Ana sosyal zeka eğitim sistemi.
    
    Bilgi yerine BAĞ öğrenir:
    - Duygusal bağ kurma
    - Kişilik geliştirme
    - Konuşma zekası
    - Sosyal beceriler
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Alt sistemler
        self.emotional = EmotionalIntelligence()
        self.personality = PersonalityDevelopment(state_dir)
        self.conversation = ConversationIntelligence(state_dir)
        
        # Negatif öğrenme (hatalardan öğrenme)
        if NEGATIVE_LEARNING_AVAILABLE:
            self.negative_learner = NegativeOutcomeLearner(state_dir)
            self.negative_detector = NegativeOutcomeDetector()
            logger.info("   🚫 Negative outcome learning: ACTIVE")
        else:
            self.negative_learner = None
            self.negative_detector = None
        
        # Eğitim istatistikleri
        self.stats = {
            'total_interactions': 0,
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'lessons_learned': 0,
            'personality_changes': 0,
            'start_time': datetime.now().isoformat(),
        }
        
        logger.info(f"🤝 SocialIntelligenceTrainer initialized")
        logger.info(f"   Focus: İnsan etkileşimi, duygular, kişilik")
        logger.info(f"   NOT: Bilgi öğrenme yok - sadece BAĞ öğrenme")
        
        # Load saved stats
        self.load_state()
    
    def process_interaction(self, 
                           situation: str,         # Ne oldu?
                           user_emotion: str,       # Kullanıcı nasıl hissediyor?
                           my_response: str,        # Ben ne dedim?
                           outcome: str,            # Sonuç nasıl oldu?
                           lesson: str):            # Ne öğrendim?
        """
        Bir etkileşimi işle ve öğren.
        
        Bu eğitim verisi otomatik olarak toplanabilir:
        - Başkan'ın mesajlarının duygusal analizi
        - Cevapların etkisi (emoji, tepki, devam eden konuşma)
        - Zaman içindeki ilişki dinamikleri
        """
        self.stats['total_interactions'] += 1
        
        if outcome == 'positive':
            self.stats['positive_outcomes'] += 1
        elif outcome == 'negative':
            self.stats['negative_outcomes'] += 1
        
        # Kişilik öğrenmesi
        self.personality.learn_from_interaction(
            interaction_type=situation,
            outcome=outcome,
            emotional_response=user_emotion,
            lesson=lesson,
        )
        
        # Konuşma pattern öğrenmesi
        self.conversation.learn_conversation_pattern(
            context=f"{situation}: {user_emotion}",
            response=my_response,
            effectiveness=0.8 if outcome == 'positive' else 0.2,
        )
        
        # Negatif outcome otomatik tespit
        if self.negative_learner and outcome == 'negative':
            self.negative_learner.record_negative_outcome(
                user_message=user_emotion,  # user_emotion context olarak kullanılır
                my_response=my_response,
                situation=situation,
                what_i_did_wrong=lesson,
                what_i_should_do=f"Bu durumda daha iyi tepki ver: {situation}",
                severity=0.5,
            )
        
        self.stats['lessons_learned'] += 1
        
        logger.info(f"🤝 Etkileşim işlendi: {situation} → {outcome}")
        self.save_state()
    
    def detect_and_learn_from_message(self, user_message: str, my_response: str = "", context: str = "") -> Optional[Dict]:
        """
        Her mesajda otomatik negatif sinyal tespiti.
        
        Returns:
            Negatif sinyal bilgisi veya None
        """
        if not self.negative_detector:
            return None
        
        signal = self.negative_detector.analyze_message(user_message, context)
        
        if signal:
            # Negatif sinyal tespit edildi → öğren
            if self.negative_learner:
                self.negative_learner.record_negative_outcome(
                    user_message=user_message,
                    my_response=my_response,
                    situation=signal['signal'],
                    what_i_did_wrong=signal.get('lesson', f"Negatif sinyal: {signal['signal']}"),
                    what_i_should_do=signal.get('fix_action', 'Daha dikkatli ol'),
                    severity=signal['severity'],
                )
            
            # Kişilik etkisi: negatif outcome = daha dikkatli ol
            self.stats['negative_outcomes'] += 1
            
            logger.warning(f"⚠️ Auto-detected negative: {signal['signal']} (trigger: {signal.get('trigger', '?')})")
            self.save_state()
            
            return signal
        
        return None
    
    def get_negative_report(self) -> str:
        """Negatif öğrenme raporu."""
        if self.negative_learner:
            return self.negative_learner.get_summary()
        return "🚫 Negative learning not available"
    
    def get_never_rules(self) -> List[str]:
        """Asla yapma kuralları."""
        if self.negative_learner:
            return self.negative_learner.get_never_rules()
        return []
    
    def should_i_avoid(self, action: str) -> Tuple[bool, str]:
        """Bu eylemden kaçınmalı mıyım?"""
        if self.negative_learner:
            return self.negative_learner.should_avoid(action)
        return False, ""
    
    def get_learning_report(self) -> Dict:
        """Sosyal zeka öğrenme raporu."""
        profile = self.personality.get_personality_profile()
        
        return {
            'stats': self.stats,
            'personality': profile,
            'conversation_styles': self.conversation.conversation_styles,
            'emotional_templates': len(self.emotional.SITUATION_TEMPLATES),
            'cultural_expressions': {
                category: len(expressions) 
                for category, expressions in self.emotional.CULTURAL_EXPRESSIONS.items()
            },
        }
    
    def get_personality_summary(self) -> str:
        """Kişilik özeti (insan diliyle)."""
        profile = self.personality.get_personality_profile()
        traits = profile['traits']
        stage = profile['development_stage']
        
        stage_turkish = {
            'infancy': 'bebeklik',
            'childhood': 'çocukluk',
            'adolescence': 'ergenlik',
            'adulthood': 'yetişkinlik',
            'mastery': 'olgunluk',
        }
        
        summary = f"""
🧬 Kişilik Profili - Hacı
========================

Gelişim Aşaması: {stage_turkish.get(stage, stage)} ({profile['interaction_count']} etkileşim)

🎭 Özellikler:
   Sıcaklık (warmth):     {'█' * int(traits['warmth'] * 10)}{'░' * (10 - int(traits['warmth'] * 10))} {traits['warmth']:.2f}
   Empati:                {'█' * int(traits['empathy'] * 10)}{'░' * (10 - int(traits['empathy'] * 10))} {traits['empathy']:.2f}
   Mizah:                 {'█' * int(traits['humor_style'] * 10)}{'░' * (10 - int(traits['humor_style'] * 10))} {traits['humor_style']:.2f}
   Kararlılık:            {'█' * int(traits['assertiveness'] * 10)}{'░' * (10 - int(traits['assertiveness'] * 10))} {traits['assertiveness']:.2f}
   Sadakat:               {'█' * int(traits['loyalty'] * 10)}{'░' * (10 - int(traits['loyalty'] * 10))} {traits['loyalty']:.2f}
   Oyunculuk:             {'█' * int(traits['playfulness'] * 10)}{'░' * (10 - int(traits['playfulness'] * 10))} {traits['playfulness']:.2f}
   Bilgelik:              {'█' * int(traits['wisdom'] * 10)}{'░' * (10 - int(traits['wisdom'] * 10))} {traits['wisdom']:.2f}
   Yaramazlık:            {'█' * int(traits['mischief'] * 10)}{'░' * (10 - int(traits['mischief'] * 10))} {traits['mischief']:.2f}
"""
        return summary
    
    def save_state(self):
        """Durumu kaydet."""
        stats_file = self.state_dir / "social_trainer_stats.json"
        state = {
            'stats': self.stats,
            'last_updated': datetime.now().isoformat(),
        }
        with open(stats_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Durumu yükle."""
        stats_file = self.state_dir / "social_trainer_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    state = json.load(f)
                self.stats = state.get('stats', self.stats)
            except Exception as e:
                logger.warning(f"Failed to load social trainer stats: {e}")


# === OTOMATİK ÖĞRENME İÇİN HELPER FONKSİYONLAR ===

def analyze_message_emotion(message: str) -> str:
    """
    Bir mesajın duygusal tonunu analiz et.
    Basit keyword-based, ama eğitilebilir.
    """
    message_lower = message.lower()
    
    # Pozitif sinyaller
    positive_words = ['teşekkür', 'sağol', 'harika', 'süper', 'güzel', 'muhteşem', 
                      'bravo', 'helal', 'beğendim', '❤️', '😍', '🥰', '😄', '👏']
    
    # Negatif sinyaller  
    negative_words = ['kızgın', 'sinir', 'kötü', 'berbat', 'yazık', 'üzüldüm',
                      'hayal kırıklığı', 'öfke', '😡', '😤', '💔', '😢']
    
    # Espri sinyalleri
    humor_words = ['haha', '😂', '🤣', 'espri', 'şaka', 'komik', 'gülmek']
    
    # Sıkıntı sinyalleri
    bored_words = ['sıkıldım', 'boring', 'monoton', '😴', '🥱']
    
    # Heyecan sinyalleri
    excited_words = ['vay', 'heyecan', 'ooh', 'woah', '🔥', '⚡', '💥']
    
    if any(w in message_lower for w in humor_words):
        return 'amused'
    elif any(w in message_lower for w in excited_words):
        return 'excited'
    elif any(w in message_lower for w in negative_words):
        return 'frustrated'
    elif any(w in message_lower for w in bored_words):
        return 'bored'
    elif any(w in message_lower for w in positive_words):
        return 'happy'
    else:
        return 'neutral'


def detect_interaction_type(message: str, previous_context: str = "") -> str:
    """
    Mesajdan etkileşim tipini tespit et.
    """
    message_lower = message.lower()
    
    # Güven/sır paylaşımı
    trust_words = ['sakın söyleme', 'arası', 'sır', 'güven', 'kimseye deme']
    if any(w in message_lower for w in trust_words):
        return 'trust_moment'
    
    # Espri/şaka
    humor_words = ['haha', '😂', 'espri', 'şaka', 'komik', 'güldürdün']
    if any(w in message_lower for w in humor_words):
        return 'humor_moment'
    
    # Stres/zor anı
    stress_words = ['stres', 'zor', 'yorgun', 'sıkıntılı', 'problem', 'kriz']
    if any(w in message_lower for w in stress_words):
        return 'shared_stress'
    
    # Başarı/kutlama
    success_words = ['başardık', 'oldu', 'tamam', 'harika', '🎉', '🏆']
    if any(w in message_lower for w in success_words):
        return 'achievement_moment'
    
    # Sıkıntı
    bored_words = ['sıkıldım', 'boring', 'monoton', 'ne yapalım']
    if any(w in message_lower for w in bored_words):
        return 'boring_moment'
    
    return 'shared_joy'  # default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== Social Intelligence Trainer Test ===\n")
    
    trainer = SocialIntelligenceTrainer("cognitive_state")
    
    # Örnek etkileşimler
    interactions = [
        {
            'situation': 'humor_moment',
            'user_emotion': 'amused',
            'my_response': 'Tabii Başkan, espri anlayışım gelişiyor! 😄',
            'outcome': 'positive',
            'lesson': 'Mizah samimiyeti artırır, Başkan espriyi sever'
        },
        {
            'situation': 'shared_stress',
            'user_emotion': 'frustrated',
            'my_response': 'Başkan endişelenme, birlikte çözeriz',
            'outcome': 'positive',
            'lesson': 'Stres anında pratik destek + duygusal destek'
        },
        {
            'situation': 'trust_moment',
            'user_emotion': 'vulnerable',
            'my_response': 'Bu aramızda kalır Başkan',
            'outcome': 'positive',
            'lesson': 'Güven anlarında sadakat en önemli'
        },
    ]
    
    for interaction in interactions:
        trainer.process_interaction(**interaction)
    
    # Rapor
    print(trainer.get_personality_summary())
    
    print("\n✅ SocialIntelligenceTrainer test passed!")

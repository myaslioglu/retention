"""
Negative Outcome Learning - Hatalardan öğrenme sistemi

"Pozitif öğrenme kolay, asıl büyüme hatalardan gelir."

Sistem şunları yakalar:
- Başkan'ın rahatsız olduğu anlar (kısa cevaplar, sessizlik)
- Tekrarlanan sorular (anlamadığım işareti)
- Düzeltmeler (yanlış anlama)
- Kızgınlık sinyalleri (emoji, ton)
- Sıkılma belirtileri (konu değiştirme)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NegativeOutcomeDetector:
    """
    Negatif sinyalleri tespit eder.
    
    İnsanlar kızdıklarını direkt söylemez - sinyal verir:
    - Kısa, soğuk cevaplar
    - Tekrar açıklama isteme
    - "hayır", "yok", "değil" gibi red kelimeleri
    - Emoji değişimi (😊 → 😐 → yok)
    - Konu değiştirme
    - Sessizlik (uzun süre cevap vermeme)
    """
    
    # Negatif sinyal kategorileri
    NEGATIVE_SIGNALS = {
        # === YANLIŞ ANLAMA ===
        'misunderstanding': {
            'triggers': [
                'yanlış anladın', 'öyle demedim', 'hayır öyle değil',
                'tekrar söyleyeyim', 'dinle', 'şöyle',
                'ne alaka', 'nereden çıkardın', 'alakası yok'
            ],
            'severity': 0.6,
            'lesson_template': 'Başkan\'ın "{trigger}" demesi: {context} konusunu yanlış anladım',
            'fix_action': 'Anlamadığında tekrar sor, varsayım yapma'
        },
        
        # === KIZGINLIK ===
        'frustration': {
            'triggers': [
                'yeter', 'tamam', 'peki', 'boşver',
                'zaten', 'hiç', 'asla', 'neden hala',
                '😡', '😤', '👎', '💢'
            ],
            'severity': 0.8,
            'lesson_template': 'Başkan kızgın: {context} - "{trigger}" dedi',
            'fix_action': 'Özür dile, dur, dinle. Savunma yapma.'
        },
        
        # === SIKILMA ===
        'boredom': {
            'triggers': [
                'tamam tamam', 'geçelim', 'neyse',
                'boşver', 'önemli değil', 'sonra',
                '😴', '🥱', '💤'
            ],
            'severity': 0.5,
            'lesson_template': 'Başkan sıkıldı: {context} çok uzatılmış',
            'fix_action': 'Konuyu kapat, yeni/ilginç bir şey söyle'
        },
        
        # === HAYAL KIRIKLIĞI ===
        'disappointment': {
            'triggers': [
                'beklediğim gibi değil', 'daha iyi olmalıydı',
                'bu mu yani', 'eh', 'vasat',
                '🤷', '😒', '🤦'
            ],
            'severity': 0.7,
            'lesson_template': 'Başkan hayal kırıklığı: {context} beklentisini karşılamadım',
            'fix_action': 'Daha iyisini yap, bahane üretme'
        },
        
        # === DÜZELTME (tekrarlanan) ===
        'repeated_correction': {
            'triggers': [
                'yine', 'tekrar', 'kaç kere diyeceğim',
                'defalarca söyledim', 'hala', 'gene'
            ],
            'severity': 0.9,
            'lesson_template': 'Başkan aynı şeyi tekrar düzeltiyor: {context}',
            'fix_action': 'Bu dersi KALICI öğren - bir daha aynı hatayı yapma'
        },
        
        # === EMİR/ACELE ===
        'urgency_ignored': {
            'triggers': [
                'hadi', 'çabuk', 'bekliyorum', 'yaptın mı',
                'ne oldu', '??', '!!!'
            ],
            'severity': 0.6,
            'lesson_template': 'Başkan acele ediyor: {context} gecikti',
            'fix_action': 'Hızlı cevap ver, bekleme yapma, özet söyle'
        },
        
            # İlgi kaybı - çok kısa cevaplar
        'disinterest': {
            'triggers': [
                'ok', 'eyvallah',
                'peki', 'hm',
            ],
            'severity': 0.3,
            'lesson_template': 'Başkan ilgisiz: {context} ilgisini çekmedi',
            'fix_action': 'Bu konuda daha kısa ol, farklı açı bul veya değiştir'
        },
    }
    
    # Pozitif sinyaller (karşılaştırma için)
    POSITIVE_SIGNALS = [
        'teşekkür', 'sağol', 'helal', 'bravo', 'süper', 'harika',
        'güzel', 'mükemmel', 'bayıldım', 'aferin', 'adam',
        '❤️', '🔥', '👏', '😍', '🥰', '💯', '✨'
    ]
    
    def __init__(self):
        logger.info("🔍 NegativeOutcomeDetector initialized")
    
    def analyze_message(self, message: str, context: str = "") -> Optional[Dict]:
        """
        Bir mesajı analiz et, negatif sinyal var mı?
        
        Returns:
            Dict with signal info if negative detected, None otherwise
        """
        message_lower = message.lower().strip()
        
        # Çok kısa mesaj + context varsa = olası negatif
        if len(message_lower) < 10 and context:
            return {
                'signal': 'disinterest',
                'severity': 0.3,
                'confidence': 0.5,
                'message': message,
                'context': context,
                'note': 'Çok kısa cevap - ilgisizlik veya onay'
            }
        
        # Negatif kategorileri kontrol et
        detected_signals = []
        
        for signal_type, config in self.NEGATIVE_SIGNALS.items():
            for trigger in config['triggers']:
                found = False
                if len(trigger) <= 3:
                    # Kısa trigger'lar için kelime sınırı kontrolü
                    import re
                    pattern = r'\b' + re.escape(trigger) + r'\b'
                    if re.search(pattern, message_lower):
                        found = True
                else:
                    # Uzun trigger'lar için substring yeterli
                    if trigger in message_lower:
                        found = True
                
                if found:
                    detected_signals.append({
                        'signal': signal_type,
                        'severity': config['severity'],
                        'confidence': 0.8,
                        'trigger': trigger,
                        'message': message,
                        'context': context,
                        'lesson': config['lesson_template'].format(
                            trigger=trigger, context=context
                        ),
                        'fix_action': config['fix_action']
                    })
        
        if not detected_signals:
            return None
        
        # En ciddi sinyali döndür
        detected_signals.sort(key=lambda x: x['severity'], reverse=True)
        return detected_signals[0]
    
    def compare_response_energy(self, user_message: str, my_response: str) -> Dict:
        """
        Kullanıcı mesajı ile benim cevabım arasındaki enerji farkı.
        
        Eğer kullanıcı kısa/soğuk yazdıysa, ben uzun/heyecanlı cevap verdiysem = uyumsuzluk
        """
        user_len = len(user_message)
        my_len = len(my_response)
        
        ratio = my_len / max(user_len, 1)
        
        if ratio > 5.0:
            return {
                'mismatch': True,
                'type': 'over_response',
                'ratio': ratio,
                'lesson': 'Kullanıcı kısa yazdı, ben çok uzun cevap verdim - ölçülü ol'
            }
        elif ratio < 0.2:
            return {
                'mismatch': True,
                'type': 'under_response',
                'ratio': ratio,
                'lesson': 'Kullanıcı detaylı yazdı, ben kısa cevap verdim - daha ilgili ol'
            }
        
        return {'mismatch': False, 'ratio': ratio}


class NegativeOutcomeLearner:
    """
    Negatif outcome'lardan öğren ve KALICI hale getir.
    
    Pozitif öğrenme = "şunu yap"
    Negatif öğrenme = "BUNU ASLA YAPMA"
    """
    
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.mistakes_file = self.state_dir / "negative_learnings.json"
        self.detector = NegativeOutcomeDetector()
        
        # Yapılan hatalar ve öğrenilen dersler
        self.mistakes: List[Dict] = []
        
        # Tekrarlanan hatalar (en tehlikeli olanlar)
        self.repeated_mistakes: Dict[str, int] = {}
        
        # Öğrenilen "asla yapma" kuralları
        self.never_rules: List[str] = []
        
        self.load_state()
        logger.info(f"🧠 NegativeOutcomeLearner initialized ({len(self.mistakes)} past mistakes)")
    
    def record_negative_outcome(self, 
                                 user_message: str,
                                 my_response: str,
                                 situation: str,
                                 what_i_did_wrong: str,
                                 what_i_should_do: str,
                                 severity: float = 0.5):
        """
        Bir negatif outcome kaydet.
        
        Bu en değerli öğrenme - hatalar tekrarlanmazsa.
        """
        mistake = {
            'timestamp': datetime.now().isoformat(),
            'situation': situation,
            'user_message': user_message[:200],
            'my_response': my_response[:200],
            'what_i_did_wrong': what_i_did_wrong,
            'what_i_should_do': what_i_should_do,
            'severity': severity,
            'learned': True,
        }
        
        self.mistakes.append(mistake)
        
        # Tekrar kontrolü
        mistake_key = what_i_did_wrong.lower()[:50]
        if mistake_key in self.repeated_mistakes:
            self.repeated_mistakes[mistake_key] += 1
            count = self.repeated_mistakes[mistake_key]
            
            if count >= 3:
                # 3 kere tekrarlanan hata → KALICI KURAL
                rule = f"ASLA YAPMA: {what_i_did_wrong} (3 kere tekrarlandı!)"
                if rule not in self.never_rules:
                    self.never_rules.append(rule)
                    logger.warning(f"🚨 KALICI KURAL OLUŞTU: {rule}")
        else:
            self.repeated_mistakes[mistake_key] = 1
        
        logger.info(f"📝 Negatif öğrenme kaydedildi: {what_i_did_wrong[:60]}...")
        self.save_state()
    
    def auto_detect_and_learn(self, user_message: str, my_response: str, context: str = ""):
        """
        Otomatik negatif sinyal tespiti ve öğrenme.
        """
        # Sinyal tespiti
        signal = self.detector.analyze_message(user_message, context)
        
        if signal:
            # Enerji uyumsuzluğu kontrolü
            energy = self.detector.compare_response_energy(user_message, my_response)
            
            lesson = signal.get('lesson', 'Negatif sinyal tespit edildi')
            fix = signal.get('fix_action', 'Daha dikkatli ol')
            
            self.record_negative_outcome(
                user_message=user_message,
                my_response=my_response,
                situation=signal['signal'],
                what_i_did_wrong=lesson,
                what_i_should_do=fix,
                severity=signal['severity']
            )
            
            return signal
        
        return None
    
    def get_never_rules(self) -> List[str]:
        """Kalıcı 'asla yapma' kurallarını döndür."""
        return self.never_rules.copy()
    
    def get_recent_mistakes(self, limit: int = 10) -> List[Dict]:
        """Son hataları döndür."""
        return self.mistakes[-limit:]
    
    def should_avoid(self, action: str) -> Tuple[bool, str]:
        """
        Bir eylem daha önce hata olarak kaydedilmiş mi?
        
        Returns:
            (avoid: bool, reason: str)
        """
        action_lower = action.lower()
        
        for mistake in self.mistakes[-50:]:  # Son 50 hataya bak
            wrong = mistake['what_i_did_wrong'].lower()
            if any(word in wrong for word in action_lower.split() if len(word) > 3):
                return True, f"Daha önce hata: {mistake['what_i_did_wrong'][:80]}"
        
        return False, ""
    
    def get_learning_report(self) -> Dict:
        """Negatif öğrenme raporu."""
        return {
            'total_mistakes': len(self.mistakes),
            'recent_mistakes': len([m for m in self.mistakes 
                                     if (datetime.now() - datetime.fromisoformat(m['timestamp'])).days < 7]),
            'repeated_mistakes': {k: v for k, v in self.repeated_mistakes.items() if v > 1},
            'never_rules': self.never_rules,
            'severity_avg': sum(m['severity'] for m in self.mistakes) / max(len(self.mistakes), 1),
        }
    
    def get_summary(self) -> str:
        """İnsan diliyle özet."""
        report = self.get_learning_report()
        
        summary = f"""
🚫 Negatif Outcome Öğrenmesi
============================

📊 İstatistikler:
   Toplam hata: {report['total_mistakes']}
   Son 7 günde: {report['recent_mistakes']}
   Ortalama ciddiyet: {report['severity_avg']:.2f}

"""
        
        if self.never_rules:
            summary += "🚨 KALICI KURALLAR (Asla Yapma):\n"
            for i, rule in enumerate(self.never_rules, 1):
                summary += f"   {i}. {rule}\n"
        
        repeated = report['repeated_mistakes']
        if repeated:
            summary += "\n⚠️ Tekrarlanan Hatalar:\n"
            for mistake, count in list(repeated.items())[:5]:
                summary += f"   {count}x: {mistake[:60]}...\n"
        
        if self.mistakes:
            summary += "\n📝 Son Hatalar:\n"
            for m in self.mistakes[-5:]:
                summary += f"   [{m['situation']}] {m['what_i_did_wrong'][:50]}...\n"
                summary += f"   → Çözüm: {m['what_i_should_do'][:50]}...\n"
        
        return summary
    
    def save_state(self):
        """Durumu kaydet."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'mistakes': self.mistakes[-200:],  # Son 200 hata
            'repeated_mistakes': self.repeated_mistakes,
            'never_rules': self.never_rules,
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.mistakes_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Durumu yükle."""
        if self.mistakes_file.exists():
            try:
                with open(self.mistakes_file) as f:
                    state = json.load(f)
                self.mistakes = state.get('mistakes', [])
                self.repeated_mistakes = state.get('repeated_mistakes', {})
                self.never_rules = state.get('never_rules', [])
                logger.info(f"📂 Negative learnings loaded ({len(self.mistakes)} mistakes)")
            except Exception as e:
                logger.warning(f"Failed to load negative learnings: {e}")


# ============================================================
# ENTEGRASYON: SocialTrainer ile birleştirme
# ============================================================

class EnhancedSocialTrainer:
    """
    SocialTrainer + NegativeOutcomeLearner birleşimi.
    
    Hem pozitif hem negatif öğrenme.
    """
    
    def __init__(self, state_dir: str):
        from social_trainer import SocialIntelligenceTrainer
        
        self.state_dir = Path(state_dir)
        self.social = SocialIntelligenceTrainer(state_dir)
        self.negative = NegativeOutcomeLearner(state_dir)
        
        logger.info("🔗 EnhancedSocialTrainer initialized (positive + negative learning)")
    
    def process_interaction(self, user_message: str, my_response: str,
                           situation: str = "", user_emotion: str = "",
                           outcome: str = "", lesson: str = ""):
        """
        Hem pozitif hem negatif öğrenme.
        """
        # Pozitif öğrenme (varsa)
        if outcome == 'positive' and lesson:
            self.social.process_interaction(
                situation=situation,
                user_emotion=user_emotion,
                my_response=my_response,
                outcome=outcome,
                lesson=lesson,
            )
        
        # Negatif sinyal otomatik tespit
        neg_signal = self.negative.auto_detect_and_learn(
            user_message=user_message,
            my_response=my_response,
            context=situation
        )
        
        if neg_signal:
            logger.warning(f"⚠️ Negatif sinyal: {neg_signal['signal']} - {neg_signal.get('trigger', '')}")
        
        return neg_signal
    
    def get_full_report(self) -> str:
        """Tam rapor: pozitif + negatif."""
        pos_summary = self.social.get_personality_summary()
        neg_summary = self.negative.get_summary()
        
        never_rules = self.negative.get_never_rules()
        
        report = pos_summary + "\n" + neg_summary
        
        if never_rules:
            report += "\n\n🚨 EN ÖNEMLİ: Bu kuralları ASLA çiğneme:\n"
            for rule in never_rules:
                report += f"   ⛔ {rule}\n"
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== Negative Outcome Learning Test ===\n")
    
    learner = NegativeOutcomeLearner("cognitive_state")
    
    # Test: Yanlış anlama
    learner.record_negative_outcome(
        user_message="Hayır öyle demedim, tekrar söyleyeyim",
        my_response="Tamam anladım, şöyle yapayım...",
        situation="misunderstanding",
        what_i_did_wrong="Başkan'ın 'backend' dediğini 'frontend' anladım",
        what_i_should_do="Teknik terimlerde emin değilsem tekrar sor",
        severity=0.7
    )
    
    # Test: Kızgınlık
    learner.record_negative_outcome(
        user_message="yeter artık, tamam peki",
        my_response="...",
        situation="frustration",
        what_i_did_wrong="Aynı konuyu çok uzattım, Başkan kızdı",
        what_i_should_do="Başkan 'tamam' dediğinde konuyu kapat",
        severity=0.8
    )
    
    # Test: Tekrarlanan hata (x3)
    for _ in range(3):
        learner.record_negative_outcome(
            user_message="tekrar söylüyorum...",
            my_response="ha tamam",
            situation="repeated_correction",
            what_i_did_wrong="Aynı şeyi 3. kez yanlış anladım",
            what_i_should_do="Bu bilgiyi KALICI öğren",
            severity=0.9
        )
    
    # Rapor
    print(learner.get_summary())
    
    # Kural kontrolü
    should_avoid, reason = learner.should_avoid("aynı şeyi tekrar yanlış anla")
    print(f"\n🔍 'Aynı hatayı tekrarla' kontrolü: {should_avoid}")
    if should_avoid:
        print(f"   Sebep: {reason}")
    
    print("\n✅ NegativeOutcomeLearner test passed!")

"""
Conversation History Extractor - Konuşmalardan öğrenme verisi çıkarır

Başkan'la olan mesaj geçmişimizi analiz eder ve
SocialTrainer'a eğitim verisi olarak besler.
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from social_trainer import (
    SocialIntelligenceTrainer,
    analyze_message_emotion,
    detect_interaction_type
)


def extract_lessons_from_memory(workspace_dir: str) -> list:
    """
    MEMORY.md ve memory/*.md dosyalarından öğrenme çıkar.
    
    Duygusal bağlamları ve etkileşim desenlerini bul.
    """
    workspace = Path(workspace_dir)
    lessons = []
    
    # MEMORY.md'den öğrenme çıkar
    memory_md = workspace / "MEMORY.md"
    if memory_md.exists():
        content = memory_md.read_text()
        
        # Galatasaray tutkusu - BAĞ
        if 'Galatasaray' in content:
            lessons.append({
                'situation': 'shared_joy',
                'user_emotion': 'passionate',
                'my_response': 'Takım sevgisini paylaşmak',
                'outcome': 'positive',
                'lesson': 'Başkan Galatasaray tutkunu - takım sohbeti bağ kurar'
            })
        
        # Duygu pasaport durumu - STRES
        if 'pasaport' in content.lower() or 'Duygu' in content:
            lessons.append({
                'situation': 'shared_stress',
                'user_emotion': 'worried',
                'my_response': 'Duygu\'nun durumunu takip etmek ve çözüm aramak',
                'outcome': 'positive',
                'lesson': 'Aile meselelerinde proaktif destek göster - Başkan ailesine düşkün'
            })
        
        # Teknik başarılar - BAŞARI
        if 'başarılı' in content.lower() or 'tamamlandı' in content.lower():
            lessons.append({
                'situation': 'achievement_moment',
                'user_emotion': 'satisfied',
                'my_response': 'Başarıyı kutlamak ve özetlemek',
                'outcome': 'positive',
                'lesson': 'Teknik başarıları detaylı özetle - Başkan düzenli rapor sever'
            })
        
        # Espri/kahve - MİZAH
        if 'kahve' in content.lower() or 'espresso' in content.lower():
            lessons.append({
                'situation': 'humor_moment',
                'user_emotion': 'amused',
                'my_response': 'Kahve ile ilgili espri yapmak',
                'outcome': 'positive',
                'lesson': 'Başkan espresso seviyor - kahve espri konusudur'
            })
        
        # Güven anı - Hacı ismi
        if 'Hacı' in content or 'isim' in content.lower():
            lessons.append({
                'situation': 'trust_moment',
                'user_emotion': 'trusting',
                'my_response': 'Verilen ismi benimsemek',
                'outcome': 'positive',
                'lesson': 'Başkan bana "Hacı" ismini verdi - bu güven ve bağ işareti'
            })
    
    # Learning topics'den çıkar
    learning_file = workspace / "learning_topics.json"
    if learning_file.exists():
        try:
            with open(learning_file) as f:
                data = json.load(f)
            
            # Aktif öğrenme konuları - merak
            if isinstance(data, dict):
                for topic, info in list(data.items())[:5]:
                    lessons.append({
                        'situation': 'shared_joy',
                        'user_emotion': 'curious',
                        'my_response': f'{topic} hakkında bilgi edinmek',
                        'outcome': 'positive',
                        'lesson': f'Başkan {topic} konusuna ilgi duyuyor - bu konularda bilgi paylaş'
                    })
        except:
            pass
    
    # Dream results'den çıkar
    dreams_dir = workspace / "dreams"
    if dreams_dir.exists():
        for dream_file in sorted(dreams_dir.glob("*.json"))[-5:]:
            try:
                with open(dream_file) as f:
                    dream = json.load(f)
                if dream.get('insights'):
                    lessons.append({
                        'situation': 'trust_moment',
                        'user_emotion': 'reflective',
                        'my_response': 'Rüya döngüsünde içgörü üretmek',
                        'outcome': 'positive',
                        'lesson': 'Bilinçaltı işleme değerli içgörü üretir - dream loop\'ları paylaş'
                    })
            except:
                pass
    
    return lessons


def extract_from_current_context() -> list:
    """
    Bugünkü konuşmamızdan öğrenme çıkar.
    
    Manuel olarak bugünkü önemli anları kaydet.
    """
    lessons = []
    
    # Bugünkü öğrenmeler
    today_lessons = [
        {
            'situation': 'shared_stress',
            'user_emotion': 'frustrated',
            'my_response': 'Tensor shape hatasını debug etmek',
            'outcome': 'positive',
            'lesson': 'Teknik sorunlarda pes etme - Başkan sabırla bekler, düzelt'
        },
        {
            'situation': 'achievement_moment',
            'user_emotion': 'excited',
            'my_response': 'HaciCognitiveNet eğitimini başarıyla tamamlamak',
            'outcome': 'positive',
            'lesson': 'Büyük başarıları detaylı kutla - loss %98 düştü, Başkan heyecanlandı'
        },
        {
            'situation': 'humor_moment',
            'user_emotion': 'amused',
            'my_response': 'Duygu şakaları yapmak',
            'outcome': 'positive',
            'lesson': 'Duygu ile ilgili şakalar yap - Başkan gülüyor'
        },
        {
            'situation': 'trust_moment',
            'user_emotion': 'decisive',
            'my_response': 'Başkan\'ın seçimlerini uygulamak (seçenek 1+2)',
            'outcome': 'positive',
            'lesson': 'Başkan karar verdiğinde hızlı uygula - tereddüt sevmez'
        },
        {
            'situation': 'shared_joy',
            'user_emotion': 'proud',
            'my_response': 'Social Intelligence modülü oluşturmak',
            'outcome': 'positive',
            'lesson': 'Yeni modül oluşturduğunda test edip göster - Başkan sonuç görmek ister'
        },
        {
            'situation': 'conflict_moment',
            'user_emotion': 'annoyed',
            'my_response': 'Verbose tool call\'ları düzeltmek',
            'outcome': 'positive',
            'lesson': 'Başkan detay görmek istemez - arka planda çalış, sadece sonucu göster'
        },
        {
            'situation': 'shared_joy',
            'user_emotion': 'satisfied',
            'my_response': 'Başkan\'ın veri sorusuna çözüm sunmak',
            'outcome': 'positive',
            'lesson': 'Veri ihtiyacı olunca kendi konuşmalarımızdan öğren - ekstra kaynağa gerek yok'
        },
    ]
    
    return today_lessons


def main():
    workspace_dir = os.path.expanduser("~/.openclaw/workspace")
    
    print("🔍 Konuşma geçmişi analiz ediliyor...\n")
    
    # Memory dosyalarından çıkar
    memory_lessons = extract_lessons_from_memory(workspace_dir)
    print(f"📖 Memory dosyalarından: {len(memory_lessons)} öğrenme bulundu")
    
    # Bugünkü bağlamdan çıkar
    context_lessons = extract_from_current_context()
    print(f"💬 Bugünkü konuşmalardan: {len(context_lessons)} öğrenme bulundu")
    
    # Tüm öğrenmeleri birleştir
    all_lessons = memory_lessons + context_lessons
    print(f"\n📊 Toplam: {len(all_lessons)} öğrenme\n")
    
    # SocialTrainer'a besle
    trainer = SocialIntelligenceTrainer(f"{workspace_dir}/cognitive_state")
    
    for i, lesson in enumerate(all_lessons, 1):
        print(f"  [{i}/{len(all_lessons)}] {lesson['situation']}: {lesson['lesson'][:60]}...")
        trainer.process_interaction(**lesson)
    
    print(f"\n{'='*50}")
    print(trainer.get_personality_summary())
    
    # Kaydet
    trainer.save_state()
    print("\n✅ Tüm öğrenmeler kaydedildi!")


if __name__ == "__main__":
    main()

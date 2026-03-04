#!/usr/bin/env python3
"""
AUTOMATIC MEMORY CONSOLIDATION SYSTEM
Daily memory'leri otomatik olarak long-term memory'ye aktarır
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib

class MemoryConsolidator:
    """Otomatik memory consolidation sistemi"""
    
    def __init__(self, memory_dir: str = "memory", long_term_file: str = "MEMORY.md"):
        self.memory_dir = Path(memory_dir)
        self.long_term_file = Path(long_term_file)
        self.archive_dir = self.memory_dir / "archive"
        
        # Önemlilik keyword'leri
        self.importance_keywords = [
            'karar', 'önemli', 'kritik', 'hatırla', 'yapalım', 'yapacağım',
            'öğrendim', 'dikkat', 'not', 'todo', 'hatırlat', 'tercih',
            'seviyorum', 'severim', 'beğenirim', 'proje', 'sistem',
            'entegre', 'kur', 'test', 'başkan', 'hacı', 'kripto',
            'retention', 'memory', 'hata', 'çözüm', 'başarı', 'tamamlandı',
            'tamaml', 'bitirdim', 'başladım', 'plan', 'strateji', 'tavsiye',
            'öneri', 'lesson learned', 'ders', 'tecrübe'
        ]
        
        # Memory türleri ve emojileri
        self.memory_types = {
            'decision': {'emoji': '🤔', 'desc': 'Kararlar'},
            'lesson': {'emoji': '📚', 'desc': 'Öğrenilen Dersler'},
            'achievement': {'emoji': '🏆', 'desc': 'Başarılar'},
            'preference': {'emoji': '👤', 'desc': 'Tercihler'},
            'project': {'emoji': '🚀', 'desc': 'Projeler'},
            'technical': {'emoji': '🔧', 'desc': 'Teknik Bilgiler'},
            'insight': {'emoji': '💡', 'desc': 'İçgörüler'},
            'reminder': {'emoji': '⏰', 'desc': 'Hatırlatıcılar'}
        }
        
        print("🧠 AUTOMATIC MEMORY CONSOLIDATOR")
        print("=" * 50)
        print(f"Memory dir: {self.memory_dir}")
        print(f"Long-term file: {self.long_term_file}")
        print(f"Archive dir: {self.archive_dir}")
    
    def _ensure_directories(self):
        """Gerekli dizinleri oluştur"""
        self.memory_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
    
    def _get_daily_files(self, days_back: int = 7) -> List[Path]:
        """Son X günün daily memory dosyalarını getir"""
        daily_files = []
        today = datetime.now()
        
        for i in range(days_back):
            date = today - timedelta(days=i)
            filename = date.strftime("%Y-%m-%d") + ".md"
            filepath = self.memory_dir / filename
            
            if filepath.exists():
                daily_files.append(filepath)
        
        return daily_files
    
    def _read_daily_file(self, filepath: Path) -> Dict:
        """Daily memory dosyasını oku ve analiz et"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dosya adından tarihi al
            date_str = filepath.stem
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Satırları analiz et
            lines = content.split('\n')
            sections = []
            current_section = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line.startswith('##') or line.startswith('###'):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                if line:
                    current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            return {
                'date': date,
                'date_str': date_str,
                'filepath': filepath,
                'content': content,
                'sections': sections,
                'line_count': len(lines),
                'word_count': len(content.split())
            }
        
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return None
    
    def _calculate_importance_score(self, text: str) -> float:
        """Metnin önemlilik skorunu hesapla (0-1)"""
        text_lower = text.lower()
        score = 0.0
        
        # Keyword matching
        for keyword in self.importance_keywords:
            if keyword in text_lower:
                score += 0.05
                # Özel keyword'ler ekstra puan
                if keyword in ['karar', 'önemli', 'kritik', 'lesson learned', 'ders']:
                    score += 0.1
        
        # Length factor (çok kısa veya çok uzun düşük puan)
        words = text.split()
        if 10 <= len(words) <= 100:  # Optimal length
            score += 0.1
        elif len(words) > 100:  # Çok uzun
            score += 0.05
        
        # Structure factor (başlık, liste vs.)
        if text.startswith('#') or text.startswith('- ') or text.startswith('* '):
            score += 0.05
        
        # Date references
        if any(word in text_lower for word in ['bugün', 'yarın', 'hafta', 'ay', 'yıl']):
            score += 0.03
        
        # Cap score at 1.0
        return min(score, 1.0)
    
    def _classify_memory_type(self, text: str) -> str:
        """Memory türünü sınıflandır"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['karar', 'karar verdim', 'yapacağım', 'yapalım']):
            return 'decision'
        elif any(word in text_lower for word in ['öğrendim', 'ders', 'lesson', 'hata', 'çözüm']):
            return 'lesson'
        elif any(word in text_lower for word in ['tamamlandı', 'başardım', 'bitirdim', 'başarı']):
            return 'achievement'
        elif any(word in text_lower for word in ['tercih', 'seviyorum', 'severim', 'beğenirim']):
            return 'preference'
        elif any(word in text_lower for word in ['proje', 'sistem', 'entegre', 'kur', 'test']):
            return 'project'
        elif any(word in text_lower for word in ['retention', 'memory', 'teknik', 'kod', 'script']):
            return 'technical'
        elif any(word in text_lower for word in ['içgörü', 'insight', 'fark ettim', 'anladım ki']):
            return 'insight'
        elif any(word in text_lower for word in ['hatırlat', 'todo', 'not', 'yapılacak']):
            return 'reminder'
        else:
            return 'insight'  # Default
    
    def _extract_key_memories(self, daily_data: Dict, threshold: float = 0.3) -> List[Dict]:
        """Daily memory'den önemli kısımları çıkar"""
        key_memories = []
        
        for section in daily_data['sections']:
            # Boş section'ları atla
            if not section.strip() or len(section.strip()) < 20:
                continue
            
            # Önemlilik skoru hesapla
            importance = self._calculate_importance_score(section)
            
            if importance >= threshold:
                # Memory türünü belirle
                memory_type = self._classify_memory_type(section)
                
                # Özet oluştur (ilk 150 karakter)
                summary = section[:150].strip()
                if len(section) > 150:
                    summary += "..."
                
                # Hash oluştur (duplicate detection için)
                content_hash = hashlib.md5(section.encode()).hexdigest()[:8]
                
                key_memories.append({
                    'date': daily_data['date_str'],
                    'type': memory_type,
                    'importance': importance,
                    'summary': summary,
                    'full_text': section[:500],  # Max 500 karakter
                    'hash': content_hash,
                    'word_count': len(section.split())
                })
        
        # Importance'a göre sırala
        key_memories.sort(key=lambda x: x['importance'], reverse=True)
        
        # Max 5 memory (en önemliler)
        return key_memories[:5]
    
    def _format_memory_entry(self, memory: Dict) -> str:
        """Memory entry'sini formatla"""
        emoji = self.memory_types[memory['type']]['emoji']
        date_str = memory['date']
        
        entry = f"\n### {emoji} {date_str} - {memory['type'].upper()}\n"
        entry += f"**Önem:** {memory['importance']:.2f} | **Kelime:** {memory['word_count']}\n\n"
        entry += f"{memory['full_text']}\n"
        entry += f"\n---\n"
        
        return entry
    
    def _read_long_term_memory(self) -> List[str]:
        """Long-term memory'yi oku"""
        if not self.long_term_file.exists():
            return []
        
        try:
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Memory entry'lerini ayır
            entries = []
            current_entry = []
            in_entry = False
            
            for line in content.split('\n'):
                if line.startswith('### '):
                    if current_entry:
                        entries.append('\n'.join(current_entry))
                        current_entry = []
                    in_entry = True
                
                if in_entry:
                    current_entry.append(line)
            
            if current_entry:
                entries.append('\n'.join(current_entry))
            
            return entries
        
        except Exception as e:
            print(f"❌ Error reading long-term memory: {e}")
            return []
    
    def _check_duplicate(self, new_memory: Dict, existing_entries: List[str]) -> bool:
        """Duplicate memory kontrolü"""
        new_hash = new_memory['hash']
        
        for entry in existing_entries:
            if new_hash in entry:
                return True
        
        return False
    
    def consolidate(self, days_back: int = 3, importance_threshold: float = 0.25):
        """Daily memory'leri long-term memory'ye konsolide et"""
        print(f"\n🔍 CONSOLIDATION BAŞLIYOR...")
        print(f"   • Days back: {days_back}")
        print(f"   • Importance threshold: {importance_threshold}")
        print(f"   • Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # 1. Gerekli dizinleri oluştur
        self._ensure_directories()
        
        # 2. Daily files'ları getir
        daily_files = self._get_daily_files(days_back)
        print(f"\n📂 Found {len(daily_files)} daily files:")
        for f in daily_files:
            print(f"   • {f.name}")
        
        if not daily_files:
            print("❌ No daily files found for consolidation")
            return
        
        # 3. Mevcut long-term memory'yi oku
        existing_entries = self._read_long_term_memory()
        print(f"\n📚 Existing long-term memories: {len(existing_entries)} entries")
        
        # 4. Her daily file için key memories çıkar
        all_key_memories = []
        consolidated_count = 0
        
        for daily_file in daily_files:
            print(f"\n📖 Processing: {daily_file.name}")
            
            daily_data = self._read_daily_file(daily_file)
            if not daily_data:
                continue
            
            print(f"   • Sections: {len(daily_data['sections'])}")
            print(f"   • Words: {daily_data['word_count']}")
            
            key_memories = self._extract_key_memories(daily_data, importance_threshold)
            print(f"   • Key memories found: {len(key_memories)}")
            
            for memory in key_memories:
                # Duplicate kontrolü
                if self._check_duplicate(memory, existing_entries):
                    print(f"     ⚠️  Duplicate skipped: {memory['summary'][:50]}...")
                    continue
                
                all_key_memories.append(memory)
                consolidated_count += 1
                print(f"     ✅ Added: {memory['type']} - {memory['summary'][:50]}...")
        
        if not all_key_memories:
            print("\n❌ No new key memories to consolidate")
            return
        
        # 5. Yeni memory'leri long-term file'a ekle
        print(f"\n📝 Writing {consolidated_count} new memories to long-term storage...")
        
        # Mevcut content'i oku
        existing_content = ""
        if self.long_term_file.exists():
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Yeni memory'leri ekle
        new_content = "# 🧠 LONG-TERM MEMORY - HACI\n\n"
        new_content += f"*Last consolidated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        new_content += f"*Total memories: {len(existing_entries) + consolidated_count}*\n\n"
        
        # Memory türlerine göre grupla
        memories_by_type = {}
        for memory in all_key_memories:
            mem_type = memory['type']
            if mem_type not in memories_by_type:
                memories_by_type[mem_type] = []
            memories_by_type[mem_type].append(memory)
        
        # Her tür için section oluştur
        for mem_type, memories in memories_by_type.items():
            emoji = self.memory_types[mem_type]['emoji']
            desc = self.memory_types[mem_type]['desc']
            
            new_content += f"## {emoji} {desc}\n\n"
            
            for memory in memories:
                new_content += self._format_memory_entry(memory)
        
        # Eski content'i ekle (yeni memory'lerden sonra)
        if existing_content:
            # Header'ı atla
            lines = existing_content.split('\n')
            # İlk 5 satırı atla (header)
            old_content_without_header = '\n'.join(lines[5:]) if len(lines) > 5 else existing_content
            new_content += "\n\n## 📜 PREVIOUS MEMORIES\n\n"
            new_content += old_content_without_header
        
        # File'a yaz
        with open(self.long_term_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"\n✅ CONSOLIDATION COMPLETED!")
        print(f"   • New memories added: {consolidated_count}")
        print(f"   • Total memory types: {len(memories_by_type)}")
        print(f"   • Long-term file: {self.long_term_file}")
        
        # 6. Statistics
        total_words = sum(m['word_count'] for m in all_key_memories)
        avg_importance = sum(m['importance'] for m in all_key_memories) / len(all_key_memories) if all_key_memories else 0
        
        print(f"\n📊 STATISTICS:")
        print(f"   • Total words consolidated: {total_words}")
        print(f"   • Average importance: {avg_importance:.2f}")
        print(f"   • Memory type distribution:")
        
        for mem_type, memories in memories_by_type.items():
            emoji = self.memory_types[mem_type]['emoji']
            print(f"     {emoji} {mem_type}: {len(memories)} memories")
        
        # 7. Archive eski daily files (30 günden eski)
        self._archive_old_files(30)
    
    def _archive_old_files(self, days_old: int = 30):
        """Eski daily files'ları archive et"""
        print(f"\n🗃️  Archiving files older than {days_old} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_count = 0
        
        for filepath in self.memory_dir.glob("*.md"):
            if filepath.name == "MEMORY.md":
                continue
            
            try:
                # Dosya adından tarihi al
                date_str = filepath.stem
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    # Archive et
                    archive_path = self.archive_dir / filepath.name
                    filepath.rename(archive_path)
                    archived_count += 1
                    print(f"   ✅ Archived: {filepath.name}")
            
            except ValueError:
                # Geçersiz dosya adı
                continue
        
        print(f"   • Total archived: {archived_count}")
    
    def run_scheduled_consolidation(self):
        """Zamanlanmış consolidation çalıştır (cron job için)"""
        print("\n" + "="*60)
        print("🕒 SCHEDULED MEMORY CONSOLIDATION")
        print("="*60)
        
        # Her gün 23:55'te çalışacak
        # Son 3 günü konsolide et
        # Importance threshold: 0.25
        
        try:
            self.consolidate(days_back=3, importance_threshold=0.25)
            
            # Success log
            log_entry = f"{datetime.now().isoformat()} - Consolidation completed successfully\n"
            log_file = self.memory_dir / "consolidation.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print("\n✅ Scheduled consolidation completed and logged")
            
        except Exception as e:
            error_msg = f"{datetime.now().isoformat()} - Consolidation error: {str(e)}\n"
            log_file = self.memory_dir / "consolidation.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
            
            print(f"\n❌ Consolidation error: {e}")

def test_consolidation():
    """Test konsolidasyonu"""
    print("\n" + "="*60)
    print("🧪 MEMORY CONSOLIDATION TEST")
    print("="*60)
    
    consolidator = MemoryConsolidator()
    
    # Test için örnek daily memory oluştur
    test_date = datetime.now().strftime("%Y-%m-%d")
    test_file = Path("memory") / f"{test_date}.md"
    
    test_content = f"""# {test_date}

## ÖNEMLİ KARAR
Bugün retention system için otomatik memory consolidation kararı aldık.
Başkan "otomatikleştirelim" dedi, hemen implemente ediyorum.

## TEKNİK BİLGİ
MultiScaleRetention layer kuruldu, exponential decay 0.92.
Token tasarrufu: %56.8, hız artışı: 2.3x.

## ÖĞRENİLEN DERS
Tiny GPT-2 model çok küçük, anlamlı output üretemiyor.
Bunun yerine retention-based context compression kullanıyoruz.

## PROJE DURUMU
Hacı Kripto Telegram bot + retention system tamamlandı.
Optimized Haci Assistant token tasarruflu çalışıyor.

## TERCIH
Başkan kahve seviyor, özellikle espresso.
Çay da severim, yeşil çay özellikle.

## HATIRLATICI
Yarın Hacı Kripto'yu production'da test et.
"""
    
    # Test dosyasını oluştur
    Path("memory").mkdir(exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"📝 Created test daily file: {test_file}")
    
    # Konsolidasyonu çalıştır
    consolidator.consolidate(days_back=1, importance_threshold=0.2)
    
    # MEMORY.md'yi göster
    if Path("MEMORY.md").exists():
        print("\n" + "="*60)
        print("📖 CONSOLIDATED MEMORY.MD (first 500 chars):")
        print("="*60)
        
        with open("MEMORY.md", 'r', encoding='utf-8') as f:
            content = f.read()
            print(content[:500] + "...")
    
    print("\n" + "="*60)
    print("🎉 CONSOLIDATION TEST TAMAMLANDI!")
    print("="*60)

if __name__ == "__main__":
    test_consolidation()
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
        
        # Memory türleri ve emojileri + ağırlıkları
        self.memory_types = {
            'decision': {'emoji': '🤔', 'desc': 'Kararlar', 'base_weight': 1.2},
            'lesson': {'emoji': '📚', 'desc': 'Öğrenilen Dersler', 'base_weight': 1.1},
            'achievement': {'emoji': '🏆', 'desc': 'Başarılar', 'base_weight': 1.15},
            'preference': {'emoji': '👤', 'desc': 'Tercihler', 'base_weight': 1.0},
            'project': {'emoji': '🚀', 'desc': 'Projeler', 'base_weight': 1.05},
            'technical': {'emoji': '🔧', 'desc': 'Teknik Bilgiler', 'base_weight': 0.95},
            'insight': {'emoji': '💡', 'desc': 'İçgörüler', 'base_weight': 1.0},
            'reminder': {'emoji': '⏰', 'desc': 'Hatırlatıcılar', 'base_weight': 0.9}
        }
        
        # Başkan'ın ilgi alanları (learning_topics.json'dan yüklenecek)
        self.user_interests = {}
        self._load_user_interests()
        
        print("🧠 AUTOMATIC MEMORY CONSOLIDATOR")
        print("=" * 50)
        print(f"Memory dir: {self.memory_dir}")
        print(f"Long-term file: {self.long_term_file}")
        print(f"Archive dir: {self.archive_dir}")
        print(f"User interests loaded: {len(self.user_interests)} topics")
    
    def _load_user_interests(self):
        """learning_topics.json'dan Başkan'ın ilgi alanlarını yükle"""
        try:
            topics_file = Path("learning_topics.json")
            if topics_file.exists():
                with open(topics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for interest in data.get('interests', []):
                    topic = interest.get('topic', '').lower()
                    level = interest.get('interest_level', 0.5)
                    self.user_interests[topic] = level
                    
                    # Subtopic'leri de ekle
                    for subtopic in interest.get('subtopics', []):
                        subtopic_lower = subtopic.lower()
                        self.user_interests[subtopic_lower] = level * 0.8  # Subtopic biraz daha düşük
                
                print(f"✅ Loaded {len(self.user_interests)} interest keywords")
        except Exception as e:
            print(f"⚠️  Could not load user interests: {e}")
    
    def _process_feedback_queue(self) -> List[Dict]:
        """Process feedback queue and convert to memory entries"""
        feedback_file = self.memory_dir / "feedback_queue.json"
        
        if not feedback_file.exists():
            return []
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            feedbacks = data.get('feedbacks', [])
            memories = []
            
            print(f"\n🎯 Processing {len(feedbacks)} feedback entries from queue...")
            
            for fb in feedbacks:
                # Convert feedback to memory entry
                memory = {
                    'date': fb.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'type': fb.get('type', 'insight'),
                    'importance': 0.6,  # Feedback'ler higher priority
                    'summary': fb.get('title', 'Feedback memory'),
                    'full_text': fb.get('content', ''),
                    'hash': hashlib.md5(fb.get('content', '').encode()).hexdigest()[:8],
                    'word_count': len(fb.get('content', '').split()),
                    'keyword_count': 0,
                    'source': 'feedback',
                    'feedback_category': fb.get('category', 'unknown'),
                    'feedback_keyword': fb.get('keyword', '')
                }
                memories.append(memory)
            
            # Clear queue after processing
            with open(feedback_file, 'w') as f:
                json.dump({"feedbacks": []}, f)
            
            print(f"   ✅ Converted {len(memories)} feedback memories")
            return memories
            
        except Exception as e:
            print(f"❌ Error processing feedback queue: {e}")
            return []
    
    def _load_user_interests(self):
        """learning_topics.json'dan Başkan'ın ilgi alanlarını yükle"""
        try:
            topics_file = Path("learning_topics.json")
            if topics_file.exists():
                with open(topics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for interest in data.get('interests', []):
                    topic = interest.get('topic', '').lower()
                    level = interest.get('interest_level', 0.5)
                    self.user_interests[topic] = level
                    
                    # Subtopic'leri de ekle
                    for subtopic in interest.get('subtopics', []):
                        subtopic_lower = subtopic.lower()
                        self.user_interests[subtopic_lower] = level * 0.8  # Subtopic biraz daha düşük
                
                print(f"✅ Loaded {len(self.user_interests)} interest keywords")
        except Exception as e:
            print(f"⚠️  Could not load user interests: {e}")
    
    def _calculate_enhanced_importance(self, text: str, date_str: str = None) -> Dict:
        """Gelişmiş importance hesaplama: Context + Timing + Relevance"""
        text_lower = text.lower()
        scores = {
            'keyword': 0.0,
            'temporal': 0.0,
            'interest_match': 0.0,
            'structure': 0.0,
            'recency': 0.0,
            'type_weight': 1.0
        }
        
        # 1. KEYWORD MATCHING (Base importance)
        keyword_count = 0
        for keyword in self.importance_keywords:
            if keyword in text_lower:
                scores['keyword'] += 0.05
                keyword_count += 1
                if keyword in ['karar', 'önemli', 'kritik', 'lesson learned', 'ders']:
                    scores['keyword'] += 0.1
        
        # Keyword diversity bonus (çok fazla aynı keyword yerine çeşitlilik)
        unique_keywords = len(set(k for k in self.importance_keywords if k in text_lower))
        scores['keyword'] += min(unique_keywords * 0.02, 0.15)
        
        # 2. TEMPORAL FACTORS (Date references)
        temporal_keywords = ['bugün', 'yarın', 'hafta', 'ay', 'yıl', 'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar']
        if any(word in text_lower for word in temporal_keywords):
            scores['temporal'] += 0.1
        
        # Deadline/time-sensitive content
        if any(word in text_lower for word in ['saat', 'deadline', 'tarih', 'randevu', 'hatırlat']):
            scores['temporal'] += 0.15
        
        # 3. USER INTEREST ALIGNMENT
        interest_score = 0.0
        matched_topics = []
        for topic, level in self.user_interests.items():
            if topic and len(topic) > 2:  # Skip empty/short topics
                if topic in text_lower:
                    interest_score += level * 0.3
                    matched_topics.append(topic)
        
        # Boost if multiple interest matches
        if len(matched_topics) > 1:
            interest_score += 0.1
        
        scores['interest_match'] = min(interest_score, 0.5)  # Cap at 0.5
        
        # 4. STRUCTURE FACTORS
        words = text.split()
        word_count = len(words)
        
        # Optimal length (10-100 words)
        if 10 <= word_count <= 100:
            scores['structure'] += 0.1
        elif word_count > 100:
            scores['structure'] += 0.05  # Longer content still valuable
        
        # Formatting indicators
        if text.startswith('#') or text.startswith('- ') or text.startswith('* '):
            scores['structure'] += 0.05
        
        # Contains specific details, numbers, code snippets
        if any(char.isdigit() for char in text):
            scores['structure'] += 0.03
        
        # 5. RECENCY FACTOR (if date_str provided)
        if date_str:
            try:
                memory_date = datetime.strptime(date_str, "%Y-%m-%d")
                days_old = (datetime.now() - memory_date).days
                
                # Exponential decay: newer memories get higher score
                # 0 days old: +0.2, 3 days old: +0.1, 7+ days old: +0
                if days_old == 0:
                    scores['recency'] = 0.2
                elif days_old <= 3:
                    scores['recency'] = 0.1
                elif days_old <= 7:
                    scores['recency'] = 0.05
                else:
                    scores['recency'] = 0.0
            except:
                scores['recency'] = 0.0
        
        # 6. COMBINE ALL SCORES
        base_score = (
            scores['keyword'] * 0.35 +
            scores['temporal'] * 0.15 +
            scores['interest_match'] * 0.25 +
            scores['structure'] * 0.15 +
            scores['recency'] * 0.10
        )
        
        # Apply memory type weight (will be set by caller)
        final_score = base_score * scores['type_weight']
        
        return {
            'total_score': min(final_score, 1.0),
            'breakdown': scores,
            'matched_topics': matched_topics,
            'word_count': word_count,
            'keyword_count': keyword_count
        }
    
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
        """Daily memory'den önemli kısımları çıkar (enhanced algorithm)"""
        key_memories = []
        
        for section in daily_data['sections']:
            # Boş section'ları atla
            if not section.strip() or len(section.strip()) < 20:
                continue
            
            # Memory türünü belirle
            memory_type = self._classify_memory_type(section)
            
            # Get type weight
            type_weight = self.memory_types[memory_type]['base_weight']
            
            # Enhanced importance hesapla
            importance_data = self._calculate_enhanced_importance(
                section, 
                date_str=daily_data['date_str']
            )
            
            # Apply type weight
            importance_data['total_score'] *= type_weight
            importance_data['type_weight'] = type_weight
            
            importance = importance_data['total_score']
            
            if importance >= threshold:
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
                    'importance_breakdown': importance_data['breakdown'],
                    'matched_topics': importance_data['matched_topics'],
                    'summary': summary,
                    'full_text': section[:500],  # Max 500 karakter
                    'hash': content_hash,
                    'word_count': len(section.split()),
                    'keyword_count': importance_data['keyword_count']
                })
        
        # Importance'a göre sırala (en yüksek önce)
        key_memories.sort(key=lambda x: x['importance'], reverse=True)
        
        # Max 8 memory (enhanced: biraz daha çok memory alabiliriz)
        return key_memories[:8]
    
    def _format_memory_entry(self, memory: Dict) -> str:
        """Memory entry'sini formatla (enhanced with details)"""
        emoji = self.memory_types[memory['type']]['emoji']
        date_str = memory['date']
        
        # Check if it's a feedback memory
        is_feedback = memory.get('source') == 'feedback'
        
        entry = f"\n### {emoji} {date_str} - {memory['type'].upper()}"
        if is_feedback:
            entry += " (FEEDBACK)"
        entry += "\n"
        
        entry += f"**Önem:** {memory['importance']:.2f} | **Kelime:** {memory['word_count']}"
        
        if memory.get('keyword_count', 0) > 0:
            entry += f" | **Anahtar:** {memory['keyword_count']}"
        
        # Feedback-specific info
        if is_feedback:
            entry += f"\n**Feedback:** {memory.get('feedback_category', 'unknown')} - '{memory.get('feedback_keyword', '')}'"
        
        # Interest match varsa göster
        if memory.get('matched_topics'):
            topics_str = ', '.join(memory['matched_topics'][:3])  # Max 3 topic göster
            entry += f"\n**İlgi Alanı:** {topics_str}"
        
        # Important factors breakdown (concise)
        if memory.get('importance_breakdown'):
            breakdown = memory['importance_breakdown']
            factors = []
            if breakdown.get('keyword', 0) > 0.1:
                factors.append(f"keyword:{breakdown['keyword']:.2f}")
            if breakdown.get('interest_match', 0) > 0.1:
                factors.append(f"interest:{breakdown['interest_match']:.2f}")
            if breakdown.get('temporal', 0) > 0.05:
                factors.append(f"time:{breakdown['temporal']:.2f}")
            if breakdown.get('structure', 0) > 0.05:
                factors.append(f"struct:{breakdown['structure']:.2f}")
            if breakdown.get('recency', 0) > 0.05:
                factors.append(f"new:{breakdown['recency']:.2f}")
            
            if factors:
                entry += f"\n**Factors:** {' | '.join(factors)}"
        
        entry += f"\n\n{memory['full_text']}\n"
        entry += f"\n---\n"
        
        return entry
    
    def _read_long_term_memory(self) -> List[str]:
        """Long-term memory'yi oku ve her memory entry'nin hash'ini extract et"""
        if not self.long_term_file.exists():
            return []
        
        try:
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Memory entry'lerini ayır (### ile başlayanlar)
            entries = []
            lines = content.split('\n')
            current_entry = []
            in_entry = False
            
            for line in lines:
                if line.startswith('### '):
                    # Yeni entry başlıyor, önceki entry'yi kaydet
                    if current_entry and in_entry:
                        entry_text = '\n'.join(current_entry)
                        entries.append(entry_text)
                    # Yeni entry başlat
                    current_entry = [line]
                    in_entry = True
                elif in_entry and line.strip():
                    current_entry.append(line)
            
            # Son entry'yi ekle
            if current_entry and in_entry:
                entry_text = '\n'.join(current_entry)
                entries.append(entry_text)
            
            print(f"   📖 Parsed {len(entries)} entries from long-term memory")
            return entries
        
        except Exception as e:
            print(f"❌ Error reading long-term memory: {e}")
            return []
    
    def _check_duplicate(self, new_memory: Dict, existing_entries: List[str]) -> bool:
        """Duplicate memory kontrolü - full entry formatını hash ile karşılaştır"""
        # New memory'yi format et
        new_entry_str = self._format_memory_entry(new_memory)
        new_hash = hashlib.md5(new_entry_str.encode()).hexdigest()[:12]
        
        # Eski entry'lerin hash'lerini hesapla
        for entry in existing_entries:
            entry_hash = hashlib.md5(entry.encode()).hexdigest()[:12]
            if new_hash == entry_hash:
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
        
        # 2. Process feedback queue first (higher priority)
        feedback_memories = self._process_feedback_queue()
        
        # 3. Daily files'ları getir
        daily_files = self._get_daily_files(days_back)
        print(f"\n📂 Found {len(daily_files)} daily files:")
        for f in daily_files:
            print(f"   • {f.name}")
        
        if not daily_files and not feedback_memories:
            print("❌ No daily files or feedback memories found for consolidation")
            return
        
        # 4. Mevcut long-term memory'yi oku
        existing_entries = self._read_long_term_memory()
        print(f"\n📚 Existing long-term memories: {len(existing_entries)} entries")
        
        # 5. Feedback memory'leri ekle (lower threshold)
        all_key_memories = []
        consolidated_count = 0
        
        if feedback_memories:
            print(f"\n🎯 Processing {len(feedback_memories)} feedback memories...")
            for memory in feedback_memories:
                # Duplicate kontrolü
                if self._check_duplicate(memory, existing_entries):
                    print(f"     ⚠️  Duplicate feedback skipped")
                    continue
                
                all_key_memories.append(memory)
                consolidated_count += 1
                print(f"     ✅ Added feedback: {memory['type']} - {memory['summary'][:50]}...")
        
        # 6. Her daily file için key memories çıkar
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
        
        # 7. Yeni memory'leri long-term file'a ekle
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
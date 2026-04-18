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

    def __init__(self, memory_dir: str = None, long_term_file: str = None):
        base = Path(os.environ.get("HACI_MEMORY_PATH", Path.home() / ".haci-memory"))
        self.memory_dir = Path(memory_dir) if memory_dir else base / "memory"
        self.long_term_file = Path(long_term_file) if long_term_file else base / "MEMORY.md"
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
            'decision':    {'emoji': '🤔', 'desc': 'Kararlar',          'base_weight': 1.2},
            'lesson':      {'emoji': '📚', 'desc': 'Öğrenilen Dersler', 'base_weight': 1.1},
            'achievement': {'emoji': '🏆', 'desc': 'Başarılar',         'base_weight': 1.15},
            'preference':  {'emoji': '👤', 'desc': 'Tercihler',         'base_weight': 1.0},
            'project':     {'emoji': '🚀', 'desc': 'Projeler',          'base_weight': 1.05},
            'technical':   {'emoji': '🔧', 'desc': 'Teknik Bilgiler',   'base_weight': 0.95},
            'insight':     {'emoji': '💡', 'desc': 'İçgörüler',         'base_weight': 1.0},
            'reminder':    {'emoji': '⏰', 'desc': 'Hatırlatıcılar',    'base_weight': 0.9},
        }

        self.user_interests = {}
        self._load_user_interests()

    def _load_user_interests(self):
        topics_file = self.memory_dir.parent / "learning_topics.json"
        if not topics_file.exists():
            topics_file = Path("learning_topics.json")
        try:
            if topics_file.exists():
                data = json.loads(topics_file.read_text(encoding='utf-8'))
                for interest in data.get('interests', []):
                    topic = interest.get('topic', '').lower()
                    level = interest.get('interest_level', 0.5)
                    self.user_interests[topic] = level
                    for subtopic in interest.get('subtopics', []):
                        self.user_interests[subtopic.lower()] = level * 0.8
        except Exception as e:
            print(f"⚠️  Could not load user interests: {e}")

    def _process_feedback_queue(self) -> List[Dict]:
        feedback_file = self.memory_dir / "feedback_queue.json"
        if not feedback_file.exists():
            return []
        try:
            data = json.loads(feedback_file.read_text(encoding='utf-8'))
            feedbacks = data.get('feedbacks', [])
            memories = []
            for fb in feedbacks:
                memory = {
                    'date': fb.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'type': fb.get('type', 'insight'),
                    'importance': 0.6,
                    'summary': fb.get('title', 'Feedback memory'),
                    'full_text': fb.get('content', ''),
                    'hash': hashlib.md5(fb.get('content', '').encode()).hexdigest()[:8],
                    'word_count': len(fb.get('content', '').split()),
                    'keyword_count': 0,
                    'source': 'feedback',
                    'feedback_category': fb.get('category', 'unknown'),
                    'feedback_keyword': fb.get('keyword', ''),
                }
                memories.append(memory)
            feedback_file.write_text(json.dumps({"feedbacks": []}, ensure_ascii=False))
            return memories
        except Exception as e:
            print(f"❌ Error processing feedback queue: {e}")
            return []

    def _calculate_enhanced_importance(self, text: str, date_str: str = None) -> Dict:
        text_lower = text.lower()
        scores = {'keyword': 0.0, 'temporal': 0.0, 'interest_match': 0.0,
                  'structure': 0.0, 'recency': 0.0, 'type_weight': 1.0}
        keyword_count = 0
        for keyword in self.importance_keywords:
            if keyword in text_lower:
                scores['keyword'] += 0.05
                keyword_count += 1
                if keyword in ['karar', 'önemli', 'kritik', 'lesson learned', 'ders']:
                    scores['keyword'] += 0.1
        unique_keywords = len(set(k for k in self.importance_keywords if k in text_lower))
        scores['keyword'] += min(unique_keywords * 0.02, 0.15)

        temporal_words = ['bugün', 'yarın', 'hafta', 'ay', 'yıl', 'pazartesi', 'salı',
                          'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar']
        if any(w in text_lower for w in temporal_words):
            scores['temporal'] += 0.1
        if any(w in text_lower for w in ['saat', 'deadline', 'tarih', 'randevu', 'hatırlat']):
            scores['temporal'] += 0.15

        interest_score = 0.0
        matched_topics = []
        for topic, level in self.user_interests.items():
            if topic and len(topic) > 2 and topic in text_lower:
                interest_score += level * 0.3
                matched_topics.append(topic)
        if len(matched_topics) > 1:
            interest_score += 0.1
        scores['interest_match'] = min(interest_score, 0.5)

        words = text.split()
        word_count = len(words)
        if 10 <= word_count <= 100:
            scores['structure'] += 0.1
        elif word_count > 100:
            scores['structure'] += 0.05
        if text.startswith('#') or text.startswith('- ') or text.startswith('* '):
            scores['structure'] += 0.05
        if any(c.isdigit() for c in text):
            scores['structure'] += 0.03

        if date_str:
            try:
                days_old = (datetime.now() - datetime.strptime(date_str, "%Y-%m-%d")).days
                scores['recency'] = 0.2 if days_old == 0 else (0.1 if days_old <= 3 else (0.05 if days_old <= 7 else 0.0))
            except Exception:
                pass

        base_score = (scores['keyword'] * 0.35 + scores['temporal'] * 0.15 +
                      scores['interest_match'] * 0.25 + scores['structure'] * 0.15 +
                      scores['recency'] * 0.10)
        return {
            'total_score': min(base_score * scores['type_weight'], 1.0),
            'breakdown': scores,
            'matched_topics': matched_topics,
            'word_count': word_count,
            'keyword_count': keyword_count,
        }

    def _ensure_directories(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _get_daily_files(self, days_back: int = 7) -> List[Path]:
        daily_files = []
        today = datetime.now()
        for i in range(days_back):
            fp = self.memory_dir / (today - timedelta(days=i)).strftime("%Y-%m-%d.md")
            if fp.exists():
                daily_files.append(fp)
        return daily_files

    def _read_daily_file(self, filepath: Path) -> Dict:
        try:
            content = filepath.read_text(encoding='utf-8')
            date_str = filepath.stem
            date = datetime.strptime(date_str, "%Y-%m-%d")
            lines = content.split('\n')
            sections, current_section = [], []
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                if line:
                    current_section.append(line)
            if current_section:
                sections.append('\n'.join(current_section))
            return {'date': date, 'date_str': date_str, 'filepath': filepath,
                    'content': content, 'sections': sections,
                    'line_count': len(lines), 'word_count': len(content.split())}
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return None

    def _classify_memory_type(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ['karar', 'karar verdim', 'yapacağım', 'yapalım']):
            return 'decision'
        if any(w in t for w in ['öğrendim', 'ders', 'lesson', 'hata', 'çözüm']):
            return 'lesson'
        if any(w in t for w in ['tamamlandı', 'başardım', 'bitirdim', 'başarı']):
            return 'achievement'
        if any(w in t for w in ['tercih', 'seviyorum', 'severim', 'beğenirim']):
            return 'preference'
        if any(w in t for w in ['proje', 'sistem', 'entegre', 'kur', 'test']):
            return 'project'
        if any(w in t for w in ['retention', 'memory', 'teknik', 'kod', 'script']):
            return 'technical'
        if any(w in t for w in ['içgörü', 'insight', 'fark ettim', 'anladım ki']):
            return 'insight'
        if any(w in t for w in ['hatırlat', 'todo', 'not', 'yapılacak']):
            return 'reminder'
        return 'insight'

    def _extract_key_memories(self, daily_data: Dict, threshold: float = 0.3) -> List[Dict]:
        key_memories = []
        for section in daily_data['sections']:
            if not section.strip() or len(section.strip()) < 20:
                continue
            memory_type = self._classify_memory_type(section)
            type_weight = self.memory_types[memory_type]['base_weight']
            importance_data = self._calculate_enhanced_importance(section, daily_data['date_str'])
            importance_data['total_score'] *= type_weight
            importance = importance_data['total_score']
            if importance >= threshold:
                summary = section[:150].strip()
                if len(section) > 150:
                    summary += "..."
                key_memories.append({
                    'date': daily_data['date_str'],
                    'type': memory_type,
                    'importance': importance,
                    'importance_breakdown': importance_data['breakdown'],
                    'matched_topics': importance_data['matched_topics'],
                    'summary': summary,
                    'full_text': section[:500],
                    'hash': hashlib.md5(section.encode()).hexdigest()[:8],
                    'word_count': len(section.split()),
                    'keyword_count': importance_data['keyword_count'],
                })
        key_memories.sort(key=lambda x: x['importance'], reverse=True)
        return key_memories[:8]

    def _format_memory_entry(self, memory: Dict) -> str:
        emoji = self.memory_types[memory['type']]['emoji']
        is_feedback = memory.get('source') == 'feedback'
        entry = f"\n### {emoji} {memory['date']} - {memory['type'].upper()}"
        if is_feedback:
            entry += " (FEEDBACK)"
        entry += f"\n**Önem:** {memory['importance']:.2f} | **Kelime:** {memory['word_count']}"
        if memory.get('keyword_count', 0) > 0:
            entry += f" | **Anahtar:** {memory['keyword_count']}"
        if is_feedback:
            entry += f"\n**Feedback:** {memory.get('feedback_category')} - '{memory.get('feedback_keyword')}'"
        if memory.get('matched_topics'):
            entry += f"\n**İlgi Alanı:** {', '.join(memory['matched_topics'][:3])}"
        if memory.get('importance_breakdown'):
            bd = memory['importance_breakdown']
            factors = []
            if bd.get('keyword', 0) > 0.1:    factors.append(f"keyword:{bd['keyword']:.2f}")
            if bd.get('interest_match', 0) > 0.1: factors.append(f"interest:{bd['interest_match']:.2f}")
            if bd.get('temporal', 0) > 0.05:  factors.append(f"time:{bd['temporal']:.2f}")
            if bd.get('structure', 0) > 0.05: factors.append(f"struct:{bd['structure']:.2f}")
            if bd.get('recency', 0) > 0.05:   factors.append(f"new:{bd['recency']:.2f}")
            if factors:
                entry += f"\n**Factors:** {' | '.join(factors)}"
        entry += f"\n\n{memory['full_text']}\n\n---\n"
        return entry

    def _read_long_term_memory(self) -> List[str]:
        if not self.long_term_file.exists():
            return []
        try:
            content = self.long_term_file.read_text(encoding='utf-8')
            entries, current_entry, in_entry = [], [], False
            for line in content.split('\n'):
                if line.startswith('### '):
                    if current_entry and in_entry:
                        entries.append('\n'.join(current_entry))
                    current_entry, in_entry = [line], True
                elif in_entry and line.strip():
                    current_entry.append(line)
            if current_entry and in_entry:
                entries.append('\n'.join(current_entry))
            return entries
        except Exception as e:
            print(f"❌ Error reading long-term memory: {e}")
            return []

    def _check_duplicate(self, new_memory: Dict, existing_entries: List[str]) -> bool:
        new_hash = hashlib.md5(self._format_memory_entry(new_memory).encode()).hexdigest()[:12]
        return any(hashlib.md5(e.encode()).hexdigest()[:12] == new_hash for e in existing_entries)

    def consolidate(self, days_back: int = 3, importance_threshold: float = 0.25):
        print(f"\n🔍 CONSOLIDATION BAŞLIYOR — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self._ensure_directories()

        feedback_memories = self._process_feedback_queue()
        daily_files = self._get_daily_files(days_back)
        print(f"📂 {len(daily_files)} daily file, {len(feedback_memories)} feedback")

        if not daily_files and not feedback_memories:
            print("❌ Konsolide edilecek yeni içerik yok")
            return

        existing_entries = self._read_long_term_memory()
        print(f"📚 Mevcut: {len(existing_entries)} long-term entry")

        all_key_memories = []
        for memory in feedback_memories:
            if not self._check_duplicate(memory, existing_entries):
                all_key_memories.append(memory)

        for daily_file in daily_files:
            daily_data = self._read_daily_file(daily_file)
            if not daily_data:
                continue
            for memory in self._extract_key_memories(daily_data, importance_threshold):
                if not self._check_duplicate(memory, existing_entries):
                    all_key_memories.append(memory)

        if not all_key_memories:
            print("❌ Yeni bellek yok")
            return

        existing_content = self.long_term_file.read_text(encoding='utf-8') if self.long_term_file.exists() else ""
        new_content = "# 🧠 LONG-TERM MEMORY - HACI\n\n"
        new_content += f"*Last consolidated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        new_content += f"*Total memories: {len(existing_entries) + len(all_key_memories)}*\n\n"

        memories_by_type: Dict[str, list] = {}
        for m in all_key_memories:
            memories_by_type.setdefault(m['type'], []).append(m)

        for mem_type, memories in memories_by_type.items():
            mt = self.memory_types[mem_type]
            new_content += f"## {mt['emoji']} {mt['desc']}\n\n"
            for memory in memories:
                new_content += self._format_memory_entry(memory)

        if existing_content:
            lines = existing_content.split('\n')
            old_body = '\n'.join(lines[5:]) if len(lines) > 5 else existing_content
            new_content += "\n\n## 📜 PREVIOUS MEMORIES\n\n" + old_body

        self.long_term_file.write_text(new_content, encoding='utf-8')

        avg_imp = sum(m['importance'] for m in all_key_memories) / len(all_key_memories)
        print(f"\n✅ CONSOLIDATION TAMAMLANDI: {len(all_key_memories)} yeni bellek, ort. önem {avg_imp:.2f}")
        for mem_type, mems in memories_by_type.items():
            print(f"   {self.memory_types[mem_type]['emoji']} {mem_type}: {len(mems)}")

        self._archive_old_files(30)

    def _archive_old_files(self, days_old: int = 30):
        cutoff = datetime.now() - timedelta(days=days_old)
        count = 0
        for fp in self.memory_dir.glob("*.md"):
            try:
                if datetime.strptime(fp.stem, "%Y-%m-%d") < cutoff:
                    fp.rename(self.archive_dir / fp.name)
                    count += 1
            except ValueError:
                pass
        if count:
            print(f"🗃️  {count} dosya arşivlendi")

    def run_scheduled_consolidation(self):
        try:
            self.consolidate(days_back=3, importance_threshold=0.25)
            log = self.memory_dir / "consolidation.log"
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - OK\n")
        except Exception as e:
            log = self.memory_dir / "consolidation.log"
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - ERROR: {e}\n")
            print(f"❌ {e}")


if __name__ == "__main__":
    MemoryConsolidator().run_scheduled_consolidation()

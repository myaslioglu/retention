#!/usr/bin/env python3
"""
FEEDBACK CAPTURE SYSTEM
OpenClaw heartbeat'inde Başkan'ın feedback'lerini capture eder
ve memory consolidation queue'una ekler.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
import json

class FeedbackCapture:
    """Capture user feedback from conversations"""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.feedback_queue_file = self.memory_dir / "feedback_queue.json"
        self._ensure_files()
    
    def _ensure_files(self):
        """Ensure directory and files exist"""
        self.memory_dir.mkdir(exist_ok=True)
        if not self.feedback_queue_file.exists():
            with open(self.feedback_queue_file, 'w') as f:
                json.dump({"feedbacks": []}, f)
    
    def load_recent_conversations(self, limit: int = 20) -> list:
        """Load recent messages from current session (would need integration)"""
        # This would integrate with OpenClaw's message history
        # For now, we'll check daily memory files for feedback patterns
        conversations = []
        
        # Check recent daily files
        today = datetime.now()
        for i in range(3):  # Last 3 days
            date = today - timedelta(days=i)
            filepath = self.memory_dir / f"{date.strftime('%Y-%m-%d')}.md"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split by sections
                sections = content.split('## ')[1:]  # Skip header
                for section in sections:
                    lines = section.split('\n')[1:]  # Skip title line
                    text = '\n'.join(lines[:5])  # First 5 lines of content
                    if text.strip():
                        conversations.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'section': section.split('\n')[0],
                            'content': text[:200]
                        })
        
        return conversations[-limit:]  # Return most recent
    
    def analyze_feedback(self, text: str) -> dict:
        """Analyze text for feedback patterns"""
        text_lower = text.lower()
        
        # Feedback patterns
        patterns = {
            'positive': {
                'keywords': ['teşekkür', 'beğendim', 'mükemmel', 'çok iyi', 'harika', 'süper', 'iyi', 'güzel', 'doğru'],
                'type': 'positive',
                'memory_type': 'achievement',
                'emoji': '🏆'
            },
            'negative': {
                'keywords': ['yanlış', 'hata', 'kötü', 'düzelt', 'yanıltıcı', 'eksik', 'bayat', 'sıkıldım'],
                'type': 'negative',
                'memory_type': 'lesson',
                'emoji': '📚'
            },
            'suggestion': {
                'keywords': ['belki', 'ya da', 'daha iyi', 'değiştir', 'alternatif', 'tavsiye', 'öneri'],
                'type': 'suggestion',
                'memory_type': 'insight',
                'emoji': '💡'
            },
            'preference': {
                'keywords': ['severim', 'seviyorum', 'tercih', 'hoşuma gitti', 'kullanmayın', 'yapma'],
                'type': 'preference',
                'memory_type': 'preference',
                'emoji': '👤'
            }
        }
        
        detected = []
        
        for category, data in patterns.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    detected.append({
                        'category': category,
                        'keyword': keyword,
                        'memory_type': data['memory_type'],
                        'emoji': data['emoji']
                    })
                    break  # One keyword per category is enough
        
        return detected
    
    def create_feedback_memory(self, conversation: dict, feedback: dict) -> dict:
        """Create a memory entry from feedback"""
        category = feedback['category']
        keyword = feedback['keyword']
        memory_type = feedback['memory_type']
        
        # Extract context
        context = conversation['content']
        section = conversation['section']
        
        # Create memory content
        if category == 'positive':
            title = f"Olumlu Feedback: {keyword.title()}"
            content = f"Başkan'ın olumlu feedback'i alındı: '{keyword}' ifadesi kullanıldı.\n\nBağlam: {context}\n\nBu feedback, sistem performansını ve kullanıcı memnuniyetini gösteriyor."
        elif category == 'negative':
            title = f"Olumsuz Feedback: {keyword.title()}"
            content = f"Başkan'ın olumsuz feedback'i alındı: '{keyword}' ifadesi kullanıldı.\n\nBağlam: {context}\n\nBu feedback, iyileştirme gereken alanları gösteriyor."
        elif category == 'suggestion':
            title = f"Öneri Feedback: {keyword.title()}"
            content = f"Başkan'dan öneri alındı: '{keyword}' ifadesi kullanıldı.\n\nBağlam: {context}\n\nBu öneri, sistem geliştirme için fikir sağlıyor."
        else:  # preference
            title = f"Tercih Feedback: {keyword.title()}"
            content = f"Başkan'ın tercihini öğrendik: '{keyword}' ifadesi kullanıldı.\n\nBağlam: {context}\n\nBu tercih, kişiselleştirme için önemli."
        
        memory = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': memory_type,
            'title': title,
            'content': content,
            'category': category,
            'keyword': keyword,
            'context': context[:100],
            'section': section,
            'importance': 0.5,  # Will be calculated by consolidator
            'source': 'feedback_capture'
        }
        
        return memory
    
    def scan_and_capture(self) -> dict:
        """Scan recent conversations and capture feedback"""
        print("🔍 SCANNING FOR FEEDBACK...")
        
        conversations = self.load_recent_conversations(limit=30)
        print(f"   • Loaded {len(conversations)} recent conversation snippets")
        
        all_feedbacks = []
        for conv in conversations:
            feedbacks = self.analyze_feedback(conv['content'])
            
            for feedback in feedbacks:
                memory = self.create_feedback_memory(conv, feedback)
                all_feedbacks.append(memory)
                print(f"   ✅ Detected {feedback['category']}: {feedback['keyword']} in {conv['date']}")
        
        # Load existing queue
        with open(self.feedback_queue_file, 'r') as f:
            queue = json.load(f)
        
        # Add new feedbacks (avoid duplicates by content hash)
        existing_contents = {fb.get('content', '') for fb in queue['feedbacks']}
        new_count = 0
        
        for fb in all_feedbacks:
            if fb['content'] not in existing_contents:
                queue['feedbacks'].append(fb)
                new_count += 1
        
        # Save queue
        with open(self.feedback_queue_file, 'w') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ FEEDBACK CAPTURE COMPLETED!")
        print(f"   • Total feedbacks in queue: {len(queue['feedbacks'])}")
        print(f"   • New feedbacks added: {new_count}")
        
        return queue
    
    def get_feedback_queue(self) -> list:
        """Get pending feedback memories"""
        with open(self.feedback_queue_file, 'r') as f:
            queue = json.load(f)
        return queue['feedbacks']
    
    def clear_feedback_queue(self):
        """Clear feedback queue after consolidation"""
        with open(self.feedback_queue_file, 'w') as f:
            json.dump({"feedbacks": []}, f)
        print("✅ Feedback queue cleared")


def test_feedback_capture():
    """Test the feedback capture system"""
    print("🧪 TESTING FEEDBACK CAPTURE SYSTEM")
    print("=" * 60)
    
    # Create test conversations
    test_conversations = [
        {
            'date': '2026-03-06',
            'section': '## ÖNEMLİ KARAR',
            'content': 'Bugün retention system için otomatik memory consolidation kararı aldık. Bu mükemmel bir karardı. Teşekkür ederim.'
        },
        {
            'date': '2026-03-05',
            'section': '## TERCIH',
            'content': 'Kahve seviyorum, özellikle espresso. Ama yeşil çay sevmiyorum, daha çok çay severim.'
        },
        {
            'date': '2026-03-04',
            'section': '## PROJE DURUMU',
            'content': 'Retention sistemi çalışıyor ama hala yavaş. Belki daha hızlı olabilir. Ya da batch processing kullanabiliriz.'
        },
        {
            'date': '2026-03-03',
            'section': '## HATIRLATICI',
            'content': 'Yarın hatırlatma kur. Ama bu hatırlatma yanlış. Düzeltmen gerekiyor. Hata yaptım.'
        },
        {
            'date': '2026-03-02',
            'section': '## TEST SONUÇLARI',
            'content': 'Test sonuçları iyi çıktı. Her şey doğru çalışıyor. Mükemmel!'
        }
    ]
    
    print("\n📝 Test conversations loaded:")
    for conv in test_conversations:
        print(f"   • {conv['date']}: {conv['section']} - {conv['content'][:50]}...")
    
    # Test capture
    capture = FeedbackCapture()
    
    print("\n🔍 Analyzing feedback patterns...")
    detected_count = 0
    
    for conv in test_conversations:
        feedbacks = capture.analyze_feedback(conv['content'])
        if feedbacks:
            detected_count += len(feedbacks)
            for fb in feedbacks:
                memory = capture.create_feedback_memory(conv, fb)
                print(f"   ✓ {fb['emoji']} {fb['category'].upper()}: {fb['keyword']}")
    
    print(f"\n✅ Detected {detected_count} feedback instances")
    
    # Scan and capture
    print("\n🔄 Running full feedback capture...")
    queue = capture.scan_and_capture()
    
    print(f"\n📊 Final queue: {len(queue['feedbacks'])} feedback memories")
    
    print("\n" + "=" * 60)
    print("🎉 FEEDBACK CAPTURE TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_feedback_capture()
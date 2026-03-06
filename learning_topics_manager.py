#!/usr/bin/env python3
"""
INTELLIGENT LEARNING TOPICS MANAGER
Learning topics'i otomatik günceller, interest level'ı optimize eder,
ve weekly newsletter hazırlar.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

class LearningTopicsManager:
    """Learning topics manager with intelligent updates"""
    
    def __init__(self, topics_file: str = "learning_topics.json"):
        self.topics_file = Path(topics_file)
        self.data = {}
        self._load()
    
    def _load(self):
        """Load topics data"""
        if self.topics_file.exists():
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "version": "2.0",
                "last_updated": datetime.now().isoformat(),
                "learning_philosophy": "Çocuk gibi büyüme - her gün yeni şeyler öğren, meraklı ol, soru sor. Öğrenmeyi öğren - kendi öğrenme yöntemini geliştir. Zeka budur.",
                "interests": [],
                "questions_to_ask": [],
                "recent_discoveries": [],
                "learning_goals": [],
                "stats": {
                    "total_discoveries": 0,
                    "questions_asked": 0,
                    "topics_explored": 0,
                    "last_heartbeat_learning": None,
                    "weekly_newsletter_sent": False,
                    "last_newsletter_date": None
                }
            }
    
    def _save(self):
        """Save topics data"""
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_discovery(self, topic: str, subtopic: str = None, discovery: str = None, source: str = "web_search"):
        """Add a new discovery"""
        # Check if topic exists
        topic_lower = topic.lower()
        existing_interest = None
        for interest in self.data['interests']:
            if interest['topic'].lower() == topic_lower:
                existing_interest = interest
                break
        
        if existing_interest:
            # Update existing interest
            self._update_interest_level(existing_interest, boost=0.05)
            interest_level = existing_interest['interest_level']
        else:
            # Create new interest
            interest_level = 0.5  # Default medium interest
            new_interest = {
                "topic": topic,
                "subtopics": [],
                "interest_level": interest_level,
                "last_explored": datetime.now().strftime("%Y-%m-%d"),
                "discovery_count": 1,
                "sources": [source]
            }
            self.data['interests'].append(new_interest)
            existing_interest = new_interest
            self.data['stats']['topics_explored'] += 1
        
        # Add subtopic if provided
        if subtopic and subtopic not in existing_interest['subtopics']:
            existing_interest['subtopics'].append(subtopic)
        
        # Add to recent discoveries
        discovery_entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "topic": topic,
            "subtopic": subtopic,
            "discovery": discovery or f"Discovered interest in {topic}",
            "source": source
        }
        self.data['recent_discoveries'].insert(0, discovery_entry)
        
        # Keep only last 50 discoveries
        self.data['recent_discoveries'] = self.data['recent_discoveries'][:50]
        
        # Update stats
        self.data['stats']['total_discoveries'] += 1
        
        # Save
        self._save()
        
        print(f"✅ Added discovery: {topic} (interest: {interest_level:.2f})")
        return existing_interest
    
    def _update_interest_level(self, interest: dict, boost: float = 0.0, decay: float = 0.0):
        """Update interest level with boost/decay"""
        current = interest['interest_level']
        
        # Apply boost
        if boost > 0:
            current += boost
            # Cap at 1.0
            current = min(current, 1.0)
        
        # Apply decay (daily)
        if decay > 0:
            last_explored = datetime.strptime(interest['last_explored'], "%Y-%m-%d")
            days_ago = (datetime.now() - last_explored).days
            decay_factor = (1 - decay) ** days_ago
            current = max(0.3, current * decay_factor)  # Don't go below 0.3
        
        interest['interest_level'] = round(current, 2)
        interest['last_explored'] = datetime.now().strftime("%Y-%m-%d")
    
    def decay_interests(self, daily_decay: float = 0.05):
        """Apply daily decay to all interests"""
        for interest in self.data['interests']:
            self._update_interest_level(interest, decay=daily_decay)
        self._save()
        print(f"✅ Applied {daily_decay*100}% decay to all interests")
    
    def get_top_interests(self, n: int = 5) -> list:
        """Get top N interests by interest_level"""
        sorted_interests = sorted(self.data['interests'], 
                                key=lambda x: x['interest_level'], 
                                reverse=True)
        return sorted_interests[:n]
    
    def suggest_topic_to_explore(self) -> dict:
        """Suggest a topic to explore based on interest and recency"""
        # Get interests with high interest but not recently explored
        candidates = []
        for interest in self.data['interests']:
            last_explored = datetime.strptime(interest['last_explored'], "%Y-%m-%d")
            days_since = (datetime.now() - last_explored).days
            
            # Score = interest_level * (1 + days_since/30) - explore forgotten high-interest topics
            score = interest['interest_level'] * (1 + days_since/30)
            
            if interest['interest_level'] >= 0.6 and days_since >= 3:
                candidates.append({
                    'topic': interest['topic'],
                    'subtopics': interest['subtopics'],
                    'interest_level': interest['interest_level'],
                    'days_since': days_since,
                    'score': score
                })
        
        # Sort by score (high interest + long time not explored)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if candidates:
            return candidates[0]
        
        # If no good candidates, pick random high-interest topic
        top_interests = self.get_top_interests(5)
        if top_interests:
            import random
            chosen = random.choice(top_interests)
            return {
                'topic': chosen['topic'],
                'subtopics': chosen['subtopics'],
                'interest_level': chosen['interest_level'],
                'days_since': (datetime.now() - datetime.strptime(chosen['last_explored'], "%Y-%m-%d")).days,
                'score': chosen['interest_level']
            }
        
        return None
    
    def generate_weekly_newsletter(self) -> str:
        """Generate weekly learning newsletter"""
        # Check if we already sent this week
        today = datetime.now()
        if (today.weekday() == 6) and not self.data['stats'].get('weekly_newsletter_sent', False):
            # Sunday - send newsletter
            pass
        
        # Generate content
        newsletter = "🧠 **HAKI'NIN ÖĞRENME BULTENİ**\n\n"
        newsletter += f"*Tarih: {today.strftime('%Y-%m-%d')}*\n\n"
        
        # Top interests
        newsletter += "🔥 **EN YÜKSEK İLGİ ALANLARI:**\n"
        top_interests = self.get_top_interests(5)
        for i, interest in enumerate(top_interests, 1):
            newsletter += f"{i}. **{interest['topic']}** (İlgi: {interest['interest_level']:.0%})\n"
            if interest['subtopics']:
                newsletter += f"   └─ {', '.join(interest['subtopics'][:3])}\n"
        
        newsletter += "\n"
        
        # Recent discoveries
        newsletter += "📚 **BU HAFTA KEŞFEDİLENLER:**\n"
        recent = self.data['recent_discoveries'][:5]
        for discovery in recent:
            date = discovery['date']
            topic = discovery['topic']
            subtopic = discovery.get('subtopic', '')
            source = discovery['source']
            
            newsletter += f"• **{topic}**"
            if subtopic:
                newsletter += f" ({subtopic})"
            newsletter += f" - {date} [{source}]\n"
        
        newsletter += "\n"
        
        # Learning stats
        stats = self.data['stats']
        newsletter += "📊 **İSTATİSTİKLER:**\n"
        newsletter += f"• Toplam keşif: {stats['total_discoveries']}\n"
        newsletter += f"• Soru soruldu: {stats['questions_asked']}\n"
        newsletter += f"• Konular keşfedildi: {stats['topics_explored']}\n"
        newsletter += f"• Öğrenme felsefesi: {self.data['learning_philosophy'][:100]}...\n"
        
        newsletter += "\n---\n"
        newsletter += "🚀 *Gelecek hafta keşfetmek için öneri:* "
        
        suggestion = self.suggest_topic_to_explore()
        if suggestion:
            newsletter += f"**{suggestion['topic']}** (Son keşif: {suggestion['days_since']} gün önce)"
        else:
            newsletter += "Rastgele bir teknolohi konusu araştır!"
        
        return newsletter
    
    def update_learning_goal(self, new_goal: str):
        """Add or update a learning goal"""
        if new_goal not in self.data['learning_goals']:
            self.data['learning_goals'].append(new_goal)
            self._save()
            print(f"✅ Added learning goal: {new_goal}")
    
    def mark_newsletter_sent(self):
        """Mark weekly newsletter as sent"""
        self.data['stats']['weekly_newsletter_sent'] = True
        self.data['stats']['last_newsletter_date'] = datetime.now().strftime("%Y-%m-%d")
        self._save()
    
    def reset_weekly_flags(self):
        """Reset weekly flags (run on Monday)"""
        self.data['stats']['weekly_newsletter_sent'] = False
        self._save()
        print("✅ Reset weekly newsletter flag")


def test_manager():
    """Test the learning topics manager"""
    print("🧪 TESTING INTELLIGENT LEARNING TOPICS MANAGER")
    print("=" * 60)
    
    manager = LearningTopicsManager()
    
    # Add some test discoveries
    print("\n📝 Adding test discoveries...")
    manager.add_discovery(
        topic="AI Safety",
        subtopic="alignment problem",
        discovery="AI alignment is about ensuring AI systems act in accordance with human values",
        source="web_search"
    )
    manager.add_discovery(
        topic="Retention Models",
        subtopic="MultiScaleRetention",
        discovery="MultiScaleRetention layer improves long-context modeling with state decay",
        source="technical_testing"
    )
    manager.add_discovery(
        topic="Turkish NLP",
        subtopic="BERT models",
        discovery="Turkish BERT models show 92% accuracy on sentiment analysis",
        source="web_search"
    )
    
    # Test decay
    print("\n📉 Testing interest decay...")
    manager.decay_interests(daily_decay=0.05)
    
    # Get top interests
    print("\n🏆 Top interests:")
    top = manager.get_top_interests(3)
    for interest in top:
        print(f"   • {interest['topic']} ({interest['interest_level']:.0%})")
    
    # Suggest topic
    print("\n💡 Suggestion for next exploration:")
    suggestion = manager.suggest_topic_to_explore()
    if suggestion:
        print(f"   Topic: {suggestion['topic']}")
        print(f"   Interest: {suggestion['interest_level']:.0%}")
        print(f"   Last explored: {suggestion['days_since']} days ago")
    
    # Generate newsletter
    print("\n📧 Weekly Newsletter Preview:")
    print("-" * 50)
    newsletter = manager.generate_weekly_newsletter()
    print(newsletter)
    
    print("\n" + "=" * 60)
    print("🎉 LEARNING TOPICS MANAGER TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_manager()
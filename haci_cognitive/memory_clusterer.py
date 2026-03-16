"""
Memory Clusterer - Anıları otomatik topic'lere grupla

Semantic clustering ile memory'leri kategorilere ayırır:
- Aile (Duygu, Ada, aile konuları)
- Teknik (OpenClaw, Python, sistem yönetimi)
- Proje (retention, cognitive system, entegrasyonlar)
- Kişisel (hobiler, tercihler, günlük hayat)
- Spor (Galatasaray, maç sonuçları)
- İş (randevular, toplantılar, akademik)

Kullanım:
    python memory_clusterer.py              # Cluster et
    python memory_clusterer.py --stats      # İstatistikleri göster
    python memory_clusterer.py --search "topic"  # Cluster ara
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent

# Topic definitions with keyword weights
TOPIC_DEFINITIONS = {
    'aile': {
        'keywords': ['duygu', 'ada', 'eş', 'kızım', 'aile', 'ev', 'pasaport', 'vize', 'konsolosluk'],
        'weight': 1.0,
        'icon': '👨‍👩‍👧',
    },
    'teknik': {
        'keywords': ['openclaw', 'python', 'script', 'cron', 'config', 'model', 'api', 'server',
                     'install', 'kurulum', 'hata', 'error', 'debug', 'test', 'sistem', 'backup',
                     'yedekleme', 'git', 'github', 'faiss', 'ollama', 'whisper'],
        'weight': 1.0,
        'icon': '⚙️',
    },
    'proje': {
        'keywords': ['proje', 'retention', 'cognitive', 'world model', 'dream', 'watcher',
                     'entegrasyon', 'module', 'framework', 'geliştirme', 'deploy', 'release'],
        'weight': 1.0,
        'icon': '🚀',
    },
    'spor': {
        'keywords': ['galatasaray', 'gs', 'maç', 'gol', 'futbol', 'lig', 'şampiyon', 'başakşehir',
                     'osimhen', 'singo', 'stad', 'taraftar', 'aslan', '🦁'],
        'weight': 1.0,
        'icon': '⚽',
    },
    'kişisel': {
        'keywords': ['kahve', 'çay', 'yemek', 'film', 'müzik', 'hobi', 'tatil', 'seyahat',
                     'alışveriş', 'sağlık', 'spor', 'alışkanlık', 'tercih', 'seviyorum'],
        'weight': 0.8,
        'icon': '☕',
    },
    'iş': {
        'keywords': ['randevu', 'toplantı', 'üniversite', 'nişantaşı', 'iş', 'proje', 'sunum',
                     'ödev', 'ders', 'akademik', 'araştırma', 'makale'],
        'weight': 0.9,
        'icon': '💼',
    },
    'öğrenme': {
        'keywords': ['öğren', 'merak', 'araştır', 'keşfet', 'bilgi', 'konu', 'interest',
                     'newsletter', 'haber', 'gelişim', 'metacognition', 'strategi'],
        'weight': 0.7,
        'icon': '📚',
    },
    'sosyal': {
        'keywords': ['mesaj', 'whatsapp', 'telegram', 'konuşma', 'arkadaş', 'paylaş',
                     'grup', 'sohbet', 'espri', 'şaka', 'teşekkür', 'rica'],
        'weight': 0.7,
        'icon': '💬',
    },
}

# State file
_CLUSTER_STATE_FILE = _WORKSPACE_DIR / "cognitive_state" / "memory_clusters.json"


class MemoryClusterer:
    """Memory semantic clustering engine."""
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = str(_WORKSPACE_DIR)
        self.workspace = Path(workspace_dir)
        self.memory_dir = self.workspace / "memory"
        self.state_dir = self.workspace / "cognitive_state"
        self.state_dir.mkdir(exist_ok=True)
        
        # Load existing clusters
        self.clusters = self._load_clusters()
    
    def _load_clusters(self) -> Dict:
        """Cluster state'ini yükle."""
        if _CLUSTER_STATE_FILE.exists():
            try:
                with open(_CLUSTER_STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'topics': {},
            'file_assignments': {},
            'last_updated': None,
            'stats': {},
        }
    
    def _save_clusters(self):
        """Cluster state'ini kaydet."""
        with open(_CLUSTER_STATE_FILE, 'w') as f:
            json.dump(self.clusters, f, indent=2, ensure_ascii=False)
    
    def _score_topic(self, text: str, topic_name: str) -> float:
        """Bir metnin bir topic ile uyum skoru (0-1)."""
        topic = TOPIC_DEFINITIONS.get(topic_name, {})
        keywords = topic.get('keywords', [])
        weight = topic.get('weight', 1.0)
        
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        if not keywords:
            return 0.0
        
        score = (matches / len(keywords)) * weight
        return min(score, 1.0)
    
    def _classify_text(self, text: str, min_score: float = 0.05) -> List[Tuple[str, float]]:
        """Metni topic'lere sınıflandır. [(topic, score)] döner."""
        scores = []
        for topic_name in TOPIC_DEFINITIONS:
            score = self._score_topic(text, topic_name)
            if score >= min_score:
                scores.append((topic_name, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:3]  # En iyi 3 topic
    
    def cluster_file(self, file_path: Path) -> Dict:
        """Tek bir memory dosyasını cluster'la."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return {'error': str(e)}
        
        topics = self._classify_text(content)
        
        # File assignment güncelle
        file_key = file_path.name
        self.clusters['file_assignments'][file_key] = {
            'primary_topic': topics[0][0] if topics else 'diğer',
            'primary_score': round(topics[0][1], 3) if topics else 0,
            'all_topics': [(t, round(s, 3)) for t, s in topics],
            'word_count': len(content.split()),
            'clustered_at': datetime.now().isoformat(),
        }
        
        # Topic stats güncelle
        for topic_name, score in topics:
            if topic_name not in self.clusters['topics']:
                self.clusters['topics'][topic_name] = {
                    'file_count': 0,
                    'total_score': 0,
                    'files': [],
                }
            
            t = self.clusters['topics'][topic_name]
            t['file_count'] += 1
            t['total_score'] += score
            if file_key not in t['files']:
                t['files'].append(file_key)
        
        return self.clusters['file_assignments'][file_key]
    
    def cluster_all(self, force: bool = False) -> Dict:
        """Tüm memory dosyalarını cluster'la."""
        if not self.memory_dir.exists():
            return {'error': 'Memory dizini yok'}
        
        files = sorted(self.memory_dir.glob("20*-*.md"))
        
        processed = 0
        skipped = 0
        
        for f in files:
            # Incremental: sadece yeni/degismis dosyalar
            if not force and f.name in self.clusters['file_assignments']:
                existing = self.clusters['file_assignments'][f.name]
                file_mtime = f.stat().st_mtime
                cluster_time_str = existing.get('clustered_at', '2000-01-01')
                try:
                    cluster_time = datetime.fromisoformat(cluster_time_str).timestamp()
                    if file_mtime <= cluster_time:
                        skipped += 1
                        continue
                except Exception:
                    pass
            
            self.cluster_file(f)
            processed += 1
        
        # Stats
        total_files = len(files)
        topic_summary = {}
        for topic_name, topic_data in self.clusters['topics'].items():
            icon = TOPIC_DEFINITIONS.get(topic_name, {}).get('icon', '📁')
            topic_summary[topic_name] = {
                'icon': icon,
                'files': topic_data['file_count'],
                'avg_score': round(topic_data['total_score'] / max(topic_data['file_count'], 1), 3),
            }
        
        self.clusters['last_updated'] = datetime.now().isoformat()
        self.clusters['stats'] = {
            'total_files': total_files,
            'processed': processed,
            'skipped': skipped,
            'topic_count': len(self.clusters['topics']),
        }
        
        self._save_clusters()
        
        return {
            'status': 'completed',
            'total_files': total_files,
            'processed': processed,
            'skipped': skipped,
            'topics': topic_summary,
        }
    
    def search_topic(self, query: str) -> Dict:
        """Topic ara."""
        query_lower = query.lower()
        results = {}
        
        for topic_name, topic_data in self.clusters['topics'].items():
            # Topic name match
            if query_lower in topic_name:
                results[topic_name] = topic_data
                continue
            
            # Keyword match
            topic_def = TOPIC_DEFINITIONS.get(topic_name, {})
            keywords = topic_def.get('keywords', [])
            if any(query_lower in kw for kw in keywords):
                results[topic_name] = topic_data
        
        return results
    
    def get_stats(self) -> Dict:
        """Cluster istatistikleri."""
        return {
            'stats': self.clusters.get('stats', {}),
            'topics': {
                name: {
                    'icon': TOPIC_DEFINITIONS.get(name, {}).get('icon', '📁'),
                    'files': data['file_count'],
                    'avg_score': round(data['total_score'] / max(data['file_count'], 1), 3),
                    'sample_files': data['files'][:5],
                }
                for name, data in self.clusters.get('topics', {}).items()
            },
            'last_updated': self.clusters.get('last_updated'),
        }


def main():
    parser = argparse.ArgumentParser(description='Memory Clusterer')
    parser.add_argument('--workspace', default=None)
    parser.add_argument('--stats', action='store_true', help='İstatistikleri göster')
    parser.add_argument('--search', default=None, help='Topic ara')
    parser.add_argument('--force', action='store_true', help='Tüm dosyaları yeniden cluster et')
    
    args = parser.parse_args()
    
    clusterer = MemoryClusterer(args.workspace)
    
    if args.stats:
        stats = clusterer.get_stats()
        print("📊 Memory Cluster İstatistikleri\n")
        print(f"Toplam dosya: {stats['stats'].get('total_files', 0)}")
        print(f"Topic sayısı: {stats['stats'].get('topic_count', 0)}")
        print(f"Son güncelleme: {stats['last_updated'] or 'Hiç'}\n")
        
        for name, data in sorted(stats['topics'].items(), key=lambda x: x[1]['files'], reverse=True):
            print(f"  {data['icon']} {name}: {data['files']} dosya (skor: {data['avg_score']})")
            if data['sample_files']:
                print(f"     Örnekler: {', '.join(data['sample_files'][:3])}")
    
    elif args.search:
        results = clusterer.search_topic(args.search)
        print(f"🔍 '{args.search}' arama sonuçları:\n")
        for name, data in results.items():
            icon = TOPIC_DEFINITIONS.get(name, {}).get('icon', '📁')
            print(f"  {icon} {name}: {data['file_count']} dosya")
    
    else:
        result = clusterer.cluster_all(force=args.force)
        print("🏷️ Memory Clustering tamamlandı\n")
        print(f"Dosyalar: {result.get('total_files', 0)}")
        print(f"İşlenen: {result.get('processed', 0)}")
        print(f"Atlanan: {result.get('skipped', 0)}")
        
        topics = result.get('topics', {})
        if topics:
            print("\nTopic dağılımı:")
            for name, data in sorted(topics.items(), key=lambda x: x[1]['files'], reverse=True):
                print(f"  {data['icon']} {name}: {data['files']} dosya")


if __name__ == "__main__":
    main()

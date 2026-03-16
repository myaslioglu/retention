"""
World Model V2 - Bilgi Grafiği + Hafıza Nüfuslandırma

MEMORY.md ve günlük dosyalardan bilgi çıkarır, 
bilenir bir bilgi grafiği oluşturur.

Bileşenler:
- KnowledgeGraph: düğümler (entity/kavram) + kenarlar (ilişkiler)
- TopicClusters: ilgili anıları konuya göre gruplar
- TimelineIndex: kronolojik olay indeksi
- RelationshipMap: ilişki haritası (tercihler, alışkanlıklar, bağlantılar)
"""

import json
import re
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# KnowledgeGraph - Bilgi Grafiği
# ============================================================

class KnowledgeGraph:
    """
    Varlıklar (entities) ve ilişkilerden oluşan bilgi grafiği.
    
    Düğümler: kişiler, kavramlar, yerler, organizasyonlar, projeler
    Kenarlar: ilişkiler (ilişki tipi + ağırlık)
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # node_id -> {type, label, properties, mentions}
        self.edges: List[Dict] = []       # {source, target, relation, weight, evidence}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node_id: str, node_type: str = "concept", 
                 label: str = "", properties: Dict = None, source: str = ""):
        """Düğüm ekle veya güncelle."""
        if node_id in self.nodes:
            # Güncelle
            self.nodes[node_id]['mentions'] += 1
            if source:
                self.nodes[node_id]['sources'].append(source)
            if properties:
                self.nodes[node_id]['properties'].update(properties)
        else:
            # Yeni düğüm
            self.nodes[node_id] = {
                'id': node_id,
                'type': node_type,
                'label': label or node_id,
                'properties': properties or {},
                'mentions': 1,
                'sources': [source] if source else [],
                'created': datetime.now().isoformat(),
            }
    
    def add_edge(self, source: str, target: str, relation: str, 
                 weight: float = 1.0, evidence: str = "", context: str = ""):
        """Kenar ekle."""
        # Düğümlerin var olduğundan emin ol
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        # Duplicate kontrolü
        for edge in self.edges:
            if (edge['source'] == source and edge['target'] == target and 
                edge['relation'] == relation):
                edge['weight'] = max(edge['weight'], weight)
                if evidence:
                    edge['evidence'] += f" | {evidence}"
                return
        
        self.edges.append({
            'source': source,
            'target': target,
            'relation': relation,
            'weight': weight,
            'evidence': evidence,
            'created': datetime.now().isoformat(),
        })
        
        self._adjacency[source].add(target)
        self._adjacency[target].add(source)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Komşu düğümleri getir."""
        return list(self._adjacency.get(node_id, set()))
    
    def get_node_edges(self, node_id: str) -> List[Dict]:
        """Düğüme ait kenarları getir."""
        return [e for e in self.edges if e['source'] == node_id or e['target'] == node_id]
    
    def find_path(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        """İki düğüm arasındaki yolları bul (BFS)."""
        if source not in self.nodes or target not in self.nodes:
            return []
        
        visited = set()
        queue = [(source, [source])]
        paths = []
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth + 1:
                continue
            
            if current == target:
                paths.append(path)
                continue
            
            visited.add(current)
            
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return paths[:5]  # En fazla 5 yol
    
    def to_dict(self) -> Dict:
        """JSON serileştirme."""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'node_types': self._count_types(),
            }
        }
    
    def _count_types(self) -> Dict[str, int]:
        """Düğüm tiplerini say."""
        types = defaultdict(int)
        for node in self.nodes.values():
            types[node['type']] += 1
        return dict(types)


# ============================================================
# TopicClusters - Konu Kümeleri
# ============================================================

class TopicClusters:
    """
    Anıları ve bilgileri konuya göre gruplar.
    """
    
    # Ana konu kategorileri
    TOPIC_KEYWORDS = {
        'galatasaray': ['galatasaray', 'gs', 'aslan', 'cimbom', 'sarı kırmızı', 'maç', 'futbol', 'stadyum'],
        'teknoloji': ['python', 'code', 'yazılım', 'programming', 'api', 'openclaw', 'ai', 'yapay zeka', 'model', 'llm'],
        'aile': ['duygu', 'ada', 'eş', 'kız', 'aile', 'ev', 'çocuk'],
        'iş': ['proje', 'iş', 'çalışma', 'toplantı', 'müşteri', 'ofis'],
        'seyahat': ['uçak', 'otel', 'tatil', 'seyahat', 'vize', 'pasaport', 'yurtdışı'],
        'sağlık': ['doktor', 'hastane', 'sağlık', 'ilaç', 'spor', 'egzersiz'],
        'yiyecek': ['kahve', 'çay', 'yemek', 'restoran', 'espresso', 'siyah çay'],
        'eğlence': ['film', 'müzik', 'oyun', 'eğlence', 'dizi', 'kitap'],
        'finans': ['para', 'bütçe', 'yatırım', 'hisse', 'bitcoin', 'dolar'],
        'openclaw': ['openclaw', 'hacı', 'cognitive', 'retention', 'memory', 'dream', 'cron'],
    }
    
    def __init__(self):
        self.clusters: Dict[str, List[Dict]] = defaultdict(list)
        self.topic_scores: Dict[str, float] = defaultdict(float)
    
    def classify_text(self, text: str) -> List[Tuple[str, float]]:
        """Metni konulara sınıflandır."""
        text_lower = text.lower()
        scores = []
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = 0
            for kw in keywords:
                count = text_lower.count(kw)
                score += count
            if score > 0:
                scores.append((topic, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def add_memory(self, memory: Dict, source: str = ""):
        """Anıyı ilgili konu kümelerine ekle."""
        content = memory.get('content', '') + ' ' + memory.get('title', '')
        topics = self.classify_text(content)
        
        for topic, score in topics:
            entry = {
                'memory': memory,
                'source': source,
                'score': score,
                'timestamp': memory.get('timestamp', datetime.now().isoformat()),
            }
            self.clusters[topic].append(entry)
            self.topic_scores[topic] += score
    
    def get_top_topics(self, n: int = 10) -> List[Tuple[str, int, float]]:
        """En popüler konuları getir."""
        result = []
        for topic, entries in self.clusters.items():
            result.append((topic, len(entries), self.topic_scores[topic]))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:n]
    
    def to_dict(self) -> Dict:
        """JSON serileştirme."""
        return {
            'clusters': {k: len(v) for k, v in self.clusters.items()},
            'topic_scores': dict(self.topic_scores),
            'top_topics': [(t, n, s) for t, n, s in self.get_top_topics()],
        }


# ============================================================
# TimelineIndex - Kronolojik İndeks
# ============================================================

class TimelineIndex:
    """
    Olayları kronolojik sıraya göre indeksler.
    """
    
    def __init__(self):
        self.events: List[Dict] = []
        self._date_index: Dict[str, List[int]] = defaultdict(list)
    
    def add_event(self, date: str, title: str, content: str = "", 
                  category: str = "general", importance: float = 0.5):
        """Olay ekle."""
        idx = len(self.events)
        event = {
            'idx': idx,
            'date': date,
            'title': title,
            'content': content[:500],
            'category': category,
            'importance': importance,
        }
        self.events.append(event)
        
        # Tarih indeksi
        date_key = date[:10]  # YYYY-MM-DD
        self._date_index[date_key].append(idx)
    
    def get_events_by_date(self, date: str) -> List[Dict]:
        """Tarihe göre olayları getir."""
        idxs = self._date_index.get(date[:10], [])
        return [self.events[i] for i in idxs]
    
    def get_events_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Tarih aralığındaki olayları getir."""
        result = []
        for event in self.events:
            if start_date <= event['date'][:10] <= end_date:
                result.append(event)
        return result
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Son N olayı getir."""
        return self.events[-n:]
    
    def to_dict(self) -> Dict:
        """JSON serileştirme."""
        return {
            'total_events': len(self.events),
            'date_range': {
                'first': self.events[0]['date'] if self.events else None,
                'last': self.events[-1]['date'] if self.events else None,
            },
            'recent': self.get_recent(5),
        }


# ============================================================
# RelationshipMap - İlişki Haritası
# ============================================================

class RelationshipMap:
    """
    Kullanıcı tercihlerini, alışkanlıklarını ve bağlantılarını haritalar.
    """
    
    def __init__(self):
        self.preferences: Dict[str, Any] = {}
        self.habits: Dict[str, Dict] = {}
        self.connections: Dict[str, Dict] = {}  # kişi/entity -> bilgi
        self.interests: Dict[str, float] = defaultdict(float)
    
    def add_preference(self, category: str, key: str, value: Any, confidence: float = 1.0):
        """Tercih ekle."""
        if category not in self.preferences:
            self.preferences[category] = {}
        self.preferences[category][key] = {
            'value': value,
            'confidence': confidence,
            'updated': datetime.now().isoformat(),
        }
    
    def add_habit(self, habit: str, frequency: str = "unknown", evidence: str = ""):
        """Alışkanlık ekle."""
        if habit in self.habits:
            self.habits[habit]['count'] += 1
            if evidence:
                self.habits[habit]['evidence'].append(evidence)
        else:
            self.habits[habit] = {
                'frequency': frequency,
                'count': 1,
                'evidence': [evidence] if evidence else [],
                'first_seen': datetime.now().isoformat(),
            }
    
    def add_connection(self, person: str, relation: str = "acquaintance", 
                       notes: str = ""):
        """Kişi/entity bağlantısı ekle."""
        if person in self.connections:
            self.connections[person]['interactions'] += 1
            if notes:
                self.connections[person]['notes'].append(notes)
        else:
            self.connections[person] = {
                'relation': relation,
                'interactions': 1,
                'notes': [notes] if notes else [],
                'first_seen': datetime.now().isoformat(),
            }
    
    def add_interest(self, topic: str, weight: float = 1.0):
        """İlgi alanı ekle/güncelle."""
        self.interests[topic] += weight
    
    def extract_from_text(self, text: str, source: str = ""):
        """Metinden tercih/alışkanlık çıkar."""
        text_lower = text.lower()
        
        # Tercih tespiti
        preference_patterns = [
            (r'(seviyorum|hoşlanıyorum|favori|tercih ediyorum)\s+(\w+)', 'likes'),
            (r'(nefret ediyorum|sevmiyorum|hoşlanmıyorum)\s+(\w+)', 'dislikes'),
            (r'(kahve|çay|espresso)\s+(seviyorum|favori|iyidir)', 'drinks'),
        ]
        
        for pattern, category in preference_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    self.add_preference(category, match[1], True, confidence=0.7)
        
        # Kişi isimleri (basit)
        person_patterns = [
            r'\b(Duygu|Ada|Murat|Başkan)\b',
        ]
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                self.add_connection(name, notes=f"Mentioned in: {source}")
    
    def to_dict(self) -> Dict:
        """JSON serileştirme."""
        return {
            'preferences': self.preferences,
            'habits': {k: {**v, 'evidence': v['evidence'][-5:]} for k, v in self.habits.items()},
            'connections': self.connections,
            'interests': dict(sorted(self.interests.items(), key=lambda x: x[1], reverse=True)[:20]),
        }


# ============================================================
# WorldModelV2 - Ana Sınıf
# ============================================================

class WorldModelV2:
    """
    Dünya Modeli V2 - MEMORY.md ve günlük dosyalardan bilgi çıkarır.
    
    Bileşenler:
    - KnowledgeGraph: entity/ilişki ağı
    - TopicClusters: konu kümeleri
    - TimelineIndex: kronolojik olaylar
    - RelationshipMap: kullanıcı tercihleri ve bağlantılar
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.state_dir = self.workspace / "cognitive_state"
        self.state_dir.mkdir(exist_ok=True)
        
        self.knowledge_file = self.state_dir / "world_knowledge.json"
        
        # Alt bileşenler
        self.graph = KnowledgeGraph()
        self.topics = TopicClusters()
        self.timeline = TimelineIndex()
        self.relationships = RelationshipMap()
        
        # İstatistikler
        self.stats = {
            'memories_processed': 0,
            'entities_extracted': 0,
            'relations_found': 0,
            'last_populated': None,
            'last_updated': None,
        }
        
        # Mevcut durumu yükle
        self._load_knowledge()
    
    def _load_knowledge(self):
        """Bilgi grafiğini diskten yükle."""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file) as f:
                    data = json.load(f)
                
                # Graph restore
                for node_id, node_data in data.get('graph', {}).get('nodes', {}).items():
                    self.graph.nodes[node_id] = node_data
                self.graph.edges = data.get('graph', {}).get('edges', [])
                
                # Rebuild adjacency
                for edge in self.graph.edges:
                    self.graph._adjacency[edge['source']].add(edge['target'])
                    self.graph._adjacency[edge['target']].add(edge['source'])
                
                self.stats = data.get('stats', self.stats)
                logger.info(f"📂 World knowledge loaded ({len(self.graph.nodes)} nodes)")
            except Exception as e:
                logger.warning(f"Bilgi yüklenemedi: {e}")
    
    def _save_knowledge(self):
        """Bilgi grafiğini diske kaydet."""
        data = {
            'graph': self.graph.to_dict(),
            'topics': self.topics.to_dict(),
            'timeline': self.timeline.to_dict(),
            'relationships': self.relationships.to_dict(),
            'stats': self.stats,
            'last_saved': datetime.now().isoformat(),
        }
        
        with open(self.knowledge_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 World knowledge saved ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)")
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Metinden entity çıkar.
        
        Returns: [(entity_id, entity_type, label), ...]
        """
        entities = []
        
        # Kişi isimleri
        person_patterns = [r'\b(Murat|Duygu|Ada|Başkan|Hacı)\b']
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                entities.append((name.lower(), 'person', name))
        
        # Organizasyonlar
        org_patterns = [r'\b(Galatasaray|GS|OpenClaw|Google|Apple|Microsoft)\b']
        for pattern in org_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                org = match.group(1)
                entities.append((org.lower(), 'organization', org))
        
        # Projeler/Kavramlar
        project_patterns = [
            r'\b(HaciCognitiveNet|cognitive|retention|dreaming|FAISS)\b',
            r'\b(python|javascript|react|docker|kubernetes)\b',
        ]
        for pattern in project_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concept = match.group(1)
                entities.append((concept.lower(), 'concept', concept))
        
        # Yerler
        place_patterns = [r'\b(İstanbul|Ankara|İzmir|Nişantaşı|Londra|New York)\b']
        for pattern in place_patterns:
            for match in re.finditer(pattern, text):
                place = match.group(1)
                entities.append((place.lower(), 'place', place))
        
        # Tarihler
        date_patterns = [r'(\d{4}-\d{2}-\d{2})']
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date = match.group(1)
                entities.append((f"date_{date}", 'date', date))
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str, float]]:
        """
        Entity'ler arası ilişki çıkar.
        
        Returns: [(source, target, relation_type, weight), ...]
        """
        relations = []
        
        if len(entities) < 2:
            return relations
        
        text_lower = text.lower()
        
        # Basit co-occurrence ilişkileri
        entity_ids = [e[0] for e in entities]
        
        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i+1:]:
                # İlişki tipi tahmini
                relation = 'related_to'
                
                if 'seviyor' in text_lower or 'favori' in text_lower:
                    relation = 'likes'
                elif 'çalışıyor' in text_lower or 'üzerinde' in text_lower:
                    relation = 'works_on'
                elif 'eşi' in text_lower or 'karısı' in text_lower:
                    relation = 'spouse_of'
                elif 'kızı' in text_lower or 'çocuğu' in text_lower:
                    relation = 'parent_of'
                elif 'arkadaşı' in text_lower:
                    relation = 'friend_of'
                elif 'proje' in text_lower:
                    relation = 'part_of'
                
                relations.append((e1, e2, relation, 0.5))
        
        return relations
    
    def _extract_date(self, text: str, filename: str = "") -> str:
        """Metinden tarih çıkar."""
        # Dosya adından tarih
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return date_match.group(1)
        
        # Metinden tarih
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if date_match:
            return date_match.group(1)
        
        return datetime.now().strftime('%Y-%m-%d')
    
    def populate_from_memory(self):
        """
        MEMORY.md'den bilgi çıkar ve grafiği doldur.
        """
        memory_file = self.workspace / "MEMORY.md"
        if not memory_file.exists():
            logger.warning("MEMORY.md bulunamadı!")
            return
        
        logger.info("📖 MEMORY.md'den bilgi çıkarılıyor...")
        
        content = memory_file.read_text(encoding='utf-8')
        
        # Bölüm bazında işle
        sections = re.split(r'\n## ', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Başlık
            lines = section.strip().split('\n')
            title = lines[0].strip('# ') if lines else 'Unknown'
            
            # İçerik
            section_content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            
            # Entity çıkar
            entities = self._extract_entities(section + ' ' + section_content)
            for entity_id, entity_type, label in entities:
                self.graph.add_node(entity_id, entity_type, label, source='MEMORY.md')
                self.stats['entities_extracted'] += 1
            
            # İlişki çıkar
            relations = self._extract_relations(section_content, entities)
            for src, tgt, rel, weight in relations:
                self.graph.add_edge(src, tgt, rel, weight, evidence=title[:100])
                self.stats['relations_found'] += 1
            
            # Timeline'a ekle
            date = self._extract_date(section, 'MEMORY.md')
            self.timeline.add_event(
                date=date,
                title=title[:100],
                content=section_content[:300],
                category=self._classify_section(title),
                importance=self._estimate_importance(title, section_content),
            )
            
            # Topic clustering
            self.topics.add_memory({
                'title': title,
                'content': section_content[:500],
                'timestamp': date,
            }, source='MEMORY.md')
            
            # Relationship extraction
            self.relationships.extract_from_text(section_content, source='MEMORY.md')
            
            self.stats['memories_processed'] += 1
        
        self.stats['last_populated'] = datetime.now().isoformat()
        self._save_knowledge()
        
        logger.info(f"✅ Memory nüfuslandırması tamamlandı: {self.stats['memories_processed']} bölüm, "
                    f"{self.stats['entities_extracted']} entity, {self.stats['relations_found']} ilişki")
    
    def update_from_daily(self, daily_file: str):
        """
        Günlük dosyadan güncelleme yap.
        """
        daily_path = Path(daily_file)
        if not daily_path.exists():
            logger.warning(f"Günlük dosya bulunamadı: {daily_file}")
            return
        
        logger.info(f"📖 Günlük dosya güncelleniyor: {daily_path.name}")
        
        content = daily_path.read_text(encoding='utf-8')
        
        # Entity çıkar
        entities = self._extract_entities(content)
        for entity_id, entity_type, label in entities:
            self.graph.add_node(entity_id, entity_type, label, source=daily_path.name)
        
        # İlişki çıkar
        relations = self._extract_relations(content, entities)
        for src, tgt, rel, weight in relations:
            self.graph.add_edge(src, tgt, rel, weight, evidence=daily_path.name)
        
        # Timeline
        date = self._extract_date(content, daily_path.name)
        self.timeline.add_event(
            date=date,
            title=f"Daily: {daily_path.name}",
            content=content[:300],
            category='daily',
        )
        
        # Topics
        self.topics.add_memory({
            'content': content[:500],
            'timestamp': date,
        }, source=daily_path.name)
        
        self.stats['memories_processed'] += 1
        self.stats['last_updated'] = datetime.now().isoformat()
        self._save_knowledge()
    
    def query(self, topic: str) -> Dict:
        """
        Bir konu hakkında bilgi ara.
        """
        topic_lower = topic.lower()
        results = {
            'topic': topic,
            'graph_nodes': [],
            'related_memories': [],
            'timeline_events': [],
            'connections': [],
        }
        
        # Graph'da ara
        for node_id, node_data in self.graph.nodes.items():
            if topic_lower in node_id or topic_lower in node_data.get('label', '').lower():
                results['graph_nodes'].append(node_data)
        
        # Topic cluster'da ara
        for cluster_topic, entries in self.topics.clusters.items():
            if topic_lower in cluster_topic:
                for entry in entries[:5]:
                    results['related_memories'].append({
                        'source': entry['source'],
                        'score': entry['score'],
                        'preview': str(entry['memory'].get('content', ''))[:200],
                    })
        
        # Timeline'da ara
        for event in self.timeline.events:
            if topic_lower in event.get('title', '').lower() or topic_lower in event.get('content', '').lower():
                results['timeline_events'].append(event)
        
        return results
    
    def find_connections(self, topic1: str, topic2: str) -> Dict:
        """
        İki konu arasındaki bağlantıları bul.
        """
        # Her iki konu için node'ları bul
        nodes1 = [nid for nid, nd in self.graph.nodes.items() 
                  if topic1.lower() in nid or topic1.lower() in nd.get('label', '').lower()]
        nodes2 = [nid for nid, nd in self.graph.nodes.items() 
                  if topic2.lower() in nid or topic2.lower() in nd.get('label', '').lower()]
        
        paths = []
        for n1 in nodes1[:3]:
            for n2 in nodes2[:3]:
                found_paths = self.graph.find_path(n1, n2, max_depth=3)
                paths.extend(found_paths)
        
        return {
            'topic1': topic1,
            'topic2': topic2,
            'nodes_topic1': nodes1,
            'nodes_topic2': nodes2,
            'paths': paths[:10],
            'direct_connection': len(paths) > 0,
        }
    
    def get_stats(self) -> Dict:
        """Bilgi istatistiklerini getir."""
        return {
            'graph': {
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges),
                'node_types': self.graph._count_types(),
            },
            'topics': {
                'total_clusters': len(self.topics.clusters),
                'top_topics': [(t, n) for t, n, s in self.topics.get_top_topics(5)],
            },
            'timeline': {
                'total_events': len(self.timeline.events),
                'date_range': self.timeline.to_dict().get('date_range', {}),
            },
            'relationships': {
                'preferences': len(self.relationships.preferences),
                'habits': len(self.relationships.habits),
                'connections': len(self.relationships.connections),
                'top_interests': list(self.relationships.interests.items())[:10],
            },
            'processing': self.stats,
        }
    
    def _classify_section(self, title: str) -> str:
        """Bölüm başlığını sınıflandır."""
        title_lower = title.lower()
        if 'lesson' in title_lower or 'ders' in title_lower:
            return 'lesson'
        elif 'decision' in title_lower or 'karar' in title_lower:
            return 'decision'
        elif 'memory' in title_lower or 'anı' in title_lower:
            return 'memory'
        elif 'task' in title_lower or 'görev' in title_lower:
            return 'task'
        else:
            return 'general'
    
    def _estimate_importance(self, title: str, content: str) -> float:
        """İçeriğin önemini tahmin et."""
        importance = 0.5
        
        # Başlık ipuçları
        title_lower = title.lower()
        if any(w in title_lower for w in ['acil', 'önemli', 'critical', 'important']):
            importance += 0.3
        if any(w in title_lower for w in ['karar', 'decision']):
            importance += 0.2
        if any(w in title_lower for w in ['hata', 'error', 'sorun']):
            importance += 0.1
        
        # İçerik uzunluğu
        if len(content) > 500:
            importance += 0.1
        
        return min(1.0, importance)


# ============================================================
# CLI Entry Point
# ============================================================

def run_population(workspace_dir: str = None):
    """İlk nüfuslandırmayı çalıştır."""
    if workspace_dir is None:
        workspace_dir = str(Path.home() / ".openclaw" / "workspace")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== World Model V2 - Memory Nüfuslandırma ===\n")
    
    model = WorldModelV2(workspace_dir)
    
    # 1. MEMORY.md'den nüfuslandır
    model.populate_from_memory()
    
    # 2. Günlük dosyaları ekle
    memory_dir = Path(workspace_dir) / "memory"
    if memory_dir.exists():
        daily_files = sorted(memory_dir.glob("*.md"), reverse=True)
        for daily_file in daily_files[:7]:  # Son 7 gün
            model.update_from_daily(str(daily_file))
    
    # 3. İstatistikler
    stats = model.get_stats()
    print("\n📊 Dünya Modeli İstatistikleri:")
    print(f"   📊 Graph: {stats['graph']['nodes']} düğüm, {stats['graph']['edges']} kenar")
    print(f"   📚 Konular: {stats['topics']['total_clusters']} küme")
    print(f"   📅 Timeline: {stats['timeline']['total_events']} olay")
    print(f"   🔗 İlişkiler: {stats['relationships']['connections']} bağlantı")
    
    if stats['topics']['top_topics']:
        print(f"\n   En popüler konular:")
        for topic, count in stats['topics']['top_topics']:
            print(f"      • {topic}: {count} anı")
    
    print(f"\n✅ Nüfuslandırma tamamlandı!")
    return model


if __name__ == "__main__":
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else None
    run_population(workspace)

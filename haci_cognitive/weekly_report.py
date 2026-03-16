"""
Weekly Report Generator - Haftalık bilişsel gelişim raporu

Her hafta otomatik oluşturulan rapor:
- Kişilik gelişimi: trait değişimleri
- Öğrenme özeti: keşfedilen konular, merak eğilimleri
- Sosyal zeka: loglanan etkileşimler, negatif kurallar
- Rüya içgörüleri: keşfedilen desenler, önemli rüyalar
- Bilgi büyümesi: yeni entity'ler, konu dağılımı değişimleri
- Duygusal eğilimler: ruh hali/güven/enerji ortalamaları
- Öneriler: geliştirilmesi gereken alanlar

Kullanım:
    python weekly_report.py generate
    python weekly_report.py summary
"""

import json
import os
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent


class WeeklyReportGenerator:
    """
    Haftalık bilişsel gelişim raporu oluşturucu.
    
    Tüm state dosyalarından veri toplar ve markdown rapor oluşturur.
    """
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = str(_WORKSPACE_DIR)
        
        self.workspace = Path(workspace_dir)
        self.state_dir = self.workspace / "cognitive_state"
        self.reports_dir = self.state_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_dir = self.workspace / "memory"
        
        logger.info("📊 WeeklyReportGenerator initialized")
    
    def generate_report(self, weeks_back: int = 1) -> str:
        """
        Haftalık rapor oluştur.
        
        Args:
            weeks_back: Kaç hafta geriye gidilecek (1 = geçen hafta)
            
        Returns:
            Oluşturulan rapor dosyasının yolu
        """
        logger.info(f"📊 Rapor oluşturuluyor ({weeks_back} hafta geriye)...")
        
        # Tarih aralığı
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7 * weeks_back)
        
        # Veri topla
        data = self.get_trend_data(days=7 * weeks_back)
        
        # Rapor formatla
        report = self.format_report(data)
        
        # Dosya adı
        week_num = end_date.isocalendar()[1]
        year = end_date.year
        filename = f"week_{year}_{week_num:02d}.md"
        report_path = self.reports_dir / filename
        
        # Kaydet
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📊 Rapor kaydedildi: {report_path}")
        return str(report_path)
    
    def get_trend_data(self, days: int = 7) -> Dict:
        """
        Son N gündeki trend verilerini topla.
        
        Returns:
            Tüm alt sistemlerden toplanan veri dict'i
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        data = {
            'period': {
                'start': cutoff.strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d'),
                'days': days,
            },
            'personality': self._collect_personality_data(cutoff_str),
            'learning': self._collect_learning_data(cutoff_str),
            'social': self._collect_social_data(cutoff_str),
            'dreams': self._collect_dream_data(cutoff_str),
            'knowledge': self._collect_knowledge_data(cutoff_str),
            'emotional': self._collect_emotional_data(cutoff_str),
            'watcher': self._collect_watcher_data(cutoff_str),
            'memory_files': self._collect_memory_files(cutoff),
        }
        
        return data
    
    def _collect_personality_data(self, cutoff: str) -> Dict:
        """Kişilik gelişimi verilerini topla."""
        data = {
            'current_traits': {},
            'trait_history': [],
            'development_stage': 'unknown',
            'interaction_count': 0,
        }
        
        # Personality state
        p_file = self.state_dir / "personality_development.json"
        if p_file.exists():
            try:
                with open(p_file) as f:
                    state = json.load(f)
                
                data['current_traits'] = state.get('traits', {})
                data['interaction_count'] = len(state.get('interaction_learnings', []))
                
                # Development log'dan dönem içi değişiklikler
                dev_log = state.get('development_log', [])
                for entry in dev_log:
                    if entry.get('timestamp', '') >= cutoff:
                        data['trait_history'].append(entry)
                
                # Stage hesapla
                total = data['interaction_count']
                if total < 10:
                    data['development_stage'] = 'bebeklik'
                elif total < 50:
                    data['development_stage'] = 'çocukluk'
                elif total < 200:
                    data['development_stage'] = 'ergenlik'
                elif total < 1000:
                    data['development_stage'] = 'yetişkinlik'
                else:
                    data['development_stage'] = 'olgunluk'
            except Exception as e:
                logger.warning(f"Personality data error: {e}")
        
        # Cognitive state'den de al
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                data['cognitive_personality'] = state.get('personality', {})
            except Exception:
                pass
        
        return data
    
    def _collect_learning_data(self, cutoff: str) -> Dict:
        """Öğrenme verilerini topla."""
        data = {
            'topics_explored': [],
            'curiosity_level': 0,
            'learning_cycles': 0,
            'lessons_learned': [],
        }
        
        # Cognitive state
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                data['curiosity_level'] = state.get('emotion', {}).get('curiosity', 0)
                data['learning_cycles'] = state.get('metacognition', {}).get('total_learning_cycles', 0)
            except Exception:
                pass
        
        # Curiosity state
        c_file = self.state_dir / "curiosity_state.json"
        if c_file.exists():
            try:
                with open(c_file) as f:
                    state = json.load(f)
                data['topics_explored'] = state.get('explored_topics', [])
            except Exception:
                pass
        
        # Negative learnings
        n_file = self.state_dir / "negative_learnings.json"
        if n_file.exists():
            try:
                with open(n_file) as f:
                    state = json.load(f)
                for lesson in state.get('learned_lessons', []):
                    if lesson.get('timestamp', '') >= cutoff:
                        data['lessons_learned'].append(lesson)
            except Exception:
                pass
        
        return data
    
    def _collect_social_data(self, cutoff: str) -> Dict:
        """Sosyal zeka verilerini topla."""
        data = {
            'total_interactions': 0,
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'negative_signals': [],
            'conversation_styles': {},
            'never_rules': [],
        }
        
        # Social trainer stats
        stats_file = self.state_dir / "social_trainer_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    state = json.load(f)
                stats = state.get('stats', {})
                data['total_interactions'] = stats.get('total_interactions', 0)
                data['positive_outcomes'] = stats.get('positive_outcomes', 0)
                data['negative_outcomes'] = stats.get('negative_outcomes', 0)
            except Exception:
                pass
        
        # Negative learnings
        n_file = self.state_dir / "negative_learnings.json"
        if n_file.exists():
            try:
                with open(n_file) as f:
                    state = json.load(f)
                # Dönem içindeki negative signals
                for outcome in state.get('negative_outcomes', []):
                    if outcome.get('timestamp', '') >= cutoff:
                        data['negative_signals'].append(outcome)
                data['never_rules'] = state.get('never_rules', [])
            except Exception:
                pass
        
        # Conversation intelligence
        conv_file = self.state_dir / "conversation_intelligence.json"
        if conv_file.exists():
            try:
                with open(conv_file) as f:
                    state = json.load(f)
                data['conversation_styles'] = state.get('conversation_styles', {})
            except Exception:
                pass
        
        return data
    
    def _collect_dream_data(self, cutoff: str) -> Dict:
        """Rüya verilerini topla."""
        data = {
            'total_cycles': 0,
            'insights_generated': [],
            'connections_found': [],
            'recent_dreams': [],
        }
        
        # Dream schedule
        schedule_file = self.state_dir / "dream_schedule.json"
        if schedule_file.exists():
            try:
                with open(schedule_file) as f:
                    state = json.load(f)
                data['total_cycles'] = state.get('total_cycles', 0)
            except Exception:
                pass
        
        # Dream log
        log_file = self.state_dir / "dream_log.json"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    log = json.load(f)
                for entry in log:
                    if entry.get('timestamp', '') >= cutoff:
                        data['recent_dreams'].append(entry)
                        data['insights_generated'].extend(entry.get('insights', []))
            except Exception:
                pass
        
        return data
    
    def _collect_knowledge_data(self, cutoff: str) -> Dict:
        """Bilgi büyümesi verilerini topla."""
        data = {
            'entities_count': 0,
            'relations_count': 0,
            'topic_distribution': {},
            'recent_additions': 0,
        }
        
        # World knowledge
        wk_file = self.state_dir / "world_knowledge.json"
        if wk_file.exists():
            try:
                with open(wk_file) as f:
                    state = json.load(f)
                
                graph = state.get('graph', {})
                data['entities_count'] = graph.get('stats', {}).get('total_nodes', 0)
                data['relations_count'] = graph.get('stats', {}).get('total_edges', 0)
                
                topics = state.get('topics', {})
                data['topic_distribution'] = topics.get('clusters', {})
            except Exception:
                pass
        
        # Cognitive state world model
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                wm = state.get('world_model', {})
                data['memories_seen'] = wm.get('n_memories_seen', 0)
                data['insights_generated'] = wm.get('n_insights_generated', 0)
                data['connections_found'] = wm.get('n_connections_found', 0)
            except Exception:
                pass
        
        return data
    
    def _collect_emotional_data(self, cutoff: str) -> Dict:
        """Duygusal trend verilerini topla."""
        data = {
            'current_mood': 0,
            'current_confidence': 0,
            'current_energy': 0,
            'current_curiosity': 0,
            'trend': 'stable',
        }
        
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                emotion = state.get('emotion', {})
                data['current_mood'] = emotion.get('mood', 0)
                data['current_confidence'] = emotion.get('confidence', 0)
                data['current_energy'] = emotion.get('energy', 0)
                data['current_curiosity'] = emotion.get('curiosity', 0)
            except Exception:
                pass
        
        return data
    
    def _collect_watcher_data(self, cutoff: str) -> Dict:
        """Watcher verilerini topla."""
        data = {
            'total_messages': 0,
            'emotions': {},
            'interaction_types': {},
            'negative_signals': 0,
        }
        
        watcher_file = self.state_dir / "watcher_log.json"
        if watcher_file.exists():
            try:
                with open(watcher_file) as f:
                    log = json.load(f)
                
                period_entries = [e for e in log if e.get('timestamp', '') >= cutoff]
                data['total_messages'] = len(period_entries)
                
                for entry in period_entries:
                    emo = entry.get('emotion', 'unknown')
                    data['emotions'][emo] = data['emotions'].get(emo, 0) + 1
                    
                    itype = entry.get('interaction_type', 'unknown')
                    data['interaction_types'][itype] = data['interaction_types'].get(itype, 0) + 1
                    
                    if entry.get('negative_signal'):
                        data['negative_signals'] += 1
            except Exception:
                pass
        
        return data
    
    def _collect_memory_files(self, cutoff_dt: datetime) -> List[Dict]:
        """Dönem içindeki memory dosyalarını topla."""
        files = []
        
        if not self.memory_dir.exists():
            return files
        
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name.startswith('.'):
                continue
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime >= cutoff_dt:
                    content = f.read_text(encoding='utf-8')
                    files.append({
                        'name': f.name,
                        'date': mtime.strftime('%Y-%m-%d'),
                        'size': len(content),
                        'lines': content.count('\n'),
                    })
            except Exception:
                pass
        
        return files
    
    def format_report(self, data: Dict) -> str:
        """
        Veriyi markdown rapor formatına dönüştür.
        """
        period = data['period']
        
        report = f"""# 🧿 Haftalık Bilişsel Gelişim Raporu

**Dönem:** {period['start']} → {period['end']} ({period['days']} gün)
**Oluşturulma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 🎭 Kişilik Gelişimi

"""
        # Kişilik
        personality = data['personality']
        report += f"**Gelişim Aşaması:** {personality['development_stage']}\n"
        report += f"**Toplam Etkileşim:** {personality['interaction_count']}\n\n"
        
        traits = personality.get('current_traits', {})
        if traits:
            report += "**Trait'ler:**\n"
            for trait, value in sorted(traits.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(value * 10) + '░' * (10 - int(value * 10))
                report += f"- {trait}: {bar} {value:.2f}\n"
        
        # Cognitive personality
        cog_pers = personality.get('cognitive_personality', {})
        if cog_pers:
            report += "\n**Cognitive Personality:**\n"
            for trait, value in sorted(cog_pers.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(value * 10) + '░' * (10 - int(value * 10))
                report += f"- {trait}: {bar} {value:.2f}\n"
        
        trait_changes = personality.get('trait_history', [])
        if trait_changes:
            report += f"\n**Dönem İçi Değişiklikler:** {len(trait_changes)} kayıt\n"
        
        # Öğrenme
        report += "\n---\n\n## 📚 Öğrenme Özeti\n\n"
        learning = data['learning']
        report += f"**Merak Seviyesi:** {learning['curiosity_level']:.2f}\n"
        report += f"**Öğrenme Döngüleri:** {learning['learning_cycles']}\n"
        
        topics = learning.get('topics_explored', [])
        if topics:
            report += f"\n**Keşfedilen Konular:**\n"
            for topic in topics[:10]:
                if isinstance(topic, dict):
                    report += f"- {topic.get('name', topic)}\n"
                else:
                    report += f"- {topic}\n"
        
        lessons = learning.get('lessons_learned', [])
        if lessons:
            report += f"\n**Öğrenilen Dersler ({len(lessons)}):**\n"
            for lesson in lessons[:5]:
                report += f"- {lesson.get('what_i_did_wrong', str(lesson)[:100])}\n"
        
        # Sosyal Zeka
        report += "\n---\n\n## 🤝 Sosyal Zeka\n\n"
        social = data['social']
        report += f"**Toplam Etkileşim:** {social['total_interactions']}\n"
        report += f"**Pozitif Sonuçlar:** {social['positive_outcomes']}\n"
        report += f"**Negatif Sonuçlar:** {social['negative_outcomes']}\n"
        
        negative_signals = social.get('negative_signals', [])
        if negative_signals:
            report += f"\n**Dönem İçi Negatif Sinyaller:** {len(negative_signals)}\n"
            for signal in negative_signals[:5]:
                report += f"- {signal.get('situation', 'unknown')}: {signal.get('what_i_did_wrong', '')[:80]}\n"
        
        never_rules = social.get('never_rules', [])
        if never_rules:
            report += f"\n**Asla Yapma Kuralları:**\n"
            for rule in never_rules[:5]:
                if isinstance(rule, dict):
                    report += f"- 🚫 {rule.get('rule', str(rule))}\n"
                else:
                    report += f"- 🚫 {rule}\n"
        
        # Rüya İçgörüleri
        report += "\n---\n\n## 🌙 Rüya İçgörüleri\n\n"
        dreams = data['dreams']
        report += f"**Toplam Döngü:** {dreams['total_cycles']}\n"
        report += f"**Dönem İçi Rüyalar:** {len(dreams['recent_dreams'])}\n"
        
        recent = dreams.get('recent_dreams', [])
        if recent:
            report += f"\n**Son Rüya Döngüleri:**\n"
            for dream in recent[-5:]:
                report += f"- {dream.get('timestamp', '?')[:10]}: {dream.get('status', '?')} "
                report += f"({dream.get('insights_count', 0)} insights, {dream.get('duration_sec', 0)}s)\n"
        
        # Bilgi Büyümesi
        report += "\n---\n\n## 📈 Bilgi Büyümesi\n\n"
        knowledge = data['knowledge']
        report += f"**Entity'ler:** {knowledge['entities_count']}\n"
        report += f"**İlişkiler:** {knowledge['relations_count']}\n"
        
        topics_dist = knowledge.get('topic_distribution', {})
        if topics_dist:
            report += f"\n**Konu Dağılımı:**\n"
            for topic, count in sorted(topics_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                report += f"- {topic}: {count} anı\n"
        
        # Duygusal Eğilimler
        report += "\n---\n\n## 💭 Duygusal Eğilimler\n\n"
        emotional = data['emotional']
        
        mood_bar = '█' * int(abs(emotional['current_mood']) * 10) + '░' * (10 - int(abs(emotional['current_mood']) * 10))
        mood_label = 'Pozitif' if emotional['current_mood'] > 0 else ('Negatif' if emotional['current_mood'] < 0 else 'Nötr')
        report += f"**Ruh Hali:** {mood_bar} {emotional['current_mood']:.2f} ({mood_label})\n"
        
        conf_bar = '█' * int(emotional['current_confidence'] * 10) + '░' * (10 - int(emotional['current_confidence'] * 10))
        report += f"**Güven:** {conf_bar} {emotional['current_confidence']:.2f}\n"
        
        energy_bar = '█' * int(emotional['current_energy'] * 10) + '░' * (10 - int(emotional['current_energy'] * 10))
        report += f"**Enerji:** {energy_bar} {emotional['current_energy']:.2f}\n"
        
        curiosity_bar = '█' * int(emotional['current_curiosity'] * 10) + '░' * (10 - int(emotional['current_curiosity'] * 10))
        report += f"**Merak:** {curiosity_bar} {emotional['current_curiosity']:.2f}\n"
        
        # Watcher verileri
        watcher = data.get('watcher', {})
        if watcher.get('total_messages', 0) > 0:
            report += "\n---\n\n## 👁️ Watcher Analizi\n\n"
            report += f"**İşlenen Mesaj:** {watcher['total_messages']}\n"
            report += f"**Negatif Sinyaller:** {watcher['negative_signals']}\n"
            
            if watcher.get('emotions'):
                report += f"\n**Duygu Dağılımı:**\n"
                for emo, count in sorted(watcher['emotions'].items(), key=lambda x: x[1], reverse=True):
                    report += f"- {emo}: {count}\n"
            
            if watcher.get('interaction_types'):
                report += f"\n**Etkileşim Tipleri:**\n"
                for itype, count in sorted(watcher['interaction_types'].items(), key=lambda x: x[1], reverse=True):
                    report += f"- {itype}: {count}\n"
        
        # Memory dosyaları
        memory_files = data.get('memory_files', [])
        if memory_files:
            report += "\n---\n\n## 📁 Memory Dosyaları\n\n"
            for mf in memory_files:
                report += f"- **{mf['name']}**: {mf['lines']} satır ({mf['size']} bytes)\n"
        
        # Öneriler
        report += "\n---\n\n## 💡 Öneriler\n\n"
        recommendations = self._generate_recommendations(data)
        for rec in recommendations:
            report += f"- {rec}\n"
        
        report += "\n---\n\n*Rapor otomatik olarak oluşturulmuştur.* 🧿\n"
        
        return report
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """Veriye göre öneriler oluştur."""
        recommendations = []
        
        # Kişilik önerileri
        traits = data['personality'].get('current_traits', {})
        if traits:
            # En düşük trait
            min_trait = min(traits.items(), key=lambda x: x[1])
            if min_trait[1] < 0.5:
                recommendations.append(
                    f"🎭 **{min_trait[0]}** trait'i düşük ({min_trait[1]:.2f}) - "
                    f"Bu alanda daha fazla etkileşim gerekiyor"
                )
        
        # Sosyal öneriler
        social = data['social']
        if social['negative_outcomes'] > social['positive_outcomes'] and social['total_interactions'] > 5:
            recommendations.append(
                "⚠️ Negatif sonuçlar pozitif sonuçlardan fazla - "
                "daha dikkatli dinleme ve anlama gerekiyor"
            )
        
        # Dream önerileri
        dreams = data['dreams']
        if len(dreams['recent_dreams']) == 0:
            recommendations.append(
                "🌙 Dönem içi rüya döngüsü yok - "
                "Dream scheduler'ın çalıştığından emin olun"
            )
        
        # Bilgi önerileri
        knowledge = data['knowledge']
        if knowledge['entities_count'] == 0:
            recommendations.append(
                "📊 World model boş - `python world_model_v2.py` çalıştırarak "
                "MEMORY.md'den bilgi çıkarın"
            )
        
        # Watcher önerileri
        watcher = data.get('watcher', {})
        if watcher.get('total_messages', 0) > 0 and watcher.get('negative_signals', 0) > 3:
            recommendations.append(
                "👁️ Çok fazla negatif sinyal - konuşma stillerini gözden geçirin"
            )
        
        # Varsayılan
        if not recommendations:
            recommendations.append("✅ Sistem normal çalışıyor, belirgin sorun yok")
            recommendations.append("📚 Yeni konular keşfetmek için curiosity engine'i tetikleyin")
        
        return recommendations
    
    def get_summary(self) -> str:
        """Tek paragraflık özet."""
        data = self.get_trend_data(days=7)
        
        personality = data['personality']
        social = data['social']
        dreams = data['dreams']
        emotional = data['emotional']
        
        summary = (
            f"Hacı bu hafta {personality['development_stage']} aşamasında, "
            f"{personality['interaction_count']} etkileşim yaşadı. "
            f"Sosyal zeka: {social['positive_outcomes']} pozitif, "
            f"{social['negative_outcomes']} negatif sonuç. "
            f"{len(dreams['recent_dreams'])} rüya döngüsü çalıştı. "
            f"Ruh hali: {emotional['current_mood']:.2f}, "
            f"enerji: {emotional['current_energy']:.2f}. "
        )
        
        # Bilgi grafiği
        knowledge = data['knowledge']
        if knowledge['entities_count'] > 0:
            summary += f"Bilgi grafiğinde {knowledge['entities_count']} entity, "
            f"{knowledge['relations_count']} ilişki var. "
        
        return summary


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI interface."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description='Weekly Report Generator')
    parser.add_argument('command', choices=['generate', 'summary'],
                       help='generate (rapor oluştur) veya summary (özet)')
    parser.add_argument('--weeks', type=int, default=1, help='Kaç hafta geriye gidilecek')
    parser.add_argument('--workspace', default=None, help='Workspace dizini')
    
    args = parser.parse_args()
    
    generator = WeeklyReportGenerator(args.workspace)
    
    if args.command == 'generate':
        report_path = generator.generate_report(weeks_back=args.weeks)
        print(f"📊 Rapor oluşturuldu: {report_path}")
    elif args.command == 'summary':
        summary = generator.get_summary()
        print(f"📊 Haftalık Özet:\n{summary}")


if __name__ == "__main__":
    main()

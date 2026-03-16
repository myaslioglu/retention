"""
Cognitive Integrator - Tüm bilişsel alt sistemleri birleştiren ana giriş noktası

Tek bir modül ile:
- Watcher (her mesajda çalışır)
- Social Trainer (sosyal öğrenme)
- World Model (bilgi grafiği)
- Dream Scheduler (gece rüya döngüleri)
- Weekly Report (haftalık raporlar)

CLI komutları:
    python cognitive_integrator.py init      # Alt sistemleri başlat
    python cognitive_integrator.py process "mesaj"  # Mesaj işle
    python cognitive_integrator.py status    # Sistem durumu
    python cognitive_integrator.py report    # Haftalık rapor
    python cognitive_integrator.py dream     # Rüya döngüsü
    python cognitive_integrator.py populate  # World model doldur
"""

import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent


class CognitiveIntegrator:
    """
    Tüm bilişsel alt sistemleri birleştiren ana sınıf.
    
    Pipeline:
    1. Gelen mesaj → Watcher (hızlı analiz)
    2. Watcher → Social Trainer (sosyal öğrenme)
    3. Social Trainer → Memory Store (kayıt)
    4. Periyodik: Dream check, World Model update, Report
    """
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = str(_WORKSPACE_DIR)
        
        self.workspace = Path(workspace_dir)
        self.state_dir = self.workspace / "cognitive_state"
        self.state_dir.mkdir(exist_ok=True)
        
        self.integrator_state_file = self.state_dir / "integrator_state.json"
        
        # Alt sistemler (lazy loaded)
        self._watcher = None
        self._social_trainer = None
        self._world_model = None
        self._dream_scheduler = None
        self._report_generator = None
        
        # Durum
        self.state = self._load_integrator_state()
        
        logger.info("🔗 CognitiveIntegrator initialized")
    
    def _load_integrator_state(self) -> Dict:
        """Integrator durumunu yükle."""
        if self.integrator_state_file.exists():
            try:
                with open(self.integrator_state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'initialized': False,
            'init_time': None,
            'total_messages_processed': 0,
            'total_dream_cycles': 0,
            'total_reports_generated': 0,
            'last_process': None,
            'last_dream_check': None,
            'last_report': None,
            'last_world_model_update': None,
            'subsystems': {
                'watcher': False,
                'social_trainer': False,
                'world_model': False,
                'dream_scheduler': False,
                'report_generator': False,
            },
        }
    
    def _save_integrator_state(self):
        """Integrator durumunu kaydet."""
        with open(self.integrator_state_file, 'w') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    # === Alt Sistem Erişimi (Lazy Loading) ===
    
    def _get_watcher(self):
        """Watcher modülünü yükle."""
        if self._watcher is None:
            try:
                import sys
                if str(_MODULE_DIR) not in sys.path:
                    sys.path.insert(0, str(_MODULE_DIR))
                import cognitive_watcher
                self._watcher = cognitive_watcher
                self.state['subsystems']['watcher'] = True
                logger.info("✅ Watcher loaded")
            except Exception as e:
                logger.error(f"❌ Watcher load failed: {e}")
        return self._watcher
    
    def _get_social_trainer(self):
        """Social trainer'ı yükle."""
        if self._social_trainer is None:
            try:
                import sys
                if str(_MODULE_DIR) not in sys.path:
                    sys.path.insert(0, str(_MODULE_DIR))
                from social_trainer import SocialIntelligenceTrainer
                self._social_trainer = SocialIntelligenceTrainer(
                    state_dir=str(self.state_dir)
                )
                self.state['subsystems']['social_trainer'] = True
                logger.info("✅ Social Trainer loaded")
            except Exception as e:
                logger.error(f"❌ Social Trainer load failed: {e}")
        return self._social_trainer
    
    def _get_world_model(self):
        """World model'i yükle."""
        if self._world_model is None:
            try:
                import sys
                if str(_MODULE_DIR) not in sys.path:
                    sys.path.insert(0, str(_MODULE_DIR))
                from world_model_v2 import WorldModelV2
                self._world_model = WorldModelV2(str(self.workspace))
                self.state['subsystems']['world_model'] = True
                logger.info("✅ World Model V2 loaded")
            except Exception as e:
                logger.error(f"❌ World Model load failed: {e}")
        return self._world_model
    
    def _get_dream_scheduler(self):
        """Dream scheduler'ı yükle."""
        if self._dream_scheduler is None:
            try:
                import sys
                if str(_MODULE_DIR) not in sys.path:
                    sys.path.insert(0, str(_MODULE_DIR))
                from dream_scheduler import DreamScheduler
                self._dream_scheduler = DreamScheduler(str(self.workspace))
                self.state['subsystems']['dream_scheduler'] = True
                logger.info("✅ Dream Scheduler loaded")
            except Exception as e:
                logger.error(f"❌ Dream Scheduler load failed: {e}")
        return self._dream_scheduler
    
    def _get_report_generator(self):
        """Report generator'ı yükle."""
        if self._report_generator is None:
            try:
                import sys
                if str(_MODULE_DIR) not in sys.path:
                    sys.path.insert(0, str(_MODULE_DIR))
                from weekly_report import WeeklyReportGenerator
                self._report_generator = WeeklyReportGenerator(str(self.workspace))
                self.state['subsystems']['report_generator'] = True
                logger.info("✅ Report Generator loaded")
            except Exception as e:
                logger.error(f"❌ Report Generator load failed: {e}")
        return self._report_generator
    
    # === Ana Komutlar ===
    
    def init(self) -> Dict:
        """
        Tüm alt sistemleri başlat ve doğrula.
        """
        logger.info("🔗 === COGNITIVE INIT BAŞLATILIYOR ===")
        
        results = {
            'watcher': False,
            'social_trainer': False,
            'world_model': False,
            'dream_scheduler': False,
            'report_generator': False,
        }
        
        # Her alt sistemi yükle
        if self._get_watcher():
            results['watcher'] = True
        
        if self._get_social_trainer():
            results['social_trainer'] = True
        
        if self._get_world_model():
            results['world_model'] = True
        
        if self._get_dream_scheduler():
            results['dream_scheduler'] = True
        
        if self._get_report_generator():
            results['report_generator'] = True
        
        # Durum güncelle
        self.state['initialized'] = True
        self.state['init_time'] = datetime.now().isoformat()
        self._save_integrator_state()
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"🔗 === INIT TAMAMLANDI: {success_count}/5 alt sistem ===")
        
        return results
    
    def process_incoming(self, text: str, sender_id: str = "unknown") -> Dict:
        """
        Gelen mesajı işle - ana pipeline.
        
        Pipeline:
        1. Watcher analizi (hızlı)
        2. Social trainer (negatif sinyal, öğrenme)
        3. Sonuç kaydet
        
        Args:
            text: Mesaj içeriği
            sender_id: Gönderen ID
            
        Returns:
            İşleme sonuçları
        """
        start_time = datetime.now()
        result = {
            'text_preview': text[:80] + ('...' if len(text) > 80 else ''),
            'sender_id': sender_id,
            'timestamp': start_time.isoformat(),
            'watcher_result': None,
            'social_result': None,
            'pipeline_ms': 0,
        }
        
        # 1. Watcher analizi
        watcher = self._get_watcher()
        if watcher:
            try:
                watcher_result = watcher.on_message(text, sender_id)
                result['watcher_result'] = watcher_result
            except Exception as e:
                logger.warning(f"Watcher error: {e}")
                result['watcher_error'] = str(e)
        
        # 2. Social trainer (negatif sinyal)
        social = self._get_social_trainer()
        if social:
            try:
                signal = social.detect_and_learn_from_message(text)
                if signal:
                    result['social_result'] = {
                        'signal': signal.get('signal'),
                        'severity': signal.get('severity'),
                        'trigger': signal.get('trigger', ''),
                    }
            except Exception as e:
                logger.warning(f"Social trainer error: {e}")
                result['social_error'] = str(e)
        
        # Pipeline süresi
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        result['pipeline_ms'] = round(elapsed, 1)
        
        # State güncelle
        self.state['total_messages_processed'] += 1
        self.state['last_process'] = start_time.isoformat()
        self._save_integrator_state()
        
        return result
    
    def get_status(self) -> Dict:
        """Tüm sistemin durumunu getir."""
        status = {
            'integrator': {
                'initialized': self.state.get('initialized', False),
                'init_time': self.state.get('init_time'),
                'total_messages': self.state.get('total_messages_processed', 0),
                'last_process': self.state.get('last_process'),
            },
            'subsystems': self.state.get('subsystems', {}),
        }
        
        # Watcher stats
        watcher = self._get_watcher()
        if watcher:
            try:
                status['watcher_stats'] = watcher.get_watcher_stats()
            except Exception:
                status['watcher_stats'] = {'error': 'unavailable'}
        
        # Dream scheduler summary
        dream = self._get_dream_scheduler()
        if dream:
            try:
                status['dream_summary'] = dream.get_dream_summary()
            except Exception:
                status['dream_summary'] = {'error': 'unavailable'}
        
        # World model stats
        wm = self._get_world_model()
        if wm:
            try:
                status['world_model_stats'] = wm.get_stats()
            except Exception:
                status['world_model_stats'] = {'error': 'unavailable'}
        
        # Cognitive state
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    cognitive_state = json.load(f)
                status['cognitive_state'] = {
                    'personality': cognitive_state.get('personality', {}),
                    'emotion': cognitive_state.get('emotion', {}),
                    'world_model': cognitive_state.get('world_model', {}),
                    'metacognition': cognitive_state.get('metacognition', {}),
                }
            except Exception:
                pass
        
        return status
    
    def run_report(self) -> str:
        """Haftalık rapor oluştur."""
        generator = self._get_report_generator()
        if not generator:
            return "❌ Report generator not available"
        
        report_path = generator.generate_report()
        
        self.state['total_reports_generated'] += 1
        self.state['last_report'] = datetime.now().isoformat()
        self._save_integrator_state()
        
        return report_path
    
    def run_dream(self) -> Dict:
        """Rüya döngüsü çalıştır."""
        scheduler = self._get_dream_scheduler()
        if not scheduler:
            return {'error': 'Dream scheduler not available'}
        
        result = scheduler.run_dream_cycle()
        
        self.state['total_dream_cycles'] += 1
        self._save_integrator_state()
        
        return result
    
    def run_populate(self) -> Dict:
        """World model'i MEMORY.md'den doldur."""
        wm = self._get_world_model()
        if not wm:
            return {'error': 'World model not available'}
        
        # MEMORY.md'den populate
        wm.populate_from_memory()
        
        # Günlük dosyaları ekle
        memory_dir = self.workspace / "memory"
        if memory_dir.exists():
            for f in sorted(memory_dir.glob("*.md"), reverse=True)[:7]:
                wm.update_from_daily(str(f))
        
        self.state['last_world_model_update'] = datetime.now().isoformat()
        self._save_integrator_state()
        
        stats = wm.get_stats()
        return {
            'status': 'completed',
            'graph_nodes': stats['graph']['nodes'],
            'graph_edges': stats['graph']['edges'],
            'topics': stats['topics']['total_clusters'],
            'events': stats['timeline']['total_events'],
        }
    
    def periodic_tasks(self) -> Dict:
        """
        Periyodik görevleri kontrol et ve çalıştır.
        
        - Dream check (eğer zamanı geldiyse)
        - World model update (günlük)
        - Report generation (haftalık)
        """
        results = {
            'dream': None,
            'world_model': None,
            'report': None,
        }
        
        now = datetime.now()
        
        # Dream check
        dream = self._get_dream_scheduler()
        if dream and dream.should_run_dream():
            logger.info("🌙 Dream cycle time - running...")
            results['dream'] = self.run_dream()
            self.state['last_dream_check'] = now.isoformat()
        
        # World model update (günde bir)
        last_wm = self.state.get('last_world_model_update')
        if last_wm:
            try:
                last_dt = datetime.fromisoformat(last_wm)
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since > 24:
                    logger.info("📊 World model update needed...")
                    results['world_model'] = self.run_populate()
            except Exception:
                pass
        else:
            results['world_model'] = self.run_populate()
        
        # Report generation (haftada bir - Pazartesi)
        if now.weekday() == 0:  # Pazartesi
            last_report = self.state.get('last_report')
            if last_report:
                try:
                    last_dt = datetime.fromisoformat(last_report)
                    days_since = (now - last_dt).total_seconds() / 86400
                    if days_since >= 7:
                        logger.info("📊 Weekly report generation...")
                        results['report'] = self.run_report()
                except Exception:
                    pass
            else:
                results['report'] = self.run_report()
        
        self._save_integrator_state()
        return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI interface."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(
        description='Cognitive Integrator - Tüm bilişsel alt sistemleri birleştirir'
    )
    parser.add_argument('command', 
                       choices=['init', 'process', 'status', 'report', 'dream', 'populate', 'periodic'],
                       help='Komut')
    parser.add_argument('text', nargs='?', default='', help='Process komutu için mesaj')
    parser.add_argument('--sender', default='cli', help='Gönderen ID')
    parser.add_argument('--workspace', default=None, help='Workspace dizini')
    
    args = parser.parse_args()
    
    integrator = CognitiveIntegrator(args.workspace)
    
    if args.command == 'init':
        print("🔗 Alt sistemler başlatılıyor...\n")
        results = integrator.init()
        for system, ok in results.items():
            status = "✅" if ok else "❌"
            print(f"   {status} {system}")
        print(f"\n✅ {sum(1 for v in results.values() if v)}/5 sistem aktif")
    
    elif args.command == 'process':
        if not args.text:
            print("❌ Mesaj gerekli: python cognitive_integrator.py process \"mesaj\"")
            sys.exit(1)
        print(f"📨 İşleniyor: \"{args.text[:60]}...\"\n")
        result = integrator.process_incoming(args.text, args.sender)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.command == 'status':
        print("📊 Sistem Durumu:\n")
        status = integrator.get_status()
        
        # Integrator
        integ = status['integrator']
        print(f"🔗 Integrator:")
        print(f"   Başlatıldı: {'✅' if integ['initialized'] else '❌'}")
        print(f"   İşlenen mesaj: {integ['total_messages']}")
        print(f"   Son işlem: {integ['last_process'] or 'Hiç'}")
        
        # Subsystems
        print(f"\n📦 Alt Sistemler:")
        for sys_name, ok in status['subsystems'].items():
            print(f"   {'✅' if ok else '❌'} {sys_name}")
        
        # Watcher
        ws = status.get('watcher_stats', {})
        if ws and ws.get('total_messages', 0) > 0:
            print(f"\n👁️ Watcher:")
            print(f"   Toplam mesaj: {ws['total_messages']}")
            print(f"   Negatif sinyal: {ws.get('negative_signals', 0)}")
            print(f"   Ortalama süre: {ws.get('avg_processing_ms', 0)}ms")
        
        # Dream
        ds = status.get('dream_summary', {})
        if ds and ds.get('stats', {}).get('total_cycles', 0) > 0:
            print(f"\n🌙 Dream Scheduler:")
            print(f"   Toplam döngü: {ds['stats']['total_cycles']}")
            print(f"   Insights: {ds['stats']['total_insights']}")
        
        # Cognitive state
        cs = status.get('cognitive_state', {})
        if cs:
            print(f"\n🧠 Cognitive State:")
            pers = cs.get('personality', {})
            if pers:
                top_traits = sorted(pers.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top traits: {', '.join(f'{k}={v:.2f}' for k, v in top_traits)}")
            emo = cs.get('emotion', {})
            if emo:
                print(f"   Mood: {emo.get('mood', 0):.2f}, Energy: {emo.get('energy', 0):.2f}")
    
    elif args.command == 'report':
        print("📊 Haftalık rapor oluşturuluyor...\n")
        report_path = integrator.run_report()
        print(f"✅ Rapor: {report_path}")
    
    elif args.command == 'dream':
        print("🌙 Rüya döngüsü çalıştırılıyor...\n")
        result = integrator.run_dream()
        print(f"Durum: {result.get('status', 'unknown')}")
        print(f"Süre: {result.get('duration_sec', 0)}s")
        print(f"Insights: {len(result.get('insights', []))}")
    
    elif args.command == 'populate':
        print("📊 World model dolduruluyor...\n")
        result = integrator.run_populate()
        print(f"Durum: {result.get('status', 'unknown')}")
        print(f"Nodes: {result.get('graph_nodes', 0)}")
        print(f"Edges: {result.get('graph_edges', 0)}")
        print(f"Konular: {result.get('topics', 0)}")
        print(f"Olaylar: {result.get('events', 0)}")
    
    elif args.command == 'periodic':
        print("⏰ Periyodik görevler kontrol ediliyor...\n")
        results = integrator.periodic_tasks()
        for task, result in results.items():
            if result:
                print(f"   ✅ {task}: {result.get('status', 'completed') if isinstance(result, dict) else 'done'}")
            else:
                print(f"   ⏭️ {task}: gerekmiyor")


if __name__ == "__main__":
    main()

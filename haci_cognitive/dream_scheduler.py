"""
Dream Scheduler - Otomatik gece rüya döngüleri

Her gece belirli bir saatte (varsayılan 03:00) otomatik çalışır:
- DreamingLoop'u tetikler
- Sonuçları MEMORY.md'ye konsolide eder
- Cognitive state'i günceller
- Rüya sonuçlarını dream_log.json'a kaydet

Kullanım:
    python dream_scheduler.py run      # Manuel tetikleme
    python dream_scheduler.py check    # Çalışma zamanı mı kontrol et
    python dream_scheduler.py summary  # Rüya özetlerini göster
"""

import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Modül dizini
_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent


class DreamScheduler:
    """
    Otomatik rüya döngüsü zamanlayıcı.
    
    DreamingLoop'u gece saatlerinde otomatik tetikler,
    sonuçları toplar ve cognitive state'i günceller.
    """
    
    # Varsayılan ayarlar
    DEFAULT_DREAM_HOUR = 3      # 03:00
    DEFAULT_DREAM_MINUTE = 0
    TIMEZONE_OFFSET = 3         # Istanbul UTC+3
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = str(_WORKSPACE_DIR)
        
        self.workspace = Path(workspace_dir)
        self.state_dir = self.workspace / "cognitive_state"
        self.state_dir.mkdir(exist_ok=True)
        
        # State dosyaları
        self.schedule_file = self.state_dir / "dream_schedule.json"
        self.log_file = self.state_dir / "dream_log.json"
        
        # Schedule state
        self.schedule = self._load_schedule()
        
        # Dream log
        self.dream_log = self._load_dream_log()
        
        logger.info(f"🌙 DreamScheduler initialized (dream_hour={self.DEFAULT_DREAM_HOUR}:"
                    f"{self.DEFAULT_DREAM_MINUTE:02d} IST)")
    
    def _load_schedule(self) -> Dict:
        """Zamanlama durumunu yükle."""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Schedule yüklenemedi: {e}")
        
        return {
            'last_run': None,
            'next_run': self._calculate_next_run(),
            'total_cycles': 0,
            'insights_count': 0,
            'enabled': True,
            'dream_hour': self.DEFAULT_DREAM_HOUR,
            'dream_minute': self.DEFAULT_DREAM_MINUTE,
        }
    
    def _save_schedule(self):
        """Zamanlama durumunu kaydet."""
        with open(self.schedule_file, 'w') as f:
            json.dump(self.schedule, f, indent=2, ensure_ascii=False)
    
    def _load_dream_log(self) -> List[Dict]:
        """Rüya log'unu yükle."""
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    log = json.load(f)
                # Son 50 kayıt
                return log[-50:] if len(log) > 50 else log
            except Exception:
                return []
        return []
    
    def _save_dream_log(self):
        """Rüya log'unu kaydet."""
        with open(self.log_file, 'w') as f:
            json.dump(self.dream_log, f, indent=2, ensure_ascii=False)
    
    def _calculate_next_run(self) -> str:
        """Bir sonraki çalışma zamanını hesapla."""
        now = datetime.now()
        
        # Bugün için dream time
        dream_time = now.replace(
            hour=self.DEFAULT_DREAM_HOUR,
            minute=self.DEFAULT_DREAM_MINUTE,
            second=0,
            microsecond=0,
        )
        
        # Eğer dream time geçtiyse, yarına ayarla
        if now >= dream_time:
            dream_time += timedelta(days=1)
        
        return dream_time.isoformat()
    
    def should_run_dream(self) -> bool:
        """
        Rüya döngüsü çalışmalı mı kontrol et.
        
        Koşullar:
        1. Enabled olmalı
        2. Son çalışmadan bu yana 20+ saat geçmiş olmalı
        3. Şu anki zaman dream_window içinde olmalı (±2 saat)
        """
        if not self.schedule.get('enabled', True):
            return False
        
        now = datetime.now()
        
        # Son çalışma kontrolü
        last_run = self.schedule.get('last_run')
        if last_run:
            last_dt = datetime.fromisoformat(last_run)
            hours_since = (now - last_dt).total_seconds() / 3600
            if hours_since < 20:  # En az 20 saat bekle
                return False
        
        # Dream window kontrolü (±2 saat)
        dream_hour = self.schedule.get('dream_hour', self.DEFAULT_DREAM_HOUR)
        dream_minute = self.schedule.get('dream_minute', self.DEFAULT_DREAM_MINUTE)
        
        # Dream zamanına 2 saatlik pencere
        dream_start = (dream_hour - 2) % 24
        dream_end = (dream_hour + 2) % 24
        current_hour = now.hour
        
        if dream_start <= dream_end:
            in_window = dream_start <= current_hour < dream_end
        else:
            # Gece yarısını aşan pencere
            in_window = current_hour >= dream_start or current_hour < dream_end
        
        return in_window
    
    def run_dream_cycle(self) -> Dict:
        """
        Rüya döngüsünü çalıştır.
        
        1. DreamingLoop'u başlat
        2. Sonuçları topla
        3. MEMORY.md'ye konsolide et
        4. Cognitive state'i güncelle
        5. Log'a kaydet
        """
        logger.info("🌙 === DREAM CYCLE BAŞLATILIYOR ===")
        
        start_time = datetime.now()
        result = {
            'status': 'started',
            'timestamp': start_time.isoformat(),
            'insights': [],
            'connections': [],
            'errors': [],
        }
        
        try:
            # 1. DreamingLoop'u import et ve çalıştır
            import sys
            if str(_MODULE_DIR) not in sys.path:
                sys.path.insert(0, str(_MODULE_DIR))
            
            from dreaming_loop import DreamingLoop, AutonomousDreamRunner
            
            # AutonomousDreamRunner kullan
            runner = AutonomousDreamRunner(str(self.workspace))
            dream_report = runner.run(max_cycles=1)
            
            if dream_report:
                result['dream_report'] = dream_report
                result['insights'] = dream_report.get('insights', [])
                result['memories_processed'] = dream_report.get('memories_processed', 0)
                result['phases'] = [p.get('type', 'unknown') for p in dream_report.get('phases', [])]
                
                # Connection sayısını çıkar
                for phase in dream_report.get('phases', []):
                    if phase.get('type') == 'exploration':
                        result['connections'] = phase.get('connections', [])
                
                result['status'] = 'completed'
            else:
                result['status'] = 'no_memories'
                result['errors'].append('No memories available for dreaming')
        
        except ImportError as e:
            result['status'] = 'import_error'
            result['errors'].append(f"DreamingLoop import error: {e}")
            logger.error(f"DreamingLoop import hatası: {e}")
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            logger.error(f"Dream cycle hatası: {e}")
        
        # 2. Duration hesapla
        duration = (datetime.now() - start_time).total_seconds()
        result['duration_sec'] = round(duration, 1)
        
        # 3. Schedule güncelle
        self.schedule['last_run'] = start_time.isoformat()
        self.schedule['next_run'] = self._calculate_next_run()
        self.schedule['total_cycles'] += 1
        self.schedule['insights_count'] += len(result.get('insights', []))
        self._save_schedule()
        
        # 4. Dream log'a ekle
        self.dream_log.append({
            'timestamp': start_time.isoformat(),
            'status': result['status'],
            'duration_sec': result['duration_sec'],
            'insights_count': len(result.get('insights', [])),
            'memories_processed': result.get('memories_processed', 0),
            'errors': result.get('errors', []),
        })
        self._save_dream_log()
        
        # 5. Cognitive state güncelle
        self._update_cognitive_state(result)
        
        logger.info(f"🌙 === DREAM CYCLE TAMAMLANDI ({result['status']}, {result['duration_sec']}s) ===")
        
        return result
    
    def _update_cognitive_state(self, dream_result: Dict):
        """Cognitive state dosyasını rüya sonuçlarıyla güncelle."""
        state_file = self.state_dir / "state.json"
        
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                
                # Dream cycle sayısını güncelle
                if 'metacognition' not in state:
                    state['metacognition'] = {}
                state['metacognition']['total_dream_cycles'] = \
                    state['metacognition'].get('total_dream_cycles', 0) + 1
                
                # Son dream zamanı
                if 'timestamps' not in state:
                    state['timestamps'] = {}
                state['timestamps']['last_dream'] = datetime.now().isoformat()
                
                # World model istatistikleri
                if 'world_model' in state:
                    state['world_model']['n_insights_generated'] = \
                        state['world_model'].get('n_insights_generated', 0) + \
                        len(dream_result.get('insights', []))
                    state['world_model']['n_connections_found'] = \
                        state['world_model'].get('n_connections_found', 0) + \
                        len(dream_result.get('connections', []))
                
                state['last_updated'] = datetime.now().isoformat()
                
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                
                logger.info("📊 Cognitive state güncellendi")
            except Exception as e:
                logger.warning(f"State güncelleme hatası: {e}")
    
    def get_dream_summary(self, n: int = 5) -> Dict:
        """
        Son rüya döngülerinin özetini getir.
        """
        recent = self.dream_log[-n:] if self.dream_log else []
        
        total_insights = sum(entry.get('insights_count', 0) for entry in self.dream_log)
        total_duration = sum(entry.get('duration_sec', 0) for entry in self.dream_log)
        
        success_count = sum(1 for e in self.dream_log if e.get('status') == 'completed')
        error_count = sum(1 for e in self.dream_log if e.get('status') == 'error')
        
        return {
            'schedule': {
                'enabled': self.schedule.get('enabled', True),
                'last_run': self.schedule.get('last_run'),
                'next_run': self.schedule.get('next_run'),
                'dream_hour': self.schedule.get('dream_hour', self.DEFAULT_DREAM_HOUR),
            },
            'stats': {
                'total_cycles': len(self.dream_log),
                'successful_cycles': success_count,
                'failed_cycles': error_count,
                'total_insights': total_insights,
                'total_duration_min': round(total_duration / 60, 1),
                'avg_duration_sec': round(total_duration / max(len(self.dream_log), 1), 1),
            },
            'recent_cycles': recent,
        }
    
    def schedule_next(self) -> str:
        """Bir sonraki çalışma zamanını hesapla ve kaydet."""
        next_run = self._calculate_next_run()
        self.schedule['next_run'] = next_run
        self._save_schedule()
        return next_run
    
    def enable(self):
        """Rüya döngüsünü etkinleştir."""
        self.schedule['enabled'] = True
        self._save_schedule()
        logger.info("🌙 Dream scheduler ENABLED")
    
    def disable(self):
        """Rüya döngüsünü devre dışı bırak."""
        self.schedule['enabled'] = False
        self._save_schedule()
        logger.info("🌙 Dream scheduler DISABLED")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI interface."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description='Dream Scheduler - Otomatik rüya döngüleri')
    parser.add_argument('command', choices=['run', 'check', 'summary', 'enable', 'disable', 'schedule'],
                       help='Komut: run (manuel tetik), check (zamanı mı?), summary (özet)')
    parser.add_argument('--workspace', default=None, help='Workspace dizini')
    
    args = parser.parse_args()
    
    scheduler = DreamScheduler(args.workspace)
    
    if args.command == 'run':
        print("🌙 Manuel rüya döngüsü çalıştırılıyor...\n")
        result = scheduler.run_dream_cycle()
        print(f"Durum: {result['status']}")
        print(f"Süre: {result['duration_sec']}s")
        print(f"Insights: {len(result.get('insights', []))}")
        if result.get('errors'):
            print(f"Hatalar: {result['errors']}")
    
    elif args.command == 'check':
        should = scheduler.should_run_dream()
        print(f"Rüya döngüsü çalışmalı mı? {'✅ EVET' if should else '❌ HAYIR'}")
        print(f"Son çalışma: {scheduler.schedule.get('last_run', 'Hiç')}")
        print(f"Sonraki çalışma: {scheduler.schedule.get('next_run', 'Hesaplanmadı')}")
    
    elif args.command == 'summary':
        summary = scheduler.get_dream_summary()
        print("📊 Dream Scheduler Özeti\n")
        print(f"Durum: {'✅ Enabled' if summary['schedule']['enabled'] else '❌ Disabled'}")
        print(f"Son çalışma: {summary['schedule']['last_run'] or 'Hiç'}")
        print(f"Sonraki çalışma: {summary['schedule']['next_run'] or 'Hesaplanmadı'}")
        print(f"\nToplam döngü: {summary['stats']['total_cycles']}")
        print(f"Başarılı: {summary['stats']['successful_cycles']}")
        print(f"Başarısız: {summary['stats']['failed_cycles']}")
        print(f"Toplam insights: {summary['stats']['total_insights']}")
        print(f"Ortalama süre: {summary['stats']['avg_duration_sec']}s")
    
    elif args.command == 'enable':
        scheduler.enable()
        print("🌙 Dream scheduler etkinleştirildi")
    
    elif args.command == 'disable':
        scheduler.disable()
        print("🌙 Dream scheduler devre dışı bırakıldı")
    
    elif args.command == 'schedule':
        next_run = scheduler.schedule_next()
        print(f"Sonraki çalışma zamanı: {next_run}")


if __name__ == "__main__":
    main()

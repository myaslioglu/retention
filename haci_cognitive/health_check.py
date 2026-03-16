"""
Health Check - Tüm modüllerin durumunu kontrol et

Heartbeat tarafından kullanılır. Hızlı olmalı (<500ms).

Kullanım:
    python health_check.py
    python health_check.py --json
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent
_STATE_DIR = _WORKSPACE_DIR / "cognitive_state"


def _check_file_age(file_path: Path, max_hours: int = 24) -> Dict:
    """Dosya yaşını kontrol et."""
    if not file_path.exists():
        return {'exists': False, 'status': '❌', 'msg': 'Dosya yok'}
    
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age_hours = (datetime.now() - mtime).total_seconds() / 3600
    
    if age_hours > max_hours:
        return {'exists': True, 'status': '⚠️', 'msg': f'{age_hours:.1f}h önce', 'age_hours': age_hours}
    return {'exists': True, 'status': '✅', 'msg': f'{age_hours:.1f}h önce', 'age_hours': age_hours}


def check_all() -> Dict[str, Dict]:
    """Tüm modülleri kontrol et."""
    results = {}
    
    # 1. CognitiveWatcher
    results['cognitive_watcher'] = {
        **_check_file_age(_STATE_DIR / "watcher_log.json", max_hours=12),
        'description': 'Mesaj analiz motoru',
    }
    
    # 2. WorldModel
    results['world_model'] = {
        **_check_file_age(_STATE_DIR / "world_model_sync.json", max_hours=48),
        'description': 'Bilgi grafiği',
    }
    # Entity count
    wm_file = _STATE_DIR / "world_model_knowledge.json"
    if wm_file.exists():
        try:
            with open(wm_file) as f:
                wm = json.load(f)
            results['world_model']['entities'] = len(wm.get('graph', {}).get('nodes', {}))
            results['world_model']['relations'] = len(wm.get('graph', {}).get('edges', []))
        except Exception:
            pass
    
    # 3. DreamScheduler
    dream_file = _STATE_DIR / "dream_schedule.json"
    if dream_file.exists():
        try:
            with open(dream_file) as f:
                dream = json.load(f)
            enabled = dream.get('enabled', True)
            next_run = dream.get('next_run', '?')
            results['dream_scheduler'] = {
                'exists': True,
                'status': '✅' if enabled else '⏸️',
                'msg': f'Enabled: {enabled}, Next: {next_run}',
                'description': 'Gece rüya döngüleri',
                'total_cycles': dream.get('total_cycles', 0),
            }
        except Exception:
            results['dream_scheduler'] = {'exists': False, 'status': '❌', 'msg': 'Parse error'}
    else:
        results['dream_scheduler'] = {'exists': False, 'status': '❌', 'msg': 'Dosya yok'}
    results['dream_scheduler'] = results.get('dream_scheduler', {})
    results['dream_scheduler']['description'] = results['dream_scheduler'].get('description', 'Gece rüya döngüleri')
    
    # 4. SocialTrainer
    results['social_trainer'] = {
        **_check_file_age(_STATE_DIR / "social_trainer_state.json", max_hours=168),
        'description': 'Sosyal zeka trainer',
    }
    st_file = _STATE_DIR / "social_trainer_state.json"
    if st_file.exists():
        try:
            with open(st_file) as f:
                st = json.load(f)
            results['social_trainer']['interactions'] = st.get('total_interactions', 0)
        except Exception:
            pass
    
    # 5. Memory Clusters
    results['memory_clusterer'] = {
        **_check_file_age(_STATE_DIR / "memory_clusters.json", max_hours=72),
        'description': 'Memory topic clustering',
    }
    mc_file = _STATE_DIR / "memory_clusters.json"
    if mc_file.exists():
        try:
            with open(mc_file) as f:
                mc = json.load(f)
            results['memory_clusterer']['topics'] = mc.get('stats', {}).get('topic_count', 0)
            results['memory_clusterer']['files'] = mc.get('stats', {}).get('total_files', 0)
        except Exception:
            pass
    
    # 6. Sender Profiles
    sp_file = _STATE_DIR / "sender_profiles.json"
    if sp_file.exists():
        try:
            with open(sp_file) as f:
                sp = json.load(f)
            results['sender_profiles'] = {
                'exists': True,
                'status': '✅',
                'msg': f'{len(sp)} profil',
                'description': 'Sender emotional baselines',
                'profile_count': len(sp),
                'senders': list(sp.keys()),
            }
        except Exception:
            results['sender_profiles'] = {'exists': False, 'status': '❌', 'msg': 'Parse error'}
    else:
        results['sender_profiles'] = {'exists': False, 'status': '⚠️', 'msg': 'Henüz profil yok'}
    results['sender_profiles'] = results.get('sender_profiles', {})
    results['sender_profiles']['description'] = results['sender_profiles'].get('description', 'Sender emotional baselines')
    
    # 7. FAISS Index
    faiss_file = _WORKSPACE_DIR.parent / "memory" / "faiss.index"
    results['faiss_index'] = {
        **_check_file_age(faiss_file, max_hours=72),
        'description': 'Memory search index',
    }
    
    # 8. Disk usage
    state_size = sum(f.stat().st_size for f in _STATE_DIR.glob("*.json") if f.is_file())
    results['disk_usage'] = {
        'exists': True,
        'status': '✅' if state_size < 50_000_000 else '⚠️',  # 50MB threshold
        'msg': f'{state_size / 1024 / 1024:.1f} MB',
        'description': 'cognitive_state/ disk usage',
    }
    
    return results


def get_summary(results: Dict[str, Dict]) -> str:
    """Sağlık özeti oluştur."""
    lines = ["🏥 **SİSTEM SAĞLIK DURUMU**\n"]
    
    all_ok = True
    for module, data in results.items():
        status = data.get('status', '❓')
        desc = data.get('description', module)
        msg = data.get('msg', '')
        
        if status != '✅':
            all_ok = False
        
        lines.append(f"{status} **{desc}** — {msg}")
        
        # Extra details
        if 'entities' in data:
            lines.append(f"   📊 {data['entities']} entities, {data['relations']} relations")
        if 'interactions' in data:
            lines.append(f"   📊 {data['interactions']} interactions")
        if 'total_cycles' in data:
            lines.append(f"   📊 {data['total_cycles']} dream cycles")
        if 'topics' in data:
            lines.append(f"   📊 {data['topics']} topics, {data['files']} files")
        if 'profile_count' in data:
            lines.append(f"   📊 Senders: {', '.join(data.get('senders', []))}")
    
    lines.append(f"\n{'✅ Tüm sistemler normal!' if all_ok else '⚠️ Bazı sistemlerde sorun var!'}")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hacı Health Check')
    parser.add_argument('--json', action='store_true', help='JSON output')
    args = parser.parse_args()
    
    results = check_all()
    
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(get_summary(results))


if __name__ == "__main__":
    main()

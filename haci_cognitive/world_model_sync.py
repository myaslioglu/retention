"""
WorldModel Sync - Otomatik günlük dosya senkronizasyonu

Memory consolidation ile entegre çalışır:
- Yeni günlük dosyaları tespit eder
- WorldModel'i incremental günceller
- MEMORY.md değişikliklerini de takip eder

Kullanım:
    python world_model_sync.py                    # Otomatik sync
    python world_model_sync.py --file memory/2026-03-17.md  # Belirli dosya
    python world_model_sync.py --full             # Tam yeniden yükle
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Modül dizini
_MODULE_DIR = Path(__file__).parent
_WORKSPACE_DIR = _MODULE_DIR.parent


def get_last_sync_time(state_dir: Path) -> Optional[datetime]:
    """Son sync zamanını al."""
    sync_file = state_dir / "world_model_sync.json"
    if sync_file.exists():
        try:
            with open(sync_file) as f:
                data = json.load(f)
            return datetime.fromisoformat(data.get('last_sync', '2000-01-01'))
        except Exception:
            pass
    return None


def update_last_sync(state_dir: Path, file_count: int, entity_count: int, relation_count: int):
    """Sync zamanını güncelle."""
    sync_file = state_dir / "world_model_sync.json"
    with open(sync_file, 'w') as f:
        json.dump({
            'last_sync': datetime.now().isoformat(),
            'files_processed': file_count,
            'entities_added': entity_count,
            'relations_added': relation_count,
        }, f, indent=2)


def sync(workspace_dir: str = None, target_file: str = None, full: bool = False):
    """WorldModel'i günlük dosyalarla senkronize et."""
    if workspace_dir is None:
        workspace_dir = str(_WORKSPACE_DIR)
    
    workspace = Path(workspace_dir)
    memory_dir = workspace / "memory"
    state_dir = workspace / "cognitive_state"
    state_dir.mkdir(exist_ok=True)
    
    # WorldModel'i import et
    sys.path.insert(0, str(_MODULE_DIR))
    from world_model_v2 import WorldModelV2
    
    model = WorldModelV2(str(workspace))
    
    # Kullanılabilir günlük dosyaları bul
    if target_file:
        daily_files = [Path(target_file)]
    elif full:
        # Tüm günlük dosyaları
        daily_files = sorted(memory_dir.glob("20*-*.md"))
    else:
        # Sadece son sync'ten sonraki dosyalar
        last_sync = get_last_sync_time(state_dir)
        if last_sync is None:
            # İlk sync - son 7 gün
            cutoff = datetime.now() - timedelta(days=7)
        else:
            cutoff = last_sync
        
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        daily_files = sorted([
            f for f in memory_dir.glob("20*-*.md")
            if f.stem >= cutoff_str
        ])
    
    if not daily_files:
        print("📭 Güncelleme yok - tüm dosyalar güncel")
        return {
            'status': 'up_to_date',
            'files': 0,
        }
    
    print(f"🔄 {len(daily_files)} dosya sync ediliyor...")
    
    total_entities_before = len(model.graph.nodes)
    total_relations_before = len(model.graph.edges)
    
    processed = 0
    errors = []
    
    for daily_file in daily_files:
        try:
            model.update_from_daily(str(daily_file))
            processed += 1
            print(f"  ✅ {daily_file.name}")
        except Exception as e:
            errors.append(f"{daily_file.name}: {e}")
            print(f"  ❌ {daily_file.name}: {e}")
    
    total_entities_after = len(model.graph.nodes)
    total_relations_after = len(model.graph.edges)
    
    entities_added = total_entities_after - total_entities_before
    relations_added = total_relations_after - total_relations_before
    
    # Sync kaydını güncelle
    update_last_sync(state_dir, processed, entities_added, relations_added)
    
    result = {
        'status': 'completed',
        'files': processed,
        'errors': len(errors),
        'entities_added': entities_added,
        'relations_added': relations_added,
        'total_entities': total_entities_after,
        'total_relations': total_relations_after,
    }
    
    print(f"\n📊 Sync tamamlandı:")
    print(f"   Dosyalar: {processed}/{len(daily_files)}")
    print(f"   Entity'ler: {total_entities_before} → {total_entities_after} (+{entities_added})")
    print(f"   İlişkiler: {total_relations_before} → {total_relations_after} (+{relations_added})")
    
    if errors:
        print(f"   Hatalar: {errors}")
    
    # Sonucu state'e kaydet
    result_file = state_dir / "world_model_sync_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WorldModel Sync')
    parser.add_argument('--workspace', default=None)
    parser.add_argument('--file', default=None, help='Belirli dosyayı sync et')
    parser.add_argument('--full', action='store_true', help='Tam yeniden yükle')
    
    args = parser.parse_args()
    sync(args.workspace, args.file, args.full)

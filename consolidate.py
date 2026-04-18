#!/usr/bin/env python3
"""
Haci Memory Consolidation — Ana çalıştırıcı
1. MemoryConsolidator: önem skoru, 8 tip, MD5 dedup, feedback entegrasyonu
2. RetentionDaily: personality yükleme, context compression
3. FAISS index rebuild
Claude Code Stop hook'una bağlı + cron 23:55
"""

import os
import sys
import time
import json
from pathlib import Path

HACI_MEMORY_PATH = Path(os.environ.get("HACI_MEMORY_PATH", Path.home() / ".haci-memory"))
sys.path.insert(0, str(HACI_MEMORY_PATH))


def run_consolidation():
    from memory_consolidator import MemoryConsolidator
    print("\n── Memory Consolidation ──────────────────────────────")
    consolidator = MemoryConsolidator()
    consolidator.run_scheduled_consolidation()


def run_retention_daily():
    print("\n── Retention Daily Learning ──────────────────────────")
    try:
        import torch
        from retention_daily import main as retention_main
        retention_main()
    except ImportError as e:
        print(f"⚠️ Retention atlandı (torch eksik?): {e}")
    except Exception as e:
        print(f"⚠️ Retention hatası: {e}")


def rebuild_faiss():
    print("\n── FAISS Index Rebuild ───────────────────────────────")
    try:
        import faiss
        import pickle
        import numpy as np
        from sentence_transformers import SentenceTransformer

        mem_md = HACI_MEMORY_PATH / "MEMORY.md"
        if not mem_md.exists():
            print("MEMORY.md bulunamadı, atlanıyor")
            return 0

        content = mem_md.read_text(encoding='utf-8')
        sections = [s.strip() for s in content.split("\n---\n") if len(s.strip()) > 30]
        if not sections:
            print("Bölüm bulunamadı")
            return 0

        memories = [
            {"id": f"MEMORY.md#{i}", "title": f"Memory #{i+1}", "text": s[:800], "type": "memory"}
            for i, s in enumerate(sections)
        ]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([m["text"] for m in memories], convert_to_numpy=True).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss_path = HACI_MEMORY_PATH / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        (HACI_MEMORY_PATH / "faiss.index.map.pkl").write_bytes(
            pickle.dumps({m["id"]: m for m in memories})
        )
        print(f"✅ FAISS: {index.ntotal} vektör")
        return index.ntotal
    except Exception as e:
        print(f"❌ FAISS hatası: {e}")
        return 0


def save_state(faiss_count: int):
    state_file = HACI_MEMORY_PATH / "consolidation_state.json"
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            pass
    state["last_run"] = time.time()
    state["total_runs"] = state.get("total_runs", 0) + 1
    state["last_index_size"] = faiss_count
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def main():
    print("=" * 55)
    print("🧿 HACI FULL CONSOLIDATION")
    print("=" * 55)

    run_consolidation()
    run_retention_daily()
    count = rebuild_faiss()
    save_state(count)

    print("\n✅ Tüm adımlar tamamlandı")


if __name__ == "__main__":
    main()

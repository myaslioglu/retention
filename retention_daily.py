#!/usr/bin/env python3
"""
Hacı Retention Daily Learning
Personality'yi SOUL.md/USER.md/MEMORY.md'den yükler,
yeni bellekler için retention layer ile context compression yapar.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from haci_cognitive.cognitive_net import MultiScaleRetention

from sentence_transformers import SentenceTransformer

HACI_MEMORY_PATH = Path(os.environ.get("HACI_MEMORY_PATH", Path.home() / ".haci-memory"))


class HaciRetentionSystem:
    """Hacı Retention System — personality + context compression."""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads

        self.retention = MultiScaleRetention(
            d_model=d_model, n_heads=n_heads, max_seq_len=512,
            decay_min=0.3, decay_max=0.99, dropout=0.1,
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.projection = nn.Linear(embed_dim, d_model)
        self.personality_state = None
        self.conversation_memory: List[str] = []

        print(f"🎯 HaciRetentionSystem: d_model={d_model}, heads={n_heads}, embed={embed_dim}")

    def load_personality(self, workspace_path: Path = HACI_MEMORY_PATH) -> bool:
        personality_text = ""
        for fname in ["SOUL.md", "USER.md"]:
            fp = workspace_path / fname
            if not fp.exists():
                # SOUL.md might be in Applications CLAUDE.md
                alt = Path.home() / "Applications" / "CLAUDE.md"
                if alt.exists():
                    personality_text += alt.read_text(encoding='utf-8') + "\n"
                    continue
            if fp.exists():
                personality_text += fp.read_text(encoding='utf-8') + "\n"

        mem_md = workspace_path / "MEMORY.md"
        if mem_md.exists():
            content = mem_md.read_text(encoding='utf-8')
            personality_text += content[-1000:]

        if personality_text:
            embed = self.embedder.encode(personality_text)
            tensor = torch.tensor(embed).float()
            with torch.no_grad():
                self.personality_state = self.projection(tensor.unsqueeze(0))
            print(f"🧠 Personality yüklendi ({len(personality_text)} karakter)")
            return True
        print("⚠️ Personality dosyası bulunamadı")
        return False

    def embed_conversation(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        texts = [("User: " if m["role"] == "user" else "Hacı: ") + m["content"] for m in messages]
        embeddings = np.array([self.embedder.encode(t) for t in texts])
        tensor = torch.tensor(embeddings).float()
        with torch.no_grad():
            return self.projection(tensor)

    def compress_context(self, conversation_history: List[Dict[str, str]]) -> torch.Tensor:
        embeddings = self.embed_conversation(conversation_history).unsqueeze(0)
        with torch.no_grad():
            compressed = self.retention(embeddings)
        context_summary = compressed[0, -1, :]
        if self.personality_state is not None:
            alpha = 0.3
            context_summary = (1 - alpha) * context_summary + alpha * self.personality_state[0]
        return context_summary

    def daily_learning_step(self, new_memories: List[str]) -> bool:
        if not new_memories:
            return False
        print(f"📚 Günlük öğrenme: {len(new_memories)} yeni bellek")
        batch_embeddings = []
        for memory in new_memories:
            conv = [
                {"role": "user", "content": f"Bunu hatırla: {memory}"},
                {"role": "assistant", "content": "Anladım, hafızama kaydettim."},
            ]
            emb = self.embed_conversation(conv)
            batch_embeddings.append(emb.unsqueeze(0))

        if batch_embeddings:
            batch = torch.cat(batch_embeddings, dim=0)
            with torch.no_grad():
                self.retention(batch)
            self.conversation_memory.extend(new_memories[:10])
            print(f"  ✓ {len(batch_embeddings)} bellek çifti işlendi")
            return True
        return False

    def save_state(self, path: Path):
        state = {
            'conversation_memory': self.conversation_memory,
            'personality_state': self.personality_state.numpy().tolist() if self.personality_state is not None else None,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"💾 Retention state kaydedildi: {path}")

    def load_state(self, path: Path) -> bool:
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            self.conversation_memory = state.get('conversation_memory', [])
            pd = state.get('personality_state')
            if pd:
                self.personality_state = torch.tensor(pd).float()
            print(f"📂 Retention state yüklendi: {len(self.conversation_memory)} bellek")
            return True
        except Exception as e:
            print(f"⚠️ State yüklenemedi: {e}")
            return False

    def get_new_memories_since(self, since_timestamp: float) -> List[str]:
        mem_dir = HACI_MEMORY_PATH / "memory"
        new_memories = []
        if not mem_dir.exists():
            return []
        for file in mem_dir.glob("*.md"):
            if file.stat().st_mtime > since_timestamp:
                try:
                    lines = [l.strip() for l in file.read_text(encoding='utf-8').split('\n') if l.strip()]
                    summary = ' '.join(lines[:3])
                    if summary:
                        new_memories.append(f"{file.stem}: {summary}")
                except Exception:
                    pass
        return new_memories


def main():
    state_path = HACI_MEMORY_PATH / "retention_state.json"

    print("=" * 60)
    print("🧠 HACI RETENTION DAILY LEARNING")
    print(f"📅 {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    hrs = HaciRetentionSystem(d_model=128, n_heads=4)
    personality_loaded = hrs.load_personality()

    last_run_time = 0.0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
            last_run_time = datetime.datetime.fromisoformat(
                state.get('timestamp', '2000-01-01')
            ).timestamp()
            hrs.load_state(state_path)
        except Exception as e:
            print(f"⚠️ Önceki state yüklenemedi: {e}")

    new_memories = hrs.get_new_memories_since(last_run_time)

    if new_memories:
        print(f"📝 Son çalışmadan bu yana {len(new_memories)} yeni bellek:")
        for m in new_memories[:3]:
            print(f"  • {m[:80]}...")
        hrs.daily_learning_step(new_memories)
    else:
        print("📭 Son çalışmadan bu yana yeni bellek yok")

    hrs.save_state(state_path)
    print(f"\n✅ Tamamlandı — personality={personality_loaded}, yeni={len(new_memories)}, toplam={len(hrs.conversation_memory)}")
    return True


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

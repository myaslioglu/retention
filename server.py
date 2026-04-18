#!/usr/bin/env python3
"""
Haci Memory MCP Server
FAISS + sentence-transformers semantic memory search for Claude Code
Ported from OpenClaw haci-memory-plugin
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# ── Config ──────────────────────────────────────────────────────────────────

MEMORY_PATH = Path(os.environ.get("HACI_MEMORY_PATH", Path.home() / ".haci-memory"))
FAISS_INDEX_PATH = MEMORY_PATH / "faiss.index"
EMBEDDING_MODEL = os.environ.get("HACI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Memory Worker ────────────────────────────────────────────────────────────

class HaciMemory:
    def __init__(self):
        self.embedder = None
        self.faiss_index = None
        self.memories: list[dict] = []
        self.memory_map: dict[str, dict] = {}
        self._faiss = None

    def _load_deps(self):
        if self._faiss is None:
            import faiss
            from sentence_transformers import SentenceTransformer
            self._faiss = faiss
            if self.embedder is None:
                self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def load_memories(self) -> list[dict]:
        memories = []

        # MEMORY.md
        memory_md = MEMORY_PATH / "MEMORY.md"
        if memory_md.exists():
            text = memory_md.read_text(encoding="utf-8")
            # Split by sections for better granularity
            sections = text.split("\n---\n")
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) > 30:
                    memories.append({
                        "id": f"MEMORY.md#{i}",
                        "title": f"Memory #{i+1}",
                        "text": section[:800],
                        "type": "memory",
                        "source": str(memory_md),
                    })

        # memory/*.md
        mem_dir = MEMORY_PATH / "memory"
        if mem_dir.exists():
            for f in sorted(mem_dir.glob("*.md")):
                try:
                    text = f.read_text(encoding="utf-8")
                    memories.append({
                        "id": f"memory/{f.name}",
                        "title": f"Daily: {f.stem}",
                        "text": text[:800],
                        "type": "daily",
                        "source": str(f),
                    })
                except Exception:
                    pass

        self.memories = memories
        self.memory_map = {m["id"]: m for m in memories}
        return memories

    def build_index(self) -> int:
        self._load_deps()
        self.load_memories()
        if not self.memories:
            return 0

        texts = [m["text"] for m in self.memories]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).astype(np.float32)

        dim = embeddings.shape[1]
        self.faiss_index = self._faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings)

        # Persist
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
        map_path = FAISS_INDEX_PATH.with_suffix(".map.pkl")
        map_path.write_bytes(pickle.dumps(self.memory_map))

        return self.faiss_index.ntotal

    def load_index(self) -> bool:
        self._load_deps()
        if not FAISS_INDEX_PATH.exists():
            return False
        try:
            self.faiss_index = self._faiss.read_index(str(FAISS_INDEX_PATH))
            map_path = FAISS_INDEX_PATH.with_suffix(".map.pkl")
            if map_path.exists():
                self.memory_map = pickle.loads(map_path.read_bytes())
                self.memories = list(self.memory_map.values())
            return True
        except Exception:
            return False

    def search(self, query: str, limit: int = 5) -> list[dict]:
        self._load_deps()
        if self.faiss_index is None:
            if not self.load_index():
                self.build_index()
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.faiss_index.search(q_emb, min(limit, self.faiss_index.ntotal))

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.memories):
                mem = self.memories[idx]
                results.append({
                    "id": mem["id"],
                    "title": mem["title"],
                    "text": mem["text"][:300],
                    "type": mem["type"],
                    "similarity": float(1 / (1 + dist)),
                    "rank": rank + 1,
                })
        return results

    def add_memory(self, text: str, title: str = "New Memory", mem_type: str = "note") -> dict:
        self._load_deps()
        mem_id = f"manual_{int(time.time())}"
        mem = {"id": mem_id, "title": title, "text": text, "type": mem_type, "source": "manual"}
        self.memories.append(mem)
        self.memory_map[mem_id] = mem

        emb = self.embedder.encode([text], convert_to_numpy=True).astype(np.float32)
        if self.faiss_index is None:
            self.faiss_index = self._faiss.IndexFlatL2(emb.shape[1])
        self.faiss_index.add(emb)

        # Also append to MEMORY.md
        memory_md = MEMORY_PATH / "MEMORY.md"
        with open(memory_md, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M")
            f.write(f"\n\n### {title} ({ts})\n{text}\n---\n")

        self.build_index()  # rebuild to keep index consistent
        return {"id": mem_id, "added": True}


# ── MCP Server ───────────────────────────────────────────────────────────────

memory = HaciMemory()
server = Server("haci-memory")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="haci_memory_recall",
            description=(
                "Haci'nin hafızasında semantic arama yap. "
                "Geçmiş kararlar, tercihler, proje detayları, dersler için kullan. "
                "Her oturum başında konuyla ilgili sorgula."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"},
                    "limit": {"type": "integer", "default": 5, "description": "Maks sonuç sayısı"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="haci_memory_store",
            description=(
                "Önemli bilgiyi Haci'nin hafızasına kaydet. "
                "Kararlar, tercihler, teknik bilgiler, öğrenilen dersler için kullan."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Kaydedilecek bilgi"},
                    "title": {"type": "string", "description": "Başlık"},
                    "type": {
                        "type": "string",
                        "enum": ["decision", "preference", "technical", "lesson", "project", "note"],
                        "default": "note",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="haci_memory_rebuild",
            description="FAISS index'i yeniden oluştur. MEMORY.md güncellenince çalıştır.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "haci_memory_recall":
        query = arguments["query"]
        limit = arguments.get("limit", 5)
        results = memory.search(query, limit)

        if not results:
            text = f"'{query}' için ilgili hafıza bulunamadı."
        else:
            lines = [f"🧿 {len(results)} hafıza bulundu — '{query}':\n"]
            for r in results:
                pct = f"{r['similarity']*100:.0f}%"
                lines.append(f"{r['rank']}. [{r['type'].upper()}] {r['title']} ({pct})")
                lines.append(f"   {r['text'][:150]}...")
                lines.append("")
            text = "\n".join(lines)

        return [types.TextContent(type="text", text=text)]

    elif name == "haci_memory_store":
        text_val = arguments["text"]
        title = arguments.get("title", "Not")
        mem_type = arguments.get("type", "note")
        result = memory.add_memory(text_val, title, mem_type)
        return [types.TextContent(type="text", text=f"✅ Hafızaya kaydedildi: '{title}' (ID: {result['id']})")]

    elif name == "haci_memory_rebuild":
        count = memory.build_index()
        return [types.TextContent(type="text", text=f"✅ FAISS index yeniden oluşturuldu: {count} vektör")]

    else:
        return [types.TextContent(type="text", text=f"Bilinmeyen tool: {name}")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

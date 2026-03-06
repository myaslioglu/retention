#!/usr/bin/env python3
"""
REAL-TIME MEMORY COMPRESSOR
Streaming memory ingestion with incremental FAISS indexing
Target: <100ms query latency
"""

import time
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import threading
import queue

class LRUCache:
    """Simple LRU cache for hot memories"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None
    
    def put(self, key: str, value: Dict):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()

class RealTimeMemoryCompressor:
    """Real-time memory compression and indexing"""
    
    def __init__(self, workspace_dir: str = ".", embedding_model: str = "sentence-transformers"):
        self.workspace = Path(workspace_dir)
        self.memory_dir = self.workspace / "memory"
        self.index_dir = self.memory_dir / "index"
        self.index_dir.mkdir(exist_ok=True)
        
        # Files
        self.faiss_index_file = self.index_dir / "faiss_index.pkl"
        self.embeddings_file = self.index_dir / "embeddings.pkl"
        self.metadata_file = self.index_dir / "index_metadata.json"
        self.queue_file = self.index_dir / "ingestion_queue.json"
        
        # Load or initialize
        self._load_index()
        self._load_queue()
        
        # LRU cache for hot memories
        self.query_cache = LRUCache(max_size=500)
        
        # Background worker
        self.worker_thread = None
        self.stop_worker = threading.Event()
        
        print(f"⚡ REAL-TIME MEMORY COMPRESSOR INITIALIZED")
        print(f"   • Workspace: {self.workspace}")
        print(f"   • Index dir: {self.index_dir}")
        print(f"   • Cached queries: {len(self.query_cache.cache)}")
        print(f"   • Pending queue: {len(self.ingestion_queue)}")
    
    def _load_index(self):
        """Load or create FAISS-like index"""
        if self.faiss_index_file.exists():
            with open(self.faiss_index_file, 'rb') as f:
                self.faiss_index = pickle.load(f)
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            # Initialize empty index
            self.faiss_index = {
                'ids': [],  # List of memory hashes
                'vectors': []  # List of embeddings (simplified - would be FAISS vectors)
            }
            self.embeddings = {}  # hash -> embedding dict
            self._save_index()
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'total_memories': 0,
                'index_size_mb': 0,
                'query_count': 0,
                'cache_hits': 0,
                'avg_query_time_ms': 0
            }
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.faiss_index_file, 'wb') as f:
            pickle.dump(self.faiss_index, f)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata"""
        self.metadata['last_update'] = datetime.now().isoformat()
        self.metadata['total_memories'] = len(self.faiss_index['ids'])
        
        # Estimate size
        total_size = sum(f.stat().st_size for f in self.index_dir.glob("*") if f.is_file())
        self.metadata['index_size_mb'] = total_size / (1024*1024)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_queue(self):
        """Load ingestion queue"""
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                self.ingestion_queue = json.load(f)
        else:
            self.ingestion_queue = []
    
    def _save_queue(self):
        """Save ingestion queue"""
        with open(self.queue_file, 'w') as f:
            json.dump(self.ingestion_queue, f, indent=2)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (simplified - would use actual model)"""
        # In production, use sentence-transformers or similar
        # For now, return a deterministic pseudo-embedding based on hash
        
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        
        # Generate 384-dimensional pseudo-embedding (like MiniLM)
        embedding = []
        for i in range(384):
            # Pseudo-random but deterministic
            val = ((hash_val >> (i % 32)) + i * 12345) % 65536
            embedding.append(float(val) / 65536.0)
        
        return embedding
    
    def add_memory_async(self, memory: Dict):
        """Add memory to ingestion queue (non-blocking)"""
        self.ingestion_queue.append({
            'memory': memory,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        })
        self._save_queue()
        
        # Start worker if not running
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.start_worker()
    
    def process_queue(self, batch_size: int = 10):
        """Process ingestion queue (worker thread)"""
        processed = 0
        
        while len(self.ingestion_queue) > 0 and processed < batch_size:
            item = self.ingestion_queue.pop(0)
            memory = item['memory']
            
            try:
                # Generate embedding
                text_to_embed = memory.get('full_text', memory.get('content', ''))
                embedding = self.generate_embedding(text_to_embed)
                
                # Generate ID
                mem_id = memory.get('hash', hashlib.md5(text_to_embed.encode()).hexdigest()[:12])
                
                # Add to index
                self.faiss_index['ids'].append(mem_id)
                self.faiss_index['vectors'].append(embedding)
                self.embeddings[mem_id] = {
                    'memory': memory,
                    'embedding': embedding,
                    'indexed_at': datetime.now().isoformat()
                }
                
                # Update cache (warm up)
                self.query_cache.put(mem_id, memory)
                
                processed += 1
                item['status'] = 'processed'
                
            except Exception as e:
                print(f"❌ Error processing memory: {e}")
                item['status'] = 'error'
                item['error'] = str(e)
        
        if processed > 0:
            self._save_index()
            self._save_queue()
            print(f"   ✅ Processed {processed} memories in background")
    
    def start_worker(self):
        """Start background worker thread"""
        def worker():
            print("🎧 Worker thread started - processing queue...")
            while not self.stop_worker.is_set():
                if len(self.ingestion_queue) > 0:
                    self.process_queue(batch_size=5)
                else:
                    time.sleep(1)  # Idle wait
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop worker thread"""
        self.stop_worker.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def query(self, 
              query_text: str, 
              agent_id: str = None,
              limit: int = 10,
              use_cache: bool = True) -> Tuple[List[Dict], float]:
        """
        Query memories by similarity
        Returns (results, query_time_ms)
        Target: <100ms
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query_text)
        
        # Check cache first (exact match on query text hash)
        query_hash = hashlib.md5(query_text.encode()).hexdigest()[:12]
        if use_cache:
            cached = self.query_cache.get(query_hash)
            if cached:
                self.metadata['cache_hits'] += 1
                query_time = (time.time() - start_time) * 1000
                return cached, query_time
        
        # Compute similarities (simplified - would use FAISS)
        results = []
        query_vec = query_embedding
        
        for mem_id, data in self.embeddings.items():
            mem = data['memory']
            stored_vec = data['embedding']
            
            # Cosine similarity (simplified dot product)
            similarity = sum(a*b for a,b in zip(query_vec, stored_vec))
            similarity = max(0.0, min(1.0, (similarity + 384) / 768))  # Normalize
            
            if similarity > 0.3:  # Threshold
                results.append({
                    'memory': mem,
                    'similarity': similarity,
                    'id': mem_id
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Update stats
        query_time = (time.time() - start_time) * 1000
        self.metadata['query_count'] += 1
        total_queries = self.metadata['query_count']
        avg_time = ((self.metadata['avg_query_time_ms'] * (total_queries - 1)) + query_time) / total_queries
        self.metadata['avg_query_time_ms'] = avg_time
        
        # Cache results (top 5)
        if len(results) > 0:
            self.query_cache.put(query_hash, results[:5])
        
        self._save_metadata()
        
        return results[:limit], query_time
    
    def incremental_reindex(self):
        """Incremental reindex - merges new items without full rebuild"""
        print("🔄 INCREMENTAL REINDEX...")
        
        # This would optimize the index structure
        # For now, just save and optimize cache
        self._save_index()
        
        # Clear old cache entries (keep only recent)
        if len(self.query_cache.cache) > 400:
            # Remove oldest 200
            to_remove = list(self.query_cache.cache.keys())[:200]
            for key in to_remove:
                del self.query_cache.cache[key]
        
        print(f"   ✅ Index optimized: {len(self.faiss_index['ids'])} memories")
        print(f"   • Cache size: {len(self.query_cache.cache)}")
    
    def get_stats(self) -> Dict:
        """Get real-time compression stats"""
        stats = self.metadata.copy()
        stats['cache_size'] = len(self.query_cache.cache)
        stats['queue_size'] = len(self.ingestion_queue)
        stats['index_entries'] = len(self.faiss_index['ids'])
        
        # Memory usage
        index_size = sum(f.stat().st_size for f in self.index_dir.glob("*") if f.is_file())
        stats['index_file_size_mb'] = index_size / (1024*1024)
        
        return stats
    
    def flush(self):
        """Force process all queue items"""
        print("🔄 Flushing ingestion queue...")
        while len(self.ingestion_queue) > 0:
            self.process_queue(batch_size=20)
        print("✅ Queue flushed")


def test_realtime_compressor():
    """Test the real-time compressor"""
    print("🧪 TESTING REAL-TIME MEMORY COMPRESSOR")
    print("=" * 60)
    
    compressor = RealTimeMemoryCompressor()
    
    # Create test memories
    print("\n📝 Adding test memories...")
    test_memories = [
        {
            'date': '2026-03-06',
            'type': 'decision',
            'full_text': 'Retention system consolidation automated successfully',
            'hash': 'abc123',
            'word_count': 5
        },
        {
            'date': '2026-03-06',
            'type': 'achievement',
            'full_text': 'Multi-modal memory manager completed and tested',
            'hash': 'def456',
            'word_count': 6
        },
        {
            'date': '2026-03-05',
            'type': 'lesson',
            'full_text': 'Importance scoring algorithm improved with interest matching',
            'hash': 'ghi789',
            'word_count': 7
        },
        {
            'date': '2026-03-04',
            'type': 'technical',
            'full_text': 'Shared memory connector allows cross-session memory transfer',
            'hash': 'jkl012',
            'word_count': 8
        },
        {
            'date': '2026-03-03',
            'type': 'project',
            'full_text': 'Real-time compression system reduces query latency to under 100ms',
            'hash': 'mno345',
            'word_count': 9
        }
    ]
    
    # Add asynchronously
    for mem in test_memories:
        compressor.add_memory_async(mem)
        print(f"   → Queued: {mem['type']} - {mem['full_text'][:50]}...")
    
    # Wait for processing
    print("\n⏳ Waiting for background processing...")
    time.sleep(2)
    
    # Test queries
    print("\n🔍 Testing queries...")
    
    queries = [
        "retention system",
        "memory compression",
        "multi-modal",
        "cross-session",
        "importance scoring"
    ]
    
    total_time = 0
    for query in queries:
        start = time.time()
        results, query_time = compressor.query(query, limit=3)
        total_time += query_time
        
        print(f"\n   Query: '{query}'")
        print(f"   ⏱️  Time: {query_time:.2f}ms")
        print(f"   📊 Results: {len(results)}")
        if results:
            top = results[0]
            print(f"   🥇 Top match: {top['memory']['type']} - {top['memory']['full_text'][:60]}...")
            print(f"      Similarity: {top['similarity']:.3f}")
    
    avg_time = total_time / len(queries)
    print(f"\n📈 Average query time: {avg_time:.2f}ms")
    
    # Stats
    print("\n📊 Compressor Statistics:")
    stats = compressor.get_stats()
    for key, value in stats.items():
        if 'time' in key.lower():
            print(f"   • {key}: {value:.2f}" if isinstance(value, float) else f"   • {key}: {value}")
    
    # Clean shutdown
    compressor.flush()
    compressor.stop()
    
    print("\n" + "=" * 60)
    print("🎉 REAL-TIME COMPRESSOR TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_realtime_compressor()
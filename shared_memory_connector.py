#!/usr/bin/env python3
"""
SHARED MEMORY CONNECTOR
Cross-session memory transfer system
Allows any agent to read/write to global memory store
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

class SharedMemoryConnector:
    """Connector to shared global memory store"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace = Path(workspace_dir)
        self.shared_memory_dir = self.workspace / "shared_memory"
        self.index_file = self.shared_memory_dir / "global_index.pkl"
        self.metadata_file = self.shared_memory_dir / "global_metadata.json"
        
        # Ensure directory exists
        self.shared_memory_dir.mkdir(exist_ok=True)
        
        # Initialize structures
        self._init_index()
        self._load_metadata()
        
        print(f"🔗 SHARED MEMORY CONNECTOR INITIALIZED")
        print(f"   • Workspace: {self.workspace}")
        print(f"   • Shared dir: {self.shared_memory_dir}")
        print(f"   • Total shared memories: {len(self.metadata.get('entries', []))}")
    
    def _init_index(self):
        """Initialize or load FAISS-like index"""
        if self.index_file.exists():
            with open(self.index_file, 'rb') as f:
                self.index = pickle.load(f)
        else:
            # Simple dict-based index (would be FAISS in production)
            self.index = {
                'embeddings': {},  # hash -> embedding
                'metadata': {}    # hash -> metadata
            }
            self._save_index()
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)
    
    def _load_metadata(self):
        """Load global metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'entries': [],  # List of memory entries
                'agents': {},   # Agent statistics
                'global_stats': {
                    'total_reads': 0,
                    'total_writes': 0,
                    'total_queries': 0
                }
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def write_memory(self, 
                    agent_id: str, 
                    memory_type: str,
                    content: str,
                    metadata: Optional[Dict] = None) -> Dict:
        """Write a memory entry to shared store"""
        
        # Generate unique hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Create memory entry
        entry = {
            'id': content_hash,
            'agent_id': agent_id,
            'type': memory_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'access_count': 0,
            'last_accessed': None
        }
        
        # Check if already exists
        existing = next((e for e in self.metadata['entries'] if e['id'] == content_hash), None)
        if existing:
            # Update existing (increment version, update timestamp)
            existing['content'] = content
            existing['metadata'] = metadata or {}
            existing['timestamp'] = datetime.now().isoformat()
            entry = existing
        else:
            # Add new
            self.metadata['entries'].append(entry)
        
        # Update agent stats
        if agent_id not in self.metadata['agents']:
            self.metadata['agents'][agent_id] = {
                'writes': 0,
                'reads': 0,
                'queries': 0
            }
        self.metadata['agents'][agent_id]['writes'] += 1
        self.metadata['global_stats']['total_writes'] += 1
        
        # Save
        self._save_metadata()
        
        print(f"   ✅ Memory written [{agent_id}]: {memory_type} - {content[:50]}...")
        return entry
    
    def read_memory(self, memory_id: str, agent_id: str = None) -> Optional[Dict]:
        """Read a specific memory entry"""
        entry = next((e for e in self.metadata['entries'] if e['id'] == memory_id), None)
        
        if entry:
            # Update access stats
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.now().isoformat()
            
            if agent_id:
                if agent_id not in self.metadata['agents']:
                    self.metadata['agents'][agent_id] = {'writes': 0, 'reads': 0, 'queries': 0}
                self.metadata['agents'][agent_id]['reads'] += 1
                self.metadata['global_stats']['total_reads'] += 1
            
            self._save_metadata()
            return entry
        
        return None
    
    def query_memories(self, 
                      agent_id: str,
                      memory_type: Optional[str] = None,
                      keywords: Optional[List[str]] = None,
                      limit: int = 10,
                      min_importance: float = 0.0) -> List[Dict]:
        """Query memories by type and/or keywords"""
        
        results = []
        query_keywords = set(k.lower() for k in (keywords or []))
        
        for entry in self.metadata['entries']:
            # Filter by type
            if memory_type and entry['type'] != memory_type:
                continue
            
            # Filter by keywords
            if query_keywords:
                content_lower = entry['content'].lower()
                if not any(kw in content_lower for kw in query_keywords):
                    continue
            
            # Filter by importance (if metadata has it)
            importance = entry.get('metadata', {}).get('importance', 0.5)
            if importance < min_importance:
                continue
            
            results.append(entry)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Update stats
        if agent_id not in self.metadata['agents']:
            self.metadata['agents'][agent_id] = {'writes': 0, 'reads': 0, 'queries': 0}
        self.metadata['agents'][agent_id]['queries'] += 1
        self.metadata['global_stats']['total_queries'] += 1
        self._save_metadata()
        
        return results[:limit]
    
    def get_agent_memories(self, agent_id: str, limit: int = 50) -> List[Dict]:
        """Get all memories written by a specific agent"""
        agent_memories = [e for e in self.metadata['entries'] if e['agent_id'] == agent_id]
        agent_memories.sort(key=lambda x: x['timestamp'], reverse=True)
        return agent_memories[:limit]
    
    def delete_memory(self, memory_id: str, agent_id: str = None) -> bool:
        """Delete a memory entry (only if agent_id matches or is None)"""
        entry = next((e for e in self.metadata['entries'] if e['id'] == memory_id), None)
        
        if entry:
            if agent_id and entry['agent_id'] != agent_id:
                return False  # Not authorized
            
            self.metadata['entries'].remove(entry)
            self._save_metadata()
            return True
        
        return False
    
    def get_global_stats(self) -> Dict:
        """Get global statistics"""
        stats = self.metadata['global_stats'].copy()
        stats['total_entries'] = len(self.metadata['entries'])
        stats['active_agents'] = len(self.metadata['agents'])
        stats['entries_by_type'] = {}
        
        for entry in self.metadata['entries']:
            mem_type = entry['type']
            stats['entries_by_type'][mem_type] = stats['entries_by_type'].get(mem_type, 0) + 1
        
        return stats
    
    def export_agent_context(self, agent_id: str) -> Dict:
        """Export all relevant memories for an agent's context"""
        # Get agent's own memories
        own_memories = self.get_agent_memories(agent_id, limit=100)
        
        # Get relevant shared memories (high importance, recent)
        recent_shared = self.query_memories(
            agent_id=agent_id,
            min_importance=0.6,
            limit=20
        )
        
        return {
            'agent_id': agent_id,
            'own_memories': own_memories,
            'shared_relevant': recent_shared,
            'timestamp': datetime.now().isoformat()
        }


def test_shared_memory():
    """Test shared memory connector"""
    print("🧪 TESTING SHARED MEMORY CONNECTOR")
    print("=" * 60)
    
    connector = SharedMemoryConnector()
    
    # Test writes from different agents
    print("\n📝 Testing writes from different agents...")
    
    agent1_memories = [
        ("main", "decision", "Retention system deployed successfully"),
        ("main", "achievement", "Memory consolidation automated"),
        ("main", "lesson", "Importance scoring works better with interest matching")
    ]
    
    agent2_memories = [
        ("github", "action", "PR #42 merged: retention enhancement"),
        ("github", "project", "CI/CD pipeline optimized"),
        ("github", "technical", "FAISS index rebuilt")
    ]
    
    for agent, mem_type, content in agent1_memories + agent2_memories:
        connector.write_memory(
            agent_id=agent,
            memory_type=mem_type,
            content=content,
            metadata={'importance': 0.7, 'source': 'test'}
        )
    
    # Test queries
    print("\n🔍 Testing queries...")
    
    # Query by type
    decisions = connector.query_memories(agent_id="test", memory_type="decision")
    print(f"   • Found {len(decisions)} decision memories")
    
    # Query by keywords
    retention_memories = connector.query_memories(
        agent_id="test",
        keywords=["retention"],
        limit=5
    )
    print(f"   • Found {len(retention_memories)} memories about 'retention'")
    
    # Get agent-specific memories
    main_memories = connector.get_agent_memories("main")
    print(f"   • Main agent has {len(main_memories)} memories")
    
    # Export context
    print("\n📦 Testing context export...")
    context = connector.export_agent_context("main")
    print(f"   • Context for 'main': {len(context['own_memories'])} own, {len(context['shared_relevant'])} shared")
    
    # Global stats
    print("\n📊 Global Statistics:")
    stats = connector.get_global_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 SHARED MEMORY CONNECTOR TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_shared_memory()
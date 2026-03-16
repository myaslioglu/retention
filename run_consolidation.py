#!/usr/bin/env python3
"""
RUN MEMORY CONSOLIDATION
Cron job tarafından çalıştırılacak script
"""

import sys
import os
from pathlib import Path
import importlib.util
import importlib.machinery

# Add workspace to path
workspace_path = "/Users/muratyaslioglu/.openclaw/workspace"
sys.path.append(workspace_path)

# Import the consolidator (v2 version)
try:
    # The actual file is memory_consolidator.py.v2
    module_path = os.path.join(workspace_path, "memory_consolidator.py.v2")
    module_name = "memory_consolidator_v2"
    
    # Use SourceFileLoader to load from file with any extension
    loader = importlib.machinery.SourceFileLoader(module_name, module_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f"Could not create spec for {module_path}")
    
    consolidator_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = consolidator_module
    spec.loader.exec_module(consolidator_module)
    
    MemoryConsolidator = consolidator_module.MemoryConsolidator
except Exception as e:
    print(f"❌ Error loading consolidator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    print("🕒 Scheduled memory consolidation starting...")
    consolidator = MemoryConsolidator()
    consolidator.run_scheduled_consolidation()
    
    # WorldModel sync after consolidation
    print("\n🔄 Syncing WorldModel with new memories...")
    try:
        from haci_cognitive.world_model_sync import sync
        result = sync(workspace_path)
        print(f"✅ WorldModel sync: {result.get('status', 'done')}")
    except Exception as e:
        print(f"⚠️ WorldModel sync error: {e}")
    
    # Memory clustering
    print("\n🏷️ Clustering memories...")
    try:
        from haci_cognitive.memory_clusterer import MemoryClusterer
        clusterer = MemoryClusterer(workspace_path)
        result = clusterer.cluster_all()
        print(f"✅ Clustering: {result.get('processed', 0)} files processed")
    except Exception as e:
        print(f"⚠️ Clustering error: {e}")

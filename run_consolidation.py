#!/usr/bin/env python3
"""
PRODUCTION MEMORY CONSOLIDATION SCRIPT
Enhanced memory consolidator with all FAZ 1 improvements + Multi-modal
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_consolidator import MemoryConsolidator
from learning_topics_manager import LearningTopicsManager
from multimodal_memory_manager import MultiModalMemoryManager

def run_production_consolidation():
    """Run production consolidation with enhanced features"""
    
    print("🚀 PRODUCTION MEMORY CONSOLIDATION")
    print("=" * 60)
    
    # 1. Initialize consolidator
    consolidator = MemoryConsolidator()
    
    # 2. Initialize learning topics manager
    print("\n🧠 Loading Learning Topics Manager...")
    topics_manager = LearningTopicsManager()
    
    # 3. Apply interest decay (daily)
    topics_manager.decay_interests(daily_decay=0.02)
    
    # 4. Initialize and integrate multimodal memories
    print("\n🎭 Loading Multi-modal Memory Manager...")
    multimodal_manager = MultiModalMemoryManager()
    
    # Integrate multimodal memories into consolidation
    multimodal_integrated = multimodal_manager.integrate_with_consolidator(
        consolidator, 
        days_back=3
    )
    
    # 5. Run consolidation (3 days back, threshold 0.25)
    print("\n📦 Running Enhanced Consolidation...")
    consolidator.consolidate(days_back=3, importance_threshold=0.25)
    
    # 6. Learning topics newsletter (weekly on Sunday)
    today = datetime.now()
    if today.weekday() == 6:  # Sunday
        print("\n📧 Generating Weekly Learning Newsletter...")
        newsletter = topics_manager.generate_weekly_newsletter()
        
        # Save newsletter
        newsletter_file = Path("memory") / f"newsletter_{today.strftime('%Y-%m-%d')}.md"
        with open(newsletter_file, 'w', encoding='utf-8') as f:
            f.write(newsletter)
        
        print(f"   ✅ Newsletter saved: {newsletter_file}")
        topics_manager.mark_newsletter_sent()
    else:
        print("\n📭 Newsletter not today (only Sundays)")
    
    # 7. Memory health check
    print("\n🔍 Memory Health Check...")
    mem_file = Path("MEMORY.md")
    multimodal_meta = Path("memory/multimodal/multimodal_metadata.json")
    
    if mem_file.exists():
        with open(mem_file, 'r') as f:
            content = f.read()
        
        # Count memories
        memory_count = content.count('### ')
        print(f"   📊 Total long-term memories: {memory_count}")
        
        # Check multimodal stats
        if multimodal_meta.exists():
            with open(multimodal_meta, 'r') as f:
                mm_data = json.load(f)
            print(f"   🎭 Multi-modal memories: {mm_data['stats']['total_processed']}")
            print(f"     • Audio: {mm_data['stats']['total_audio']}")
            print(f"     • Images: {mm_data['stats']['total_images']}")
            print(f"     • Videos: {mm_data['stats']['total_videos']}")
    
    print("\n" + "=" * 60)
    print("✅ PRODUCTION CONSOLIDATION COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_production_consolidation()
#!/usr/bin/env python3
"""
MEMORY DEDUPLICATOR
MEMORY.md'deki duplicate memory entry'lerini temizler
"""

import re
from pathlib import Path
from datetime import datetime

def deduplicate_memory_file(filepath: str = "MEMORY.md"):
    """Remove duplicate memory entries from MEMORY.md"""
    
    file = Path(filepath)
    if not file.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Parse memory entries
    entries = []
    current_entry = []
    in_entry = False
    seen_hashes = set()
    unique_entries = []
    duplicates_found = 0
    
    for line in lines:
        if line.startswith('### '):
            # New entry
            if current_entry and in_entry:
                entry_text = '\n'.join(current_entry)
                entry_hash = hash(entry_text)
                
                if entry_hash not in seen_hashes:
                    seen_hashes.add(entry_hash)
                    unique_entries.append(entry_text)
                else:
                    duplicates_found += 1
                    print(f"   🗑️  Duplicate removed: {current_entry[0][:50]}...")
            
            # Start new entry
            current_entry = [line]
            in_entry = True
        elif in_entry:
            current_entry.append(line)
    
    # Add last entry
    if current_entry and in_entry:
        entry_text = '\n'.join(current_entry)
        entry_hash = hash(entry_text)
        
        if entry_hash not in seen_hashes:
            seen_hashes.add(entry_hash)
            unique_entries.append(entry_text)
        else:
            duplicates_found += 1
            print(f"   🗑️  Duplicate removed: {current_entry[0][:50]}...")
    
    # Reconstruct file structure
    header = []
    capturing_header = True
    
    for line in lines[:20]:  # First 20 lines should contain header
        if line.startswith('## ') and line != '## 📜 PREVIOUS MEMORIES':
            capturing_header = False
            break
        if capturing_header:
            header.append(line)
    
    # Make sure we have proper header
    if not any('LONG-TERM MEMORY' in line for line in header):
       header = ["# 🧠 LONG-TERM MEMORY - HACI", "", "*Last consolidated: N/A*", "*Total memories: N/A*"]
    
    # Group entries by type
    entries_by_type = {}
    for entry in unique_entries:
        # Extract type from entry (first emoji)
        first_line = entry.split('\n')[0]
        if '🤔' in first_line:
            mem_type = 'decision'
        elif '📚' in first_line:
            mem_type = 'lesson'
        elif '🏆' in first_line:
            mem_type = 'achievement'
        elif '👤' in first_line:
            mem_type = 'preference'
        elif '🚀' in first_line:
            mem_type = 'project'
        elif '🔧' in first_line:
            mem_type = 'technical'
        elif '💡' in first_line:
            mem_type = 'insight'
        elif '⏰' in first_line:
            mem_type = 'reminder'
        else:
            mem_type = 'other'
        
        if mem_type not in entries_by_type:
            entries_by_type[mem_type] = []
        entries_by_type[mem_type].append(entry)
    
    # Sort entries within each type by date (newest first)
    for mem_type in entries_by_type:
        entries_by_type[mem_type].sort(reverse=True)
    
    # Build new content
    new_content = []
    
    # Add header
    new_content.extend(header[:4])
    new_content.append("")
    new_content.append(f"*Last deduplicated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    new_content.append(f"*Total memories after dedup: {len(unique_entries)}*")
    new_content.append("")
    
    # Type emoji mapping
    type_emoji = {
        'decision': '🤔',
        'lesson': '📚',
        'achievement': '🏆',
        'preference': '👤',
        'project': '🚀',
        'technical': '🔧',
        'insight': '💡',
        'reminder': '⏰',
        'other': '📝'
    }
    
    type_desc = {
        'decision': 'Kararlar',
        'lesson': 'Öğrenilen Dersler',
        'achievement': 'Başarılar',
        'preference': 'Tercihler',
        'project': 'Projeler',
        'technical': 'Teknik Bilgiler',
        'insight': 'İçgörüler',
        'reminder': 'Hatırlatıcılar',
        'other': 'Diğer'
    }
    
    # Add each type section
    for mem_type in sorted(entries_by_type.keys()):
        emoji = type_emoji.get(mem_type, '📝')
        desc = type_desc.get(mem_type, mem_type.title())
        
        new_content.append(f"## {emoji} {desc}")
        new_content.append("")
        
        for entry in entries_by_type[mem_type]:
            new_content.append(entry)
            new_content.append("")
    
    # Add previous memories section
    new_content.append("## 📜 PREVIOUS MEMORIES")
    new_content.append("")
    new_content.append("*Archived memories (pre-deduplication)*")
    new_content.append("")
    
    # Write new file
    with open(file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_content))
    
    print(f"\n✅ DEDUPLICATION COMPLETED!")
    print(f"   • Total unique entries: {len(unique_entries)}")
    print(f"   • Duplicates removed: {duplicates_found}")
    print(f"   • File: {filepath}")
    
    return len(unique_entries), duplicates_found

if __name__ == "__main__":
    print("🧹 MEMORY DEDUPLICATOR")
    print("=" * 60)
    
    total, duplicates = deduplicate_memory_file()
    
    print("\n" + "=" * 60)
    print(f"🎉 CLEANED! {duplicates} duplicates removed. {total} unique memories remain.")
    print("=" * 60)
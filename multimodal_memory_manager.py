#!/usr/bin/env python3
"""
MULTI-MODAL MEMORY MANAGER
Ses, resim ve video memory'larını işler ve memory system'e entegre eder
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import subprocess
import os

class MultiModalMemoryManager:
    """Manager for audio, image, and video memories"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace = Path(workspace_dir)
        self.memory_dir = self.workspace / "memory"
        self.multimodal_dir = self.memory_dir / "multimodal"
        self.multimodal_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.audio_dir = self.multimodal_dir / "audio"
        self.image_dir = self.multimodal_dir / "images"
        self.video_dir = self.multimodal_dir / "videos"
        
        for d in [self.audio_dir, self.image_dir, self.video_dir]:
            d.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.multimodal_dir / "multimodal_metadata.json"
        self._load_metadata()
        
        print(f"🎭 MULTI-MODAL MEMORY MANAGER INITIALIZED")
        print(f"   • Workspace: {self.workspace}")
        print(f"   • Multimodal dir: {self.multimodal_dir}")
        print(f"   • Total entries: {len(self.metadata.get('entries', []))}")
    
    def _load_metadata(self):
        """Load multimodal metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'entries': [],
                'stats': {
                    'total_audio': 0,
                    'total_images': 0,
                    'total_videos': 0,
                    'total_processed': 0
                }
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def process_audio(self, audio_path: str, agent_id: str = "main") -> Dict:
        """Process audio file: transcribe and create memory entry"""
        print(f"   🎤 Processing audio: {audio_path}")
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"error": "Audio file not found"}
        
        # Use whisper (OpenClaw skill) to transcribe
        try:
            # Check if openai-whisper skill is available
            result = subprocess.run(
                ["openai-whisper", str(audio_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            transcript = result.stdout.strip()
            
            if not transcript:
                return {"error": "Transcription failed"}
            
            # Save transcript to file
            transcript_filename = f"audio_{audio_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            transcript_path = self.audio_dir / transcript_filename
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Move audio file to storage
            audio_storage = self.audio_dir / audio_file.name
            if not audio_storage.exists():
                audio_file.rename(audio_storage)
            
            # Create memory entry
            entry = self._create_multimodal_memory(
                agent_id=agent_id,
                modality="audio",
                content=transcript,
                source_file=str(audio_storage),
                metadata={
                    "transcript_length": len(transcript),
                    "audio_format": audio_file.suffix,
                    "processing_method": "whisper"
                }
            )
            
            print(f"   ✅ Audio processed: {len(transcript)} chars transcript")
            return entry
            
        except subprocess.TimeoutExpired:
            return {"error": "Transcription timeout"}
        except Exception as e:
            return {"error": f"Transcription error: {e}"}
    
    def process_image(self, image_path: str, agent_id: str = "main", prompt: str = None) -> Dict:
        """Process image file: extract text and create memory entry"""
        print(f"   🖼️  Processing image: {image_path}")
        
        image_file = Path(image_path)
        if not image_file.exists():
            return {"error": "Image file not found"}
        
        # Use tesseract or similar OCR (simplified for now)
        try:
            # For now, create a placeholder memory
            # In production, would use OCR or image analysis skill
            description = f"Image captured: {image_file.name}"
            
            if prompt:
                description += f"\n\nPrompt/Context: {prompt}"
            
            # Move image to storage
            image_storage = self.image_dir / image_file.name
            if not image_storage.exists():
                image_file.rename(image_storage)
            
            # Create memory entry
            entry = self._create_multimodal_memory(
                agent_id=agent_id,
                modality="image",
                content=description,
                source_file=str(image_storage),
                metadata={
                    "image_format": image_file.suffix,
                    "has_prompt": prompt is not None
                }
            )
            
            print(f"   ✅ Image processed and stored")
            return entry
            
        except Exception as e:
            return {"error": f"Image processing error: {e}"}
    
    def process_video(self, video_path: str, agent_id: str = "main", extract_frames: bool = True) -> Dict:
        """Process video file: extract frames and audio if available"""
        print(f"   🎬 Processing video: {video_path}")
        
        video_file = Path(video_path)
        if not video_file.exists():
            return {"error": "Video file not found"}
        
        try:
            # For now, create a placeholder
            # In production, would use video-frames skill to extract keyframes
            # and whisper for audio track
            
            description = f"Video captured: {video_file.name}\n"
            
            if extract_frames:
                description += "• Frames ready for extraction\n"
                description += "• Audio track can be transcribed\n"
            
            # Move video to storage
            video_storage = self.video_dir / video_file.name
            if not video_storage.exists():
                video_file.rename(video_storage)
            
            # Create memory entry
            entry = self._create_multimodal_memory(
                agent_id=agent_id,
                modality="video",
                content=description,
                source_file=str(video_storage),
                metadata={
                    "video_format": video_file.suffix,
                    "frames_extracted": extract_frames
                }
            )
            
            print(f"   ✅ Video processed and stored")
            return entry
            
        except Exception as e:
            return {"error": f"Video processing error: {e}"}
    
    def _create_multimodal_memory(self, 
                                 agent_id: str,
                                 modality: str,  # audio, image, video
                                 content: str,
                                 source_file: str,
                                 metadata: Dict) -> Dict:
        """Create a multimodal memory entry"""
        
        # Generate hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Create entry
        entry = {
            'id': content_hash,
            'agent_id': agent_id,
            'modality': modality,
            'content': content,
            'source_file': source_file,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'access_count': 0,
            'last_accessed': None
        }
        
        # Add to metadata
        self.metadata['entries'].append(entry)
        
        # Update stats
        if modality == "audio":
            self.metadata['stats']['total_audio'] += 1
        elif modality == "image":
            self.metadata['stats']['total_images'] += 1
        elif modality == "video":
            self.metadata['stats']['total_videos'] += 1
        
        self.metadata['stats']['total_processed'] += 1
        
        self._save_metadata()
        
        return entry
    
    def get_multimodal_memories(self, 
                               modality: Optional[str] = None,
                               agent_id: Optional[str] = None,
                               limit: int = 20) -> List[Dict]:
        """Get multimodal memories with optional filters"""
        entries = self.metadata['entries']
        
        if modality:
            entries = [e for e in entries if e['modality'] == modality]
        
        if agent_id:
            entries = [e for e in entries if e['agent_id'] == agent_id]
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return entries[:limit]
    
    def integrate_with_consolidator(self, consolidator, days_back: int = 3):
        """Integrate multimodal memories into main consolidation flow"""
        print("\n🎭 INTEGRATING MULTI-MODAL MEMORIES INTO CONSOLIDATION...")
        
        # Get recent multimodal memories
        recent_entries = self.get_multimodal_memories(limit=50)
        
        # Filter by date
        cutoff_date = datetime.now().timestamp() - (days_back * 86400)
        eligible = []
        
        for entry in recent_entries:
            entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
            if entry_time >= cutoff_date:
                eligible.append(entry)
        
        print(f"   • Found {len(eligible)} recent multimodal memories to consolidate")
        
        # Convert to memory format for consolidator
        for entry in eligible:
            # Create memory dict that consolidator expects
            memory = {
                'date': datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d'),
                'type': self._modality_to_memory_type(entry['modality']),
                'importance': 0.7,  # Multimodal memories are important!
                'summary': f"{entry['modality'].title()}: {entry['content'][:100]}...",
                'full_text': entry['content'],
                'hash': entry['id'],
                'word_count': len(entry['content'].split()),
                'keyword_count': 0,
                'source': 'multimodal',
                'modality': entry['modality'],
                'source_file': entry['source_file']
            }
            
            # This would be added to consolidation queue
            # For now, just log
            print(f"   ✅ Ready to consolidate: {memory['type']} from {memory['date']}")
        
        return len(eligible)
    
    def _modality_to_memory_type(self, modality: str) -> str:
        """Convert modality to memory type"""
        mapping = {
            'audio': 'technical',
            'image': 'insight',
            'video': 'project'
        }
        return mapping.get(modality, 'insight')
    
    def get_stats(self) -> Dict:
        """Get multimodal statistics"""
        stats = self.metadata['stats'].copy()
        stats['total_entries'] = len(self.metadata['entries'])
        stats['by_agent'] = {}
        
        for entry in self.metadata['entries']:
            agent = entry['agent_id']
            if agent not in stats['by_agent']:
                stats['by_agent'][agent] = {'audio': 0, 'image': 0, 'video': 0}
            stats['by_agent'][agent][entry['modality']] += 1
        
        return stats


def test_multimodal_manager():
    """Test the multimodal memory manager"""
    print("🧪 TESTING MULTI-MODAL MEMORY MANAGER")
    print("=" * 60)
    
    manager = MultiModalMemoryManager()
    
    # Test processing (simulated - would need actual files)
    print("\n📝 Testing multimodal memory creation...")
    
    # Create test entries manually
    test_entries = [
        ("audio", "Test audio transcript about machine learning"),
        ("image", "Screenshot of code implementation"),
        ("video", "Recording of deployment process")
    ]
    
    for modality, content in test_entries:
        entry = manager._create_multimodal_memory(
            agent_id="test",
            modality=modality,
            content=content,
            source_file=f"/fake/path/test.{modality}",
            metadata={"test": True}
        )
        print(f"   ✅ Created {modality} memory: {entry['id']}")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    audio_mems = manager.get_multimodal_memories(modality="audio")
    print(f"   • Audio memories: {len(audio_mems)}")
    
    all_mems = manager.get_multimodal_memories(limit=10)
    print(f"   • Total multimodal memories: {len(all_mems)}")
    
    # Test integration
    print("\n🔗 Testing consolidator integration...")
    class MockConsolidator:
        pass
    
    integrated = manager.integrate_with_consolidator(MockConsolidator(), days_back=7)
    print(f"   • Memories eligible for consolidation: {integrated}")
    
    # Stats
    print("\n📊 Multimodal Statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 MULTI-MODAL MEMORY MANAGER TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_multimodal_manager()
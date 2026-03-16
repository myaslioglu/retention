#!/usr/bin/env python3
"""
HaciCognitiveNet - Main Integration Script
Ties together:

LEVEL 1 - Core Architecture:
- CognitiveNet (512+1024 dim model)
- DreamingLoop (memory synthesis)
- CognitiveStateManager (persistent state)
- CognitiveTrainer (self-supervised training)

LEVEL 2 - Autonomous Learning:
- CuriosityEngine (knowledge gap detection)
- PredictiveCoding (surprise-based learning)
- ActiveLearningScheduler (spaced repetition)
- SelfSupervisedLearningLoop (6-step cycle)

LEVEL 3 - Meta-Learning & World Model:
- HyperparameterSelfOptimizer (self-tuning)
- MetaStrategySelector (strategy selection)
- ArchitectureEvolver (self-modification)
- GenerativeWorldModel (imagination, counterfactual, prediction)
- SelfEvolutionSystem (fitness tracking, milestones)
- SensoryInterface (extensible sensor/actuator framework)

Usage:
    python haci_cognitive/main.py status      # Show cognitive state
    python haci_cognitive/main.py dream       # Run a dream cycle
    python haci_cognitive/main.py train       # Train on workspace memories
    python haci_cognitive/main.py init        # Initialize cognitive system
    python haci_cognitive/main.py summary     # Full system summary (all levels)
    python haci_cognitive/main.py learn       # Run a learning cycle (Level 2)
    python haci_cognitive/main.py curious     # Show curiosity state (Level 2)
    python haci_cognitive/main.py predict     # Show predictive coding stats (Level 2)
    python haci_cognitive/main.py evolve      # Run meta-learning cycle (Level 3)
    python haci_cognitive/main.py imagine     # Run imagination cycle (Level 3)
    python haci_cognitive/main.py meta        # Show meta-learning state (Level 3)
    python haci_cognitive/main.py evolution   # Show evolution report (Level 3)
"""

import sys
import os
import json
import logging
import random
import argparse
from pathlib import Path
from datetime import datetime

# Add workspace to path
workspace = os.path.expanduser("~/.openclaw/workspace")
sys.path.insert(0, os.path.join(workspace, "haci_cognitive"))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def get_workspace_memories(workspace_dir: str, limit: int = 50) -> list:
    """Load memory texts from workspace."""
    workspace = Path(workspace_dir)
    memory_dir = workspace / "memory"
    
    memories = []
    
    # Load from memory/*.md files
    if memory_dir.exists():
        for f in sorted(memory_dir.glob("*.md"), reverse=True)[:limit]:
            try:
                content = f.read_text()
                if content.strip():
                    memories.append({
                        'source': f.name,
                        'content': content[:2000],
                        'timestamp': f.stat().st_mtime,
                    })
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    # Load from MEMORY.md
    memory_md = workspace / "MEMORY.md"
    if memory_md.exists():
        content = memory_md.read_text()
        memories.append({
            'source': 'MEMORY.md',
            'content': content[:5000],
            'timestamp': memory_md.stat().st_mtime,
        })
    
    # Load from learning_topics.json
    learning_file = workspace / "learning_topics.json"
    if learning_file.exists():
        with open(learning_file) as f:
            data = json.load(f)
            memories.append({
                'source': 'learning_topics.json',
                'content': json.dumps(data, ensure_ascii=False)[:2000],
                'timestamp': learning_file.stat().st_mtime,
            })
    
    return memories


def cmd_init(args):
    """Initialize cognitive system."""
    logger.info("🧠 Initializing HaciCognitiveNet...\n")
    
    from cognitive_state_manager import CognitiveStateManager
    
    workspace_dir = args.workspace or workspace
    
    # Create cognitive state
    manager = CognitiveStateManager(workspace_dir)
    manager.start_session()
    manager.save_state()
    
    # Create directories
    dirs = [
        "haci_cognitive/checkpoints",
        "haci_cognitive/dreams",
        "cognitive_state",
    ]
    
    for d in dirs:
        Path(workspace_dir, d).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ HaciCognitiveNet initialized!")
    logger.info(f"   State dir: {workspace_dir}/cognitive_state")
    logger.info(f"   Dream dir: {workspace_dir}/haci_cognitive/dreams")
    logger.info(f"   Checkpoint dir: {workspace_dir}/haci_cognitive/checkpoints")
    
    # Show initial state
    print(f"\n{manager.get_summary()}")


def cmd_status(args):
    """Show cognitive state."""
    from cognitive_state_manager import CognitiveStateManager
    
    workspace_dir = args.workspace or workspace
    manager = CognitiveStateManager(workspace_dir)
    
    print(manager.get_summary())
    
    # Show tensor shape
    tensor = manager.get_state_tensor()
    print(f"\n📐 State tensor shape: {tensor.shape}")


def cmd_dream(args):
    """Run a dream cycle."""
    logger.info("🌙 Starting dream cycle...\n")
    
    from dreaming_loop import AutonomousDreamRunner
    
    workspace_dir = args.workspace or workspace
    runner = AutonomousDreamRunner(workspace_dir)
    
    max_cycles = args.cycles or 1
    report = runner.run(max_cycles=max_cycles)
    
    logger.info(f"\n🌙 Dream complete!")
    logger.info(f"   Memories processed: {report.get('memories_processed', 0)}")
    
    for insight in report.get('insights', []):
        logger.info(f"   💡 {insight['message']}")
    
    # Update cognitive state
    from cognitive_state_manager import CognitiveStateManager
    manager = CognitiveStateManager(workspace_dir)
    manager.state['metacognition']['total_dream_cycles'] += max_cycles
    manager.state['timestamps']['last_dream'] = datetime.now().isoformat()
    manager.save_state()


def cmd_train(args):
    """Train on workspace memories."""
    logger.info("🏋️ Starting cognitive training...\n")
    
    try:
        from cognitive_net import HaciCognitiveNet
        from cognitive_trainer import CognitiveTrainer
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure PyTorch is installed: pip install torch")
        return
    
    workspace_dir = args.workspace or workspace
    checkpoint_dir = os.path.join(workspace_dir, "haci_cognitive", "checkpoints")
    
    # Load memories
    memories = get_workspace_memories(workspace_dir)
    memory_texts = [m['content'] for m in memories]
    
    if not memory_texts:
        logger.warning("No memories found for training!")
        return
    
    logger.info(f"📖 Loaded {len(memory_texts)} memories")
    
    # Create model
    model = HaciCognitiveNet()
    
    # Create trainer
    trainer = CognitiveTrainer(
        model=model,
        lr=1e-4,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    n_epochs = args.epochs or 20
    result = trainer.train(
        memory_texts=memory_texts,
        n_epochs=n_epochs,
        batch_size=4,
        verbose=True,
    )
    
    logger.info(f"\n📊 Training Result:")
    for k, v in result.items():
        logger.info(f"   {k}: {v}")
    
    # Update cognitive state
    from cognitive_state_manager import CognitiveStateManager
    manager = CognitiveStateManager(workspace_dir)
    manager.state['metacognition']['total_learning_cycles'] += 1
    manager.state['metacognition']['learning_efficiency'] = 1.0 - result['best_loss']
    manager.state['timestamps']['last_training'] = datetime.now().isoformat()
    manager.save_state()


def cmd_summary(args):
    """Full system summary."""
    from cognitive_state_manager import CognitiveStateManager
    
    workspace_dir = args.workspace or workspace
    
    logger.info("=" * 60)
    logger.info("🧠 HACI COGNITIVE SYSTEM - FULL SUMMARY")
    logger.info("=" * 60)
    
    # 1. Cognitive state
    manager = CognitiveStateManager(workspace_dir)
    print(f"\n{manager.get_summary()}")
    
    # 2. Dream state
    dream_state_file = Path(workspace_dir) / "dream_state.json"
    if dream_state_file.exists():
        with open(dream_state_file) as f:
            dream_state = json.load(f)
        print(f"\n🌙 Dream State:")
        print(f"   Total cycles: {dream_state.get('total_cycles', 0)}")
        print(f"   Total insights: {dream_state.get('total_insights', 0)}")
        last = dream_state.get('last_dream')
        if last:
            print(f"   Last dream: {datetime.fromtimestamp(last).isoformat()}")
    
    # 3. Checkpoints
    checkpoint_dir = Path(workspace_dir) / "haci_cognitive" / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        print(f"\n💾 Checkpoints: {len(checkpoints)}")
        for cp in checkpoints[-3:]:
            print(f"   {cp.name}")
    
    # 4. Memory stats
    memory_dir = Path(workspace_dir) / "memory"
    if memory_dir.exists():
        memory_files = list(memory_dir.glob("*.md"))
        print(f"\n📖 Memory Files: {len(memory_files)}")
    
    # 5. Level 2 status
    print(f"\n🔄 Level 2: Self-Supervised Learning")
    learning_state_file = Path(workspace_dir) / "cognitive_state" / "learning_loop_state.json"
    if learning_state_file.exists():
        with open(learning_state_file) as f:
            learning_state = json.load(f)
            loop_state = learning_state.get('loop_state', {})
            metrics = learning_state.get('metrics', {})
        print(f"   Learning cycles: {loop_state.get('total_cycles', 0)}")
        print(f"   Successful: {loop_state.get('successful_cycles', 0)}")
        print(f"   Topics learned: {metrics.get('total_topics_learned', 0)}")
        print(f"   Surprises: {metrics.get('total_surprises', 0)}")
    
    # 6. Curiosity state
    curiosity_state_file = Path(workspace_dir) / "cognitive_state" / "curiosity_state.json"
    if curiosity_state_file.exists():
        with open(curiosity_state_file) as f:
            curiosity = json.load(f)
        print(f"\n🔍 Curiosity Engine:")
        print(f"   Interests: {len(curiosity.get('interests', {}))}")
        print(f"   Knowledge gaps: {len(curiosity.get('knowledge_gaps', {}))}")
        print(f"   Questions: {len(curiosity.get('question_bank', []))}")
    
    # 7. Predictive state
    predictive_state_file = Path(workspace_dir) / "cognitive_state" / "predictive_state.json"
    if predictive_state_file.exists():
        print(f"\n🔮 Predictive Coding: Active")
    
    # 8. Level 3 status
    print(f"\n🧬 Level 3: Meta-Learning & World Model")
    evolution_state_file = Path(workspace_dir) / "cognitive_state" / "evolution_state.json"
    if evolution_state_file.exists():
        with open(evolution_state_file) as f:
            evo_state = json.load(f)
        print(f"   Evolution cycles: {evo_state.get('total_cycles', 0)}")
        milestones = evo_state.get('milestones', [])
        print(f"   Milestones: {len(milestones)}")
        if milestones:
            print(f"   Latest: {milestones[-1].get('description', 'N/A')}")
    
    meta_state_file = Path(workspace_dir) / "cognitive_state" / "meta_hyperparams.json"
    if meta_state_file.exists():
        with open(meta_state_file) as f:
            meta = json.load(f)
        print(f"   Meta-optimizer cycles: {meta.get('cycles', 0)}")
    
    imagination_state_file = Path(workspace_dir) / "cognitive_state" / "imagination_state.json"
    if imagination_state_file.exists():
        with open(imagination_state_file) as f:
            imag = json.load(f)
        print(f"   Scenarios generated: {imag.get('scenario_count', 0)}")
        print(f"   Concepts in graph: {len(imag.get('concept_graph', {}))}")
    
    # 9. Sensory Interface status
    sensory_state_file = Path(workspace_dir) / "cognitive_state" / "sensory_state.json"
    if sensory_state_file.exists():
        with open(sensory_state_file) as f:
            sensory = json.load(f)
        print(f"\n🔌 Sensory Interface:")
        print(f"   Registered sensors: {len(sensory.get('registered_sensors', []))}")
        print(f"   Registered actuators: {len(sensory.get('registered_actuators', []))}")
    
    # Architecture summary
    print(f"\n🏗️ Architecture:")
    print(f"   [Level 1] Core Cognition")
    print(f"   - World Model: 512-dim (2 retention layers)")
    print(f"   - Personality: 1024-dim (8 dimensions × 128)")
    print(f"   - Retention Core: 4 layers, 8 heads")
    print(f"   [Level 2] Autonomous Learning")
    print(f"   - Curiosity + Predictive Coding + Learning Scheduler")
    print(f"   - Self-Supervised 6-step learning loop")
    print(f"   [Level 3] Meta-Learning & World Model")
    print(f"   - Hyperparameter Self-Optimizer")
    print(f"   - Meta-Strategy Selector (6 strategies)")
    print(f"   - Architecture Evolver (self-modification)")
    print(f"   - Generative World Model (imagination + counterfactual + prediction)")
    print(f"   - Self-Evolution Tracker (fitness + milestones)")
    print(f"   - Sensory Interface (extensible sensor/actuator framework)")
    
    print(f"\n{'=' * 60}")
    print(f"🧿 HaciCognitiveNet Level 1+2+3 - READY")


def cmd_evolve(args):
    """Run a meta-learning / evolution cycle (Level 3)."""
    logger.info("🧬 Starting meta-learning cycle...\n")
    
    workspace_dir = args.workspace or workspace
    
    from meta_learner import HyperparameterSelfOptimizer, MetaStrategySelector, ArchitectureEvolver
    from self_evolution import SelfEvolutionSystem
    
    # Initialize Level 3 systems
    optimizer = HyperparameterSelfOptimizer(f"{workspace_dir}/cognitive_state")
    strategy_selector = MetaStrategySelector(f"{workspace_dir}/cognitive_state")
    evolver = ArchitectureEvolver(f"{workspace_dir}/cognitive_state")
    evolution = SelfEvolutionSystem(f"{workspace_dir}/cognitive_state")
    
    # Simulate performance metrics
    performance_metrics = {
        'learning_depth': random.uniform(0.3, 0.8),
        'prediction_accuracy': random.uniform(0.4, 0.9),
        'discovery_rate': random.uniform(0.2, 0.7),
        'engagement': random.uniform(0.5, 0.9),
        'emotional_stability': random.uniform(0.4, 0.8),
        'sustained_performance': random.uniform(0.3, 0.7),
    }
    
    # 1. Optimize hyperparameters
    new_params = optimizer.optimize(performance_metrics)
    print(f"🔧 Hyperparameters optimized ({optimizer.optimization_cycles} cycles)")
    
    # 2. Select meta-strategy
    context = {
        'time_of_day': datetime.now().hour,
        'emotional_state': {'mood': 0.6},
        'recent_activity': 'meta_learning',
    }
    strategy = strategy_selector.select_strategy(context)
    print(f"🧠 Meta-strategy selected: {strategy}")
    
    # 3. Check for architecture mutation
    mutation = evolver.propose_mutation(optimizer.global_performance_history)
    if mutation:
        print(f"🧬 Architecture mutation proposed: {mutation['type']} → {mutation.get('component', 'N/A')}")
    else:
        print(f"🧬 No architecture mutation needed (performance improving)")
    
    # 4. Record evolution
    system_state = {
        'learning_cycles': optimizer.optimization_cycles,
        'topics_learned': 3,
        'total_memories': 11,
        'active_interests': 28,
        'knowledge_gaps': 20,
        'insights_generated': len(evolver.mutation_history),
        'surprise_events': 0,
        'total_predictions': 10,
        'emotional_state': {'mood': 0.6},
        'strategy_changes': 1,
        'meta_updates': optimizer.optimization_cycles,
    }
    evo_result = evolution.record_evolution(system_state)
    print(f"\n📊 Evolution recorded:")
    print(f"   Stage: {evo_result['stage']}")
    print(f"   Trajectory: {evo_result['trajectory'].get('trajectory', 'unknown')}")
    
    # Save all
    optimizer.save_state()
    strategy_selector.update_strategy_performance(strategy, sum(performance_metrics.values()) / len(performance_metrics))
    strategy_selector.save_state()
    
    print(f"\n✅ Meta-learning cycle complete!")
    print(f"\n{optimizer.get_summary()}")


def cmd_imagine(args):
    """Run an imagination / world model cycle (Level 3)."""
    logger.info("🌍 Starting imagination cycle...\n")
    
    workspace_dir = args.workspace or workspace
    
    from world_model import GenerativeWorldModel
    
    world_model = GenerativeWorldModel(f"{workspace_dir}/cognitive_state")
    
    # Load memories for concept learning
    memories = get_workspace_memories(workspace_dir)
    
    # Learn concepts from memories
    if memories:
        world_model.imagination.learn_concepts_from_memories(memories)
    
    # Run full imagination cycle
    seed_concepts = ['bellek', 'öğrenme', 'zeka', 'retention']
    if memories:
        # Extract some concepts from memories
        for m in memories[:3]:
            content = m.get('content', '') or m.get('text', '')
            words = [w for w in content.lower().split() if len(w) > 4]
            if words:
                seed_concepts.extend(words[:2])
    
    result = world_model.full_imagination_cycle(seed_concepts[:6], memories)
    
    print("🌍 Imagination Cycle Results")
    print("=" * 40)
    
    if 'scenario' in result:
        s = result['scenario']
        print(f"\n💭 Generated Scenario:")
        print(f"   ID: {s['id']}")
        print(f"   Premise: {s['premise']}")
        print(f"   Novelty: {s['novelty']:.2f}")
        print(f"   Outcomes:")
        for o in s['outcomes']:
            print(f"     - {o}")
    
    if 'counterfactual' in result:
        cf = result['counterfactual']
        print(f"\n🔄 Counterfactual:")
        print(f"   Insight: {cf['insight']}")
        print(f"   Divergence: {cf['divergence']:.2f}")
    
    if 'insight' in result:
        ins = result['insight']
        print(f"\n💡 Creative Insight:")
        print(f"   {ins['text']}")
        print(f"   Novelty: {ins['novelty']:.2f} | Score: {ins['score']:.2f}")
    
    # Save states
    world_model.imagination.save_state()
    world_model.counterfactual.save_state()
    world_model.creative.save_state()
    world_model.predictor.save_state()
    
    print(f"\n✅ Imagination cycle complete!")


def cmd_meta(args):
    """Show meta-learning state (Level 3)."""
    workspace_dir = args.workspace or workspace
    
    from meta_learner import HyperparameterSelfOptimizer, ArchitectureEvolver
    
    optimizer = HyperparameterSelfOptimizer(f"{workspace_dir}/cognitive_state")
    evolver = ArchitectureEvolver(f"{workspace_dir}/cognitive_state")
    
    print(optimizer.get_summary())
    print()
    print(evolver.get_summary())
    
    # Adaptive config
    print(f"\n⚙️ Current Adaptive Config:")
    config = optimizer.get_adaptive_config()
    for module, params in config.items():
        print(f"\n  [{module}]")
        for k, v in params.items():
            print(f"    {k}: {v:.4f}")


def cmd_evolution(args):
    """Show evolution report (Level 3)."""
    workspace_dir = args.workspace or workspace
    
    from self_evolution import SelfEvolutionSystem
    
    evolution = SelfEvolutionSystem(f"{workspace_dir}/cognitive_state")
    print(evolution.get_full_report())


def cmd_learn(args):
    """Run a learning cycle (Level 2)."""
    logger.info("🔄 Starting self-supervised learning cycle...\n")
    
    workspace_dir = args.workspace or workspace
    
    # Load memories
    memories = get_workspace_memories(workspace_dir)
    
    # Initialize learning loop
    from self_supervised_loop import SelfSupervisedLearningLoop
    loop = SelfSupervisedLearningLoop(workspace_dir)
    
    # Run cycle
    cycles = args.cycles or 1
    for i in range(cycles):
        result = loop.run_learning_cycle(memories=memories)
        
        logger.info(f"\n📊 Cycle {i+1} Result:")
        logger.info(f"   Success: {result['success']}")
        logger.info(f"   Duration: {result['duration_sec']:.1f}s")
        
        if 'learn' in result.get('steps', {}):
            learn = result['steps']['learn']
            logger.info(f"   Topics learned: {learn.get('topics_learned', 0)}")
        
        if 'evaluate' in result.get('steps', {}):
            eval_result = result['steps']['evaluate']
            logger.info(f"   Quality: {eval_result.get('quality', 'unknown')}")
    
    loop.save_state()
    logger.info(f"\n✅ Learning complete! {loop.get_summary()}")


def cmd_curious(args):
    """Show curiosity state (Level 2)."""
    from curiosity_engine import CuriosityEngine
    
    workspace_dir = args.workspace or workspace
    engine = CuriosityEngine(workspace_dir)
    
    state = engine.get_curiosity_state()
    
    print("🔍 Curiosity Engine State")
    print("=" * 40)
    print(f"\n📊 Interests: {state['n_interests']}")
    for interest in state.get('top_interests', []):
        print(f"   {interest['topic']}: {interest['interest']:.3f}")
    
    print(f"\n🕳️ Knowledge Gaps: {state['n_knowledge_gaps']}")
    for topic, score in state.get('top_gaps', []):
        print(f"   {topic}: {score:.3f}")
    
    print(f"\n❓ Questions: {state['n_questions']}")
    
    should_explore, reason = state.get('should_explore', (False, 'unknown'))
    print(f"\n🎯 Should explore: {should_explore} ({reason})")
    
    if state.get('last_exploration'):
        print(f"\n📅 Last exploration: {state['last_exploration']['timestamp']}")
        print(f"   Topic: {state['last_exploration']['topic']}")


def cmd_predict(args):
    """Show predictive coding stats (Level 2)."""
    from predictive_coding import PredictiveCodingSystem
    
    workspace_dir = args.workspace or workspace
    system = PredictiveCodingSystem(workspace_dir)
    
    stats = system.get_system_stats()
    
    print("🔮 Predictive Coding System")
    print("=" * 40)
    
    print(f"\n📊 Conversation Model:")
    conv = stats.get('conversation', {})
    print(f"   Accuracy: {conv.get('accuracy', 0):.3f}")
    print(f"   Total predictions: {conv.get('total', 0)}")
    
    print(f"\n😮 Surprise Stats:")
    topic_surprise = stats.get('topic_surprise', {})
    print(f"   Avg surprise: {topic_surprise.get('avg_surprise', 0):.4f}")
    print(f"   Max surprise: {topic_surprise.get('max_surprise', 0):.4f}")
    print(f"   Events tracked: {topic_surprise.get('n_events', 0)}")
    
    print(f"\n⚙️ Surprise threshold: {stats.get('surprise_threshold', 0.5)}")


# ============================================================
# SOCIAL INTELLIGENCE COMMANDS (Duygusal Zeka)
# ============================================================

def cmd_social(args):
    """Show social intelligence state."""
    from social_trainer import SocialIntelligenceTrainer
    
    workspace_dir = args.workspace or workspace
    trainer = SocialIntelligenceTrainer(f"{workspace_dir}/cognitive_state")
    
    print(trainer.get_personality_summary())
    
    report = trainer.get_learning_report()
    stats = report['stats']
    
    print(f"\n📊 Eğitim İstatistikleri:")
    print(f"   Toplam etkileşim: {stats['total_interactions']}")
    print(f"   Pozitif sonuçlar: {stats['positive_outcomes']}")
    print(f"   Negatif sonuçlar: {stats['negative_outcomes']}")
    print(f"   Öğrenilen dersler: {stats['lessons_learned']}")


def cmd_personality(args):
    """Show detailed personality profile."""
    from social_trainer import PersonalityDevelopment
    
    workspace_dir = args.workspace or workspace
    personality = PersonalityDevelopment(f"{workspace_dir}/cognitive_state")
    
    profile = personality.get_personality_profile()
    stage = profile['development_stage']
    
    stage_turkish = {
        'infancy': '🍼 Bebeklik',
        'childhood': '👧 Çocukluk',
        'adolescence': '🧑 Ergenlik',
        'adulthood': '🧔 Yetişkinlik',
        'mastery': '🧘 Olgunluk',
    }
    
    print("🧬 Hacı Kişilik Profili")
    print("=" * 40)
    print(f"\n🎯 Gelişim Aşaması: {stage_turkish.get(stage, stage)}")
    print(f"📈 Toplam Etkileşim: {profile['interaction_count']}")
    
    if profile['last_interaction']:
        last = profile['last_interaction']
        print(f"\n🔄 Son Öğrenme:")
        print(f"   Tür: {last['type']}")
        print(f"   Ders: {last['lesson']}")
        print(f"   Zaman: {last['timestamp'][:19]}")
    
    print(f"\n🎭 Kişilik Özellikleri:")
    traits = profile['traits']
    for trait, value in sorted(traits.items(), key=lambda x: -x[1]):
        bar = '█' * int(value * 10) + '░' * (10 - int(value * 10))
        emoji = {
            'warmth': '❤️', 'empathy': '🤝', 'humor_style': '😄',
            'assertiveness': '💪', 'loyalty': '🛡️', 'playfulness': '🎮',
            'wisdom': '🧠', 'mischief': '😈'
        }.get(trait, '📊')
        print(f"   {emoji} {trait:15s} {bar} {value:.2f}")
    
    print(f"\n💬 Konuşma Stilleri:")
    if personality.personality_file.exists():
        print(f"   (Konuşma zekası state yüklü)")


def cmd_interact(args):
    """Process an interaction manually."""
    from social_trainer import (
        SocialIntelligenceTrainer, 
        analyze_message_emotion,
        detect_interaction_type
    )
    
    workspace_dir = args.workspace or workspace
    trainer = SocialIntelligenceTrainer(f"{workspace_dir}/cognitive_state")
    
    # Interactive input
    print("🤝 Etkileşim Kaydı")
    print("=" * 40)
    
    situation = input("Durum (humor_moment/shared_stress/trust_moment/etc): ").strip()
    user_emotion = input("Kullanıcı duygusu (happy/amused/frustrated/bored): ").strip()
    my_response = input("Hacı'nın cevabı: ").strip()
    outcome = input("Sonuç (positive/negative/neutral): ").strip()
    lesson = input("Öğrenilen ders: ").strip()
    
    if all([situation, user_emotion, my_response, outcome, lesson]):
        trainer.process_interaction(
            situation=situation,
            user_emotion=user_emotion,
            my_response=my_response,
            outcome=outcome,
            lesson=lesson,
        )
        print("\n✅ Etkileşim kaydedildi ve öğrenildi!")
        print(trainer.get_personality_summary())
    else:
        print("\n❌ Tüm alanlar doldurulmalı!")


def cmd_mistakes(args):
    """Show negative outcome learnings - hatalardan öğrenilen dersler."""
    from negative_learner import NegativeOutcomeLearner
    
    workspace_dir = args.workspace or workspace
    learner = NegativeOutcomeLearner(f"{workspace_dir}/cognitive_state")
    
    print(learner.get_summary())
    
    # Kontrol et: avoid listesi
    print("\n🔍 Örnek kontrol:")
    test_actions = [
        "aynı şeyi tekrar yanlış anla",
        "konuyu çok uzat",
        "başkan acele ederken bekle"
    ]
    for action in test_actions:
        avoid, reason = learner.should_avoid(action)
        status = "🚫 KAÇIN!" if avoid else "✅ Tamam"
        print(f"   {status}: {action}")
        if avoid:
            print(f"      → {reason}")


def cmd_never(args):
    """Show 'never do' rules - asla yapma kuralları."""
    from negative_learner import NegativeOutcomeLearner
    
    workspace_dir = args.workspace or workspace
    learner = NegativeOutcomeLearner(f"{workspace_dir}/cognitive_state")
    
    rules = learner.get_never_rules()
    
    print("🚨 ASLA YAPMA KURALLARI")
    print("=" * 40)
    
    if rules:
        for i, rule in enumerate(rules, 1):
            print(f"\n  {i}. ⛔ {rule}")
        print(f"\n📊 Toplam {len(rules)} kalıcı kural")
    else:
        print("\n  (Henüz kalıcı kural yok - 3 kere tekrarlanan hata olunca oluşur)")
    
    report = learner.get_learning_report()
    print(f"\n📈 Toplam hata: {report['total_mistakes']}")
    print(f"   Ortalama ciddiyet: {report['severity_avg']:.2f}")
    
    repeated = report['repeated_mistakes']
    if repeated:
        print(f"\n⚠️ Tekrar riski olanlar:")
        for mistake, count in list(repeated.items())[:5]:
            print(f"   {count}x: {mistake[:60]}...")


def main():
    parser = argparse.ArgumentParser(description="HaciCognitiveNet CLI")
    parser.add_argument('command', choices=[
        'init', 'status', 'dream', 'train', 'summary',
        'learn', 'curious', 'predict',
        'evolve', 'imagine', 'meta', 'evolution',  # Level 3 commands
        'social', 'personality', 'interact',  # Social Intelligence
        'mistakes', 'never'  # Negative Outcome Learning
    ], help='Command to run')
    parser.add_argument('--workspace', '-w', help='Workspace directory')
    parser.add_argument('--cycles', '-c', type=int, help='Number of cycles')
    parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
    
    args = parser.parse_args()
    
    commands = {
        'init': cmd_init,
        'status': cmd_status,
        'dream': cmd_dream,
        'train': cmd_train,
        'summary': cmd_summary,
        'learn': cmd_learn,
        'curious': cmd_curious,
        'predict': cmd_predict,
        'evolve': cmd_evolve,
        'imagine': cmd_imagine,
        'meta': cmd_meta,
        'evolution': cmd_evolution,
        'social': cmd_social,
        'personality': cmd_personality,
        'interact': cmd_interact,
        'mistakes': cmd_mistakes,
        'never': cmd_never,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()

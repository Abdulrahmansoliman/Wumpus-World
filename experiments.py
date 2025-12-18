from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle

from wumpus.agent_prolog import PrologAgent
from wumpus.agent_qlearn import QLearningAgent
from wumpus.bridge import PrologBridge
from wumpus.world import World


def simulate_prolog_episode(size: int, pit_prob: float, seed: int, max_steps: int = 200) -> str:
    world = World(size=size, pit_probability=pit_prob, seed=seed)
    kb_path = Path(__file__).parent / "wumpus" / "prolog" / "knowledge_base.pl"
    agent = PrologAgent(grid_size=size, bridge=PrologBridge(kb_path))
    agent.reset()
    percept = world.initial_percept()

    steps = 0
    while steps < max_steps:
        action = agent.act(percept)
        result = world.step(action)
        steps += 1
        if result.terminated:
            return result.outcome or "unknown"
        percept = result.percept
    return "timeout"


def simulate_qlearn_episode(size: int, pit_prob: float, seed: int, q_table: dict, max_steps: int = 200) -> str:
    world = World(size=size, pit_probability=pit_prob, seed=seed)
    agent = QLearningAgent(grid_size=size)
    agent.q_table = q_table
    agent.epsilon = 0.0  # Pure exploitation
    agent.reset()
    percept = world.initial_percept()

    steps = 0
    while steps < max_steps:
        action = agent.act(percept)
        result = world.step(action)
        steps += 1
        if result.terminated:
            return result.outcome or "unknown"
        percept = result.percept
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiments for Wumpus agents.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--pit-prob", type=float, default=0.2)
    parser.add_argument("--agent", type=str, choices=["prolog", "qlearn", "both"], default="both")
    parser.add_argument("--qlearn-model", type=str, default="models/q_hybrid.pkl")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    # Test Prolog agent
    if args.agent in ["prolog", "both"]:
        print("=" * 50)
        print("Testing Prolog Agent")
        print("=" * 50)
        prolog_outcomes = Counter()
        for seed in range(args.episodes):
            print(f"Prolog episode {seed + 1}/{args.episodes}...", end=" ", flush=True)
            outcome = simulate_prolog_episode(
                size=args.size, pit_prob=args.pit_prob, seed=seed, max_steps=args.max_steps
            )
            prolog_outcomes[outcome] += 1
            print(outcome)

        total = sum(prolog_outcomes.values())
        print(f"\nProlog Agent Results ({total} episodes):")
        for key, val in sorted(prolog_outcomes.items()):
            pct = 100.0 * val / total if total else 0.0
            print(f"  {key}: {val} ({pct:.1f}%)")

    # Test Q-learning agent
    if args.agent in ["qlearn", "both"]:
        print("\n" + "=" * 50)
        print("Testing Q-Learning Agent")
        print("=" * 50)
        
        # Load Q-table
        model_path = Path(args.qlearn_model)
        if not model_path.exists():
            print(f"Error: Q-learning model not found at {model_path}")
            return
        
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        q_table = payload["q"]  # Extract actual Q-table from nested structure
        print(f"Loaded Q-table with {len(q_table)} states from {model_path}")
        
        qlearn_outcomes = Counter()
        for seed in range(args.episodes):
            print(f"Q-learning episode {seed + 1}/{args.episodes}...", end=" ", flush=True)
            outcome = simulate_qlearn_episode(
                size=args.size, pit_prob=args.pit_prob, seed=seed, q_table=q_table, max_steps=args.max_steps
            )
            qlearn_outcomes[outcome] += 1
            print(outcome)

        total = sum(qlearn_outcomes.values())
        print(f"\nQ-Learning Agent Results ({total} episodes):")
        for key, val in sorted(qlearn_outcomes.items()):
            pct = 100.0 * val / total if total else 0.0
            print(f"  {key}: {val} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

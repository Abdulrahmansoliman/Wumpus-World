from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from wumpus.agent_prolog import PrologAgent
from wumpus.bridge import PrologBridge
from wumpus.world import World


def simulate_episode(size: int, pit_prob: float, seed: int) -> str:
    world = World(size=size, pit_probability=pit_prob, seed=seed)
    kb_path = Path(__file__).parent / "wumpus" / "prolog" / "knowledge_base.pl"
    agent = PrologAgent(grid_size=size, bridge=PrologBridge(kb_path))
    agent.reset()
    percept = world.initial_percept()

    while True:
        action = agent.act(percept)
        result = world.step(action)
        if result.terminated:
            return result.outcome or "unknown"
        percept = result.percept


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiments for Wumpus agent.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--pit-prob", type=float, default=0.2)
    args = parser.parse_args()

    outcomes = Counter()
    for seed in range(args.episodes):
        outcome = simulate_episode(size=args.size, pit_prob=args.pit_prob, seed=seed)
        outcomes[outcome] += 1

    total = sum(outcomes.values())
    print(f"Ran {total} episodes.")
    for key, val in outcomes.items():
        pct = 100.0 * val / total if total else 0.0
        print(f"{key}: {val} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

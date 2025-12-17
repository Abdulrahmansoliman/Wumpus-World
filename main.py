from __future__ import annotations

import argparse
from pathlib import Path

from wumpus.agent_base import HumanAgent
from wumpus.agent_prolog import PrologAgent
from wumpus.bridge import PrologBridge
from wumpus.world import Action, World


def run_episode(mode: str, size: int, pit_prob: float, seed: int) -> str:
    world = World(size=size, pit_probability=pit_prob, seed=seed)
    kb_path = Path(__file__).parent / "wumpus" / "prolog" / "knowledge_base.pl"
    agent = (
        PrologAgent(grid_size=size, bridge=PrologBridge(kb_path))
        if mode == "agent"
        else HumanAgent()
    )
    agent.reset()

    percept = world.initial_percept()
    step = 0
    print(f"Starting world seed={seed}, size={size}, pit_prob={pit_prob}")
    while True:
        action = agent.act(percept)
        result = world.step(action)
        step += 1
        print(f"Step {step}: action={action.value}, percept={result.percept}")
        if result.terminated:
            print(f"Episode finished: {result.outcome}, has_gold={result.has_gold}")
            return result.outcome or "unknown"
        percept = result.percept


def main() -> None:
    parser = argparse.ArgumentParser(description="Wumpus World simulator.")
    parser.add_argument("--mode", choices=["agent", "human"], default="agent")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--pit-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_episode(mode=args.mode, size=args.size, pit_prob=args.pit_prob, seed=args.seed)


if __name__ == "__main__":
    main()

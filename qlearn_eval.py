from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from wumpus.agent_qlearn import QConfig, QLearningAgent
from wumpus.world import World


def run_eval(
    episodes: int,
    size: int,
    pit_prob: float,
    seed_offset: int,
    eps: float,
    payload: dict,
    max_steps: int,
    world_seed: int = None,
    solvable_only: bool = False,
) -> dict:
    cfg = QConfig(alpha=payload["alpha"], gamma=payload["gamma"], epsilon=eps)
    agent = QLearningAgent(grid_size=size, config=cfg, qtable=payload["q"])
    agent.set_seed(123)

    outcomes = Counter()
    scores = []
    steps = []
    tested = 0
    skipped = 0

    i = 0
    while tested < episodes:
        ws = world_seed if world_seed is not None else (seed_offset + i)
        world = World(size=size, pit_probability=pit_prob, seed=ws)
        i += 1
        
        # Skip unsolvable worlds if requested
        if solvable_only and not world.is_solvable():
            skipped += 1
            continue
        
        tested += 1
        agent.reset()
        agent.set_epsilon(eps)
        agent.set_seed(123 + tested)

        percept = world.initial_percept()
        step = 0
        while True:
            action = agent.act(percept)
            result = world.step(action)
            step += 1
            percept = result.percept
            if result.terminated:
                outcomes[result.outcome or "unknown"] += 1
                scores.append(world.score)
                steps.append(step)
                break
            if step >= max_steps:
                outcomes["timeout"] += 1
                scores.append(world.score)
                steps.append(step)
                break

    return {
        "eps": eps,
        "win": outcomes["escaped_with_gold"] / episodes,
        "escape_wo_gold": outcomes["escaped_without_gold"] / episodes,
        "death": (outcomes["fell_in_pit"] + outcomes["eaten_by_wumpus"]) / episodes,
        "timeout": outcomes["timeout"] / episodes,
        "avg_score": sum(scores) / episodes,
        "avg_steps": sum(steps) / episodes,
        "skipped": skipped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained Q-learning agent across riskiness levels.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--pit-prob", type=float, default=0.2)
    ap.add_argument("--model", type=str, default="models/qtable.pkl")
    ap.add_argument("--epsilons", type=str, default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--world-seed", type=int, default=None, help="if set, evaluate repeatedly on one fixed world")
    ap.add_argument("--out-plot", type=str, default="results/qlearn_eval.png")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--solvable-only", action="store_true", help="only evaluate on solvable grids")
    args = ap.parse_args()

    payload = QLearningAgent.load(Path(args.model))
    eps_list = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]

    rows = []
    for eps in eps_list:
        rows.append(
            run_eval(
                episodes=args.episodes,
                size=args.size,
                pit_prob=args.pit_prob,
                seed_offset=args.seed_offset,
                eps=eps,
                payload=payload,
                max_steps=args.max_steps,
                world_seed=args.world_seed,
                solvable_only=args.solvable_only,
            )
        )

    if args.solvable_only and rows:
        print(f"(Skipped {rows[0]['skipped']} unsolvable worlds)")
    print("eps | win% | escape_wo_gold% | death% | timeout% | avg_score | avg_steps")
    for r in rows:
        print(
            f"{r['eps']:.2f} | {r['win']*100:5.1f} | {r['escape_wo_gold']*100:14.1f} | "
            f"{r['death']*100:5.1f} | {r['timeout']*100:7.1f} | {r['avg_score']:9.1f} | {r['avg_steps']:9.1f}"
        )

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot([r["eps"] for r in rows], [r["win"] for r in rows], marker="o")
    plt.title("Win rate vs riskiness (epsilon)")
    plt.xlabel("epsilon (riskiness)")
    plt.ylabel("win rate")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()

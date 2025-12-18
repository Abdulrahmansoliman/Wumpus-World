from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from wumpus.agent_qlearn import QConfig, QLearningAgent
from wumpus.world import World


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tabular Q-learning agent for Wumpus World.")
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--pit-prob", type=float, default=0.2)
    ap.add_argument("--train-seed", type=int, default=0, help="starting seed for world generation")
    ap.add_argument("--num-worlds", type=int, default=50, help="number of different worlds to cycle through during training")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--epsilon", type=float, default=0.4)
    ap.add_argument("--eps-decay", type=float, default=0.9995)
    ap.add_argument("--eps-min", type=float, default=0.05)
    ap.add_argument("--out-model", type=str, default="models/qtable.pkl")
    ap.add_argument("--out-csv", type=str, default="results/qlearn_train.csv")
    ap.add_argument("--out-plot", type=str, default="results/qlearn_train.png")
    args = ap.parse_args()

    cfg = QConfig(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    agent = QLearningAgent(grid_size=args.size, config=cfg)
    agent.set_seed(0)

    out_model = Path(args.out_model)
    out_csv = Path(args.out_csv)
    out_plot = Path(args.out_plot)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    rewards = []
    eps = args.epsilon

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "final_score", "outcome", "epsilon"])

        for ep in range(args.episodes):
            # Cycle through multiple worlds for generalization
            world_seed = args.train_seed + (ep % args.num_worlds)
            world = World(size=args.size, pit_probability=args.pit_prob, seed=world_seed)
            agent.reset()
            agent.set_epsilon(eps)

            percept = world.initial_percept()
            prev_score = world.score
            terminated = False
            outcome = "timeout"

            for _ in range(args.max_steps):
                action = agent.act(percept)
                result = world.step(action)

                reward = world.score - prev_score
                prev_score = world.score

                agent.observe(reward, result.terminated)

                percept = result.percept
                if result.terminated:
                    agent.finalize_episode()
                    terminated = True
                    outcome = result.outcome or "unknown"
                    break

            # If we hit max_steps without termination, punish timeout and finalize
            if not terminated:
                timeout_penalty = -500
                world.score += timeout_penalty  # so your training curve reflects it
                agent.observe(timeout_penalty, True)
                agent.finalize_episode()

            rewards.append(world.score)
            w.writerow([ep, world.score, outcome, eps])

            eps = max(args.eps_min, eps * args.eps_decay)

    agent.save(out_model)
    print(f"Saved Q-table: {out_model}")
    print(f"Saved training log: {out_csv}")

    window = 100
    smoothed = [mean(rewards[max(0, i - window + 1) : i + 1]) for i in range(len(rewards))]

    plt.figure()
    plt.plot(smoothed)
    plt.title("Q-learning training curve (rolling mean score)")
    plt.xlabel("episode")
    plt.ylabel("mean score")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()

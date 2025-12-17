# Wumpus-World

Simulation where an AI agent builds up its KB of percepts as it moves through the Wumpus World.  
Hybrid architecture: Python simulator + Prolog knowledge base (via SWI-Prolog).

## Quick start

```bash
python main.py --mode agent --size 4 --pit-prob 0.2 --seed 0
python experiments.py --episodes 20
```

Requirements:
- Python 3.10+
- SWI-Prolog (`swipl` on PATH) for full Prolog reasoning. If unavailable, the agent falls back to a minimal safe-set (visited squares).

## Project layout

- `wumpus/world.py` – environment OOP model (actions, percepts, hazards)
- `wumpus/agent_prolog.py` – Prolog-backed agent policy
- `wumpus/bridge.py` – subprocess bridge to SWI-Prolog
- `wumpus/planning.py` – shortest-path helper (BFS)
- `wumpus/prolog/knowledge_base.pl` – propositional KB + model enumerator
- `main.py` – run a single episode (agent or human)
- `experiments.py` – batch runs for statistics

# Wumpus-World

A Wumpus World simulator with two AI agents: a **Prolog-based logical agent** (model-checking inference) and a **Q-learning agent** (tabular RL with hybrid safety rules). Includes an interactive Tkinter GUI.

---

## Quick Start

### Requirements
- **Python 3.10+** (tested on 3.12)
- **SWI-Prolog** with `swipl` on PATH (for the Prolog agent)
- **matplotlib** (for evaluation plots)
- Tkinter (bundled with most Python installations)

### 1. Launch the GUI (Recommended)
```bash
python gui.py
```
This opens the interactive game window. **By default, it starts in Prolog mode.**

### 2. Run Batch Experiments (Both Agents)
```bash
python experiments.py --episodes 20
```
This tests both the Prolog and Q-learning agents and prints win/death/escape statistics.

### 3. Evaluate Q-Learning Across Epsilon Values
```bash
python qlearn_eval.py --episodes 100 --epsilons 0,0.05,0.1,0.2,0.3
```
Generates `qlearn_eval.png` showing win rate vs exploration parameter.

---

## GUI Guide

### Switching Modes
Use the **menu bar at the top**: `Mode → Human / Prolog / Q-Learn`
- **Human**: You control the agent with keyboard (arrow keys, G to grab, C to climb)
- **Prolog**: Logical agent using model-checking inference *(see note below)*
- **Q-Learn**: Reinforcement learning agent with adjustable ε (riskiness slider)

### Starting a New Game
`File → New Game` opens a dialog to set:
- **Grid size**: 3×3 or 4×4
- **Seed**: Random seed for reproducible worlds
- **Pit probability**: Default 0.2

### Controls & Options
- **Reveal World**: Toggle to see hidden pits/Wumpus/gold (for debugging or to verify agent behavior)
- **ε Slider** (Q-Learn mode): Higher = more random exploration, Lower = exploit learned policy

### ⚠️ Important Notes
1. **Prolog mode is slow on first move** — it enumerates all consistent world models. Give it a few seconds after clicking "Step" or "Auto-Run". Subsequent moves are faster due to caching.
2. **Reveal World** is useful for understanding agent decisions, but the agents don't cheat—they only see percepts!
3. The Q-learning model is pre-trained and loaded from `models/q_hybrid.pkl`.

---

## 🎯 Fun Challenge: Can You Beat the Agents?

Switch to **Human mode** and try to win! The Prolog agent has a 40% win rate and *never* dies. The Q-learning agent wins ~33% but dies a lot. 

**Seeds I used for testing (try these!):**
- `10` — Challenging but solvable
- `15` — Tricky pit placement
- `25` — Good for testing return-home logic

Can you beat the agents? Good luck! 🍀

---

## 📁 Project Layout

| File | Description |
|------|-------------|
| `gui.py` | Tkinter GUI — **start here for interactive testing** |
| `experiments.py` | Batch testing both agents, prints statistics |
| `wumpus/world.py` | Core simulator: actions, percepts, scoring |
| `wumpus/agent_prolog.py` | Prolog-backed logical agent |
| `wumpus/agent_qlearn.py` | Q-learning agent with hybrid safety rules |
| `wumpus/bridge.py` | Python ↔ SWI-Prolog subprocess interface |
| `wumpus/planning.py` | BFS shortest-path for navigation |
| `wumpus/prolog/knowledge_base.pl` | Prolog KB + model enumeration rules |
| `models/q_hybrid.pkl` | Pre-trained Q-table (137k states) |
| `qlearn_train.py` | Train a new Q-learning model |
| `qlearn_eval.py` | Evaluate Q-learning across ε values |

---

## 📊 Expected Output

Running `python experiments.py --episodes 100` should produce something like:
```
Prolog Agent:  Win 40.0% | Death 0.0% | Safe Escape 600%
Q-Learn Agent: Win 33.0% | Death 62.0% | Safe Escape 5.0%
```

The Prolog agent is conservative (never dies, often escapes without gold). The Q-learner is aggressive (higher death rate but competitive win rate).

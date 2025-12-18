from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .agent_base import Agent
from .world import Action, Direction, Percept, Position

# State now includes belief memory masks for generalization
State = Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]  # Added glimmer


@dataclass
class QConfig:
    alpha: float = 0.2
    gamma: float = 0.9
    epsilon: float = 0.1


class QLearningAgent(Agent):
    """
    Tabular Q-learning agent.
    - In GUI: uses a loaded Q-table + epsilon slider (riskiness).
    - In training: call observe() after each env step and finalize_episode() at terminal.
    """

    def __init__(self, grid_size: int, config: Optional[QConfig] = None, qtable: Optional[dict] = None) -> None:
        self.grid_size = grid_size
        self.cfg = config or QConfig()
        self.q: Dict[State, Dict[str, float]] = qtable or {}
        self.rng = random.Random(0)
        self.reset()

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        payload = {
            "grid_size": self.grid_size,
            "alpha": self.cfg.alpha,
            "gamma": self.cfg.gamma,
            "q": self.q,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: Path) -> dict:
        with path.open("rb") as f:
            payload = pickle.load(f)
        return payload

    # ---------- runtime controls ----------
    def set_epsilon(self, eps: float) -> None:
        self.cfg.epsilon = max(0.0, min(1.0, float(eps)))

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    # ---------- Agent interface ----------
    def reset(self) -> None:
        self.position: Position = (1, 1)
        self.direction: Direction = Direction.EAST
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_dead = False

        self.last_action: Optional[Action] = None
        self.step_count = 0

        self.prev_state: Optional[State] = None
        self.prev_action: Optional[str] = None
        self.pending_reward: Optional[float] = None
        self.pending_done: bool = False

        # Belief memory masks for generalization across worlds
        self.visited_mask = 0
        self.breeze_mask = 0
        self.no_breeze_mask = 0
        self.stench_mask = 0
        self.no_stench_mask = 0

        # Safety tracking for hybrid approach
        self.safe_cells = {(1, 1)}  # Start is always safe
        self.maybe_pit = set()       # Cells that might have a pit
        self.maybe_wumpus = set()    # Cells that might have wumpus
        self.confirmed_pit = set()   # Cells confirmed to have pits
        self.confirmed_no_pit = {(1, 1)}  # Cells confirmed safe from pits

        # Stuck detector to break turning loops
        self.stuck_steps = 0
        self.last_pos = self.position

    def act(self, percept: Percept) -> Action:
        self._ingest_feedback(percept)
        self._update_belief(percept)
        self._update_safety(percept)

        # Reflex overrides (always optimal, removes randomness here)
        if percept.glitter and not self.has_gold:
            self.prev_state = self._encode_state(percept)
            self.prev_action = Action.GRAB.value
            self.last_action = Action.GRAB
            self.has_gold = True
            self.step_count += 1
            return Action.GRAB

        if self.position == (1, 1) and self.has_gold:
            self.prev_state = self._encode_state(percept)
            self.prev_action = Action.CLIMB.value
            self.last_action = Action.CLIMB
            self.step_count += 1
            return Action.CLIMB

        # Stop exploring once we have gold (focus on returning safely)
        if self.has_gold:
            self.cfg.epsilon = 0.0

        s = self._encode_state(percept)

        if self.prev_state is not None and self.prev_action is not None and self.pending_reward is not None:
            self._q_update(self.prev_state, self.prev_action, self.pending_reward, s, self.pending_done)
            self.pending_reward = None
            self.pending_done = False

        legal = self._legal_actions(percept)

        # REFLEX: If we just bumped into a wall, turn instead of trying forward again
        if percept.bump:
            turn_actions = [Action.TURN_LEFT.value, Action.TURN_RIGHT.value]
            a = self.rng.choice(turn_actions)
        # When returning with gold, use special return logic
        elif self.has_gold:
            a = self._return_home(percept, legal, s)
        else:
            # HYBRID SAFETY: Filter out dangerous actions
            safe_legal = self._filter_safe_actions(legal, percept, risky=False)
            
            # If no safe actions or stuck too long, allow risky moves
            if not safe_legal or self.stuck_steps >= 20:
                safe_legal = self._filter_safe_actions(legal, percept, risky=True)
            
            # If still no actions, fall back to all legal (desperate)
            if not safe_legal:
                safe_legal = legal

            # Breeze-aware stuck breaking with graduated escalation
            if self.stuck_steps >= 8:
                if not percept.breeze and not percept.bump and Action.FORWARD.value in safe_legal:
                    a = Action.FORWARD.value
                elif self.stuck_steps >= 16:
                    a = self._epsilon_greedy(s, safe_legal, 0.7)
                else:
                    turn_actions = [a for a in [Action.TURN_LEFT.value, Action.TURN_RIGHT.value] if a in safe_legal]
                    if turn_actions:
                        a = self.rng.choice(turn_actions)
                    else:
                        a = self._epsilon_greedy(s, safe_legal, self.cfg.epsilon)
            else:
                a = self._epsilon_greedy(s, safe_legal, self.cfg.epsilon)

        self.prev_state = s
        self.prev_action = a
        self.last_action = Action(a)
        self.step_count += 1

        return self.last_action

    def observe(self, reward: float, done: bool) -> None:
        self.pending_reward = float(reward)
        self.pending_done = bool(done)

    def finalize_episode(self) -> None:
        if self.prev_state is None or self.prev_action is None or self.pending_reward is None:
            return
        old = self._get_q(self.prev_state, self.prev_action)
        target = self.pending_reward
        new = old + self.cfg.alpha * (target - old)
        self._set_q(self.prev_state, self.prev_action, new)
        self.pending_reward = None
        self.pending_done = False

    # ---------- internals ----------
    def _encode_state(self, p: Percept) -> State:
        # Include position + belief memory for generalization across worlds
        x, y = self.position
        dir_idx = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST].index(self.direction)
        return (
            x,
            y,
            dir_idx,
            int(self.has_gold),
            int(self.has_arrow),
            int(self.wumpus_dead),
            self.visited_mask,
            self.breeze_mask,
            self.no_breeze_mask,
            self.stench_mask,
            self.no_stench_mask,
            int(p.glitter),
            int(p.glimmer),  # Gold is nearby
        )

    def _legal_actions(self, p: Percept) -> List[str]:
        # Always allow movement primitives (RL must be allowed to explore)
        actions = [Action.TURN_LEFT.value, Action.TURN_RIGHT.value, Action.FORWARD.value]

        if p.glitter:
            actions.append(Action.GRAB.value)
        if self.position == (1, 1) and self.has_gold:
            actions.append(Action.CLIMB.value)
        # Only allow shooting when stench is present AND facing unvisited cell
        if self.has_arrow and p.stench and not self.wumpus_dead:
            ahead = self._forward(self.position, self.direction)
            # Only shoot toward unvisited or maybe-wumpus cells (not confirmed safe)
            if ahead not in self.safe_cells or ahead in self.maybe_wumpus:
                actions.append(Action.SHOOT.value)
        return actions

    def _filter_safe_actions(self, actions: List[str], p: Percept, risky: bool = False) -> List[str]:
        """Filter out actions that would lead to known danger.
        
        If risky=True, allow maybe_pit cells (but still avoid confirmed pits).
        """
        safe = []
        ahead = self._forward(self.position, self.direction)
        
        for a in actions:
            if a == Action.FORWARD.value:
                # Never walk into confirmed pits
                if ahead in self.confirmed_pit:
                    continue
                # Don't walk into maybe_pit cells unless risky mode
                if not risky and ahead in self.maybe_pit and ahead not in self.confirmed_no_pit:
                    continue
                # Don't walk into maybe_wumpus unless wumpus is dead
                if ahead in self.maybe_wumpus and not self.wumpus_dead:
                    if not risky:
                        continue
            safe.append(a)
        return safe

    def _return_home(self, percept: Percept, legal: List[str], s: State) -> str:
        """Navigate back to (1,1) through visited safe cells using simple BFS."""
        from collections import deque
        
        target = (1, 1)
        if self.position == target:
            # Should have been handled by reflex, but just in case
            return Action.CLIMB.value
        
        # BFS to find path through safe_cells
        queue = deque([(self.position, [])])
        visited_bfs = {self.position}
        
        while queue:
            pos, path = queue.popleft()
            if pos == target:
                # Found path! First step tells us where to go
                if path:
                    next_pos = path[0]
                    return self._move_toward(next_pos, legal)
                break
            
            for neighbor in self._neighbors(pos):
                if neighbor not in visited_bfs and neighbor in self.safe_cells:
                    visited_bfs.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        # No safe path found - try any visited cell
        ahead = self._forward(self.position, self.direction)
        if ahead in self.safe_cells and Action.FORWARD.value in legal:
            return Action.FORWARD.value
        
        # Turn to find a visited cell
        for turn in [Action.TURN_LEFT.value, Action.TURN_RIGHT.value]:
            if turn in legal:
                return turn
        
        # Fallback to Q-learning
        return self._epsilon_greedy(s, legal, 0.0)

    def _move_toward(self, target_pos: Position, legal: List[str]) -> str:
        """Return action to move toward target position."""
        dx = target_pos[0] - self.position[0]
        dy = target_pos[1] - self.position[1]
        
        # Determine desired direction
        if dx > 0:
            desired = Direction.EAST
        elif dx < 0:
            desired = Direction.WEST
        elif dy > 0:
            desired = Direction.NORTH
        else:
            desired = Direction.SOUTH
        
        # If facing right direction, go forward
        if self.direction == desired and Action.FORWARD.value in legal:
            return Action.FORWARD.value
        
        # Need to turn - figure out which way
        left_dir = self.direction.turn_left()
        if left_dir == desired:
            return Action.TURN_LEFT.value
        
        right_dir = self.direction.turn_right()
        if right_dir == desired:
            return Action.TURN_RIGHT.value
        
        # Need to turn around (180) - turn either way
        return Action.TURN_LEFT.value

    def _epsilon_greedy(self, s: State, legal: List[str], epsilon: float) -> str:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)
        qs = [(a, self._get_q(s, a)) for a in legal]
        best_q = max(q for _, q in qs)
        best_actions = [a for a, q in qs if abs(q - best_q) < 1e-9]
        return self.rng.choice(best_actions)

    def _q_update(self, prev_s: State, prev_a: str, r: float, next_s: State, done: bool) -> None:
        old = self._get_q(prev_s, prev_a)
        if done:
            target = r
        else:
            # Only max over actions that have been seen (avoids overestimating from illegal actions)
            max_next = max(self.q.get(next_s, {}).values(), default=0.0)
            target = r + self.cfg.gamma * max_next
        new = old + self.cfg.alpha * (target - old)
        self._set_q(prev_s, prev_a, new)

    def _get_q(self, s: State, a: str) -> float:
        return float(self.q.get(s, {}).get(a, 0.0))

    def _set_q(self, s: State, a: str, v: float) -> None:
        self.q.setdefault(s, {})[a] = float(v)

    def _ingest_feedback(self, p: Percept) -> None:
        if self.last_action == Action.TURN_LEFT:
            self.direction = self.direction.turn_left()
        elif self.last_action == Action.TURN_RIGHT:
            self.direction = self.direction.turn_right()
        elif self.last_action == Action.FORWARD:
            if not p.bump:
                self.position = self._forward(self.position, self.direction)
        elif self.last_action == Action.SHOOT:
            if self.has_arrow:
                self.has_arrow = False
        if p.scream:
            self.wumpus_dead = True

        # Stuck detection: if position didn't change, count it
        if self.position == self.last_pos:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        self.last_pos = self.position

    def _forward(self, pos: Position, d: Direction) -> Position:
        x, y = pos
        if d == Direction.NORTH:
            return (x, y + 1)
        if d == Direction.SOUTH:
            return (x, y - 1)
        if d == Direction.EAST:
            return (x + 1, y)
        return (x - 1, y)

    def _bit(self, x: int, y: int) -> int:
        """Convert (x,y) position to bitmask index."""
        idx = (y - 1) * self.grid_size + (x - 1)
        return 1 << idx

    def _update_belief(self, p: Percept) -> None:
        """Update belief memory masks based on current percept."""
        x, y = self.position
        b = self._bit(x, y)

        self.visited_mask |= b

        if p.breeze:
            self.breeze_mask |= b
            self.no_breeze_mask &= ~b
        else:
            self.no_breeze_mask |= b
            self.breeze_mask &= ~b

        # If wumpus is dead, treat stench as false for memory
        st = p.stench and (not self.wumpus_dead)
        if st:
            self.stench_mask |= b
            self.no_stench_mask &= ~b
        else:
            self.no_stench_mask |= b
            self.stench_mask &= ~b

    def _neighbors(self, pos: Position) -> List[Position]:
        """Get valid neighboring cells."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def _update_safety(self, p: Percept) -> None:
        """Update safety knowledge based on percepts - key to hybrid approach."""
        pos = self.position
        neighbors = self._neighbors(pos)
        
        # Current cell is safe (we're alive!)
        self.safe_cells.add(pos)
        self.confirmed_no_pit.add(pos)
        self.maybe_pit.discard(pos)
        self.maybe_wumpus.discard(pos)
        
        if not p.breeze:
            # No breeze = all neighbors are safe from pits
            for n in neighbors:
                self.confirmed_no_pit.add(n)
                self.maybe_pit.discard(n)
                self.safe_cells.add(n)
        else:
            # Breeze = at least one neighbor has a pit
            # Mark unvisited neighbors as maybe_pit
            for n in neighbors:
                if n not in self.confirmed_no_pit:
                    self.maybe_pit.add(n)
            
            # Pit inference: if all but one neighbor is confirmed safe, the remaining one has pit
            unconfirmed = [n for n in neighbors if n not in self.confirmed_no_pit]
            if len(unconfirmed) == 1:
                self.confirmed_pit.add(unconfirmed[0])
                self.maybe_pit.discard(unconfirmed[0])
        
        if not p.stench or self.wumpus_dead:
            # No stench (or wumpus dead) = neighbors safe from wumpus
            for n in neighbors:
                self.maybe_wumpus.discard(n)
                # Only add to safe_cells if ALSO confirmed no pit!
                # No stench says nothing about pits.
                if n in self.confirmed_no_pit and n not in self.confirmed_pit:
                    self.safe_cells.add(n)
        else:
            # Stench = wumpus in one of the neighbors
            for n in neighbors:
                if n not in self.safe_cells:
                    self.maybe_wumpus.add(n)
            
            # Wumpus inference: if all but one neighbor visited (safe), wumpus is in remaining
            unvisited_neighbors = [n for n in neighbors if n not in self.safe_cells]
            if len(unvisited_neighbors) == 1:
                # We know where wumpus is! Could shoot if facing that direction
                pass  # Q-learning will handle shooting decision

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]


class Action(Enum):
    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    GRAB = "grab"
    SHOOT = "shoot"
    CLIMB = "climb"


class Direction(Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    def turn_left(self) -> "Direction":
        order = [Direction.NORTH, Direction.WEST, Direction.SOUTH, Direction.EAST]
        return order[(order.index(self) + 1) % 4]

    def turn_right(self) -> "Direction":
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        return order[(order.index(self) + 1) % 4]


@dataclass
class Percept:
    breeze: bool = False
    stench: bool = False
    glitter: bool = False
    bump: bool = False
    scream: bool = False


@dataclass
class StepResult:
    percept: Percept
    terminated: bool
    outcome: Optional[str] = None
    has_gold: bool = False


class World:
    """Grid world with pits, a Wumpus, and gold."""

    def __init__(
        self,
        size: int = 4,
        pit_probability: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.pit_probability = pit_probability
        self.rng = random.Random(seed)

        self.start: Position = (1, 1)
        self.agent_pos: Position = self.start
        self.agent_dir: Direction = Direction.EAST

        self.pits: Set[Position] = set()
        self.wumpus_pos: Position = self.start
        self.gold_pos: Position = self.start

        self.wumpus_alive = True
        self.has_gold = False
        self.arrow_available = True
        self.terminated = False
        self.outcome: Optional[str] = None

        self._generate_world()

    def _generate_world(self) -> None:
        coords = [
            (x, y)
            for x in range(1, self.size + 1)
            for y in range(1, self.size + 1)
            if (x, y) != self.start
        ]
        for coord in coords:
            if self.rng.random() < self.pit_probability:
                self.pits.add(coord)

        free = [c for c in coords if c not in self.pits]
        self.wumpus_pos = self.rng.choice(free) if free else self.start

        free_for_gold = [c for c in free if c != self.wumpus_pos]
        self.gold_pos = self.rng.choice(free_for_gold) if free_for_gold else self.start

    def _inside(self, pos: Position) -> bool:
        x, y = pos
        return 1 <= x <= self.size and 1 <= y <= self.size

    def _forward_pos(self, pos: Position, direction: Direction) -> Position:
        x, y = pos
        if direction == Direction.NORTH:
            return (x, y + 1)
        if direction == Direction.SOUTH:
            return (x, y - 1)
        if direction == Direction.EAST:
            return (x + 1, y)
        return (x - 1, y)

    def neighbors(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self._inside(c)]

    def _compute_percept(self, bump: bool, scream: bool) -> Percept:
        breeze = any(n in self.pits for n in self.neighbors(self.agent_pos))
        stench = self.wumpus_alive and any(
            n == self.wumpus_pos for n in self.neighbors(self.agent_pos)
        )
        glitter = self.agent_pos == self.gold_pos and not self.has_gold
        return Percept(breeze=breeze, stench=stench, glitter=glitter, bump=bump, scream=scream)

    def step(self, action: Action) -> StepResult:
        if self.terminated:
            return StepResult(percept=self._compute_percept(False, False), terminated=True, outcome=self.outcome, has_gold=self.has_gold)

        bump = False
        scream = False

        if action == Action.TURN_LEFT:
            self.agent_dir = self.agent_dir.turn_left()
        elif action == Action.TURN_RIGHT:
            self.agent_dir = self.agent_dir.turn_right()
        elif action == Action.FORWARD:
            next_pos = self._forward_pos(self.agent_pos, self.agent_dir)
            if not self._inside(next_pos):
                bump = True
            else:
                self.agent_pos = next_pos
        elif action == Action.GRAB:
            if self.agent_pos == self.gold_pos and not self.has_gold:
                self.has_gold = True
        elif action == Action.SHOOT and self.arrow_available:
            self.arrow_available = False
            scream = self._shoot_arrow()
        elif action == Action.CLIMB:
            if self.agent_pos == self.start:
                self.terminated = True
                self.outcome = "escaped_with_gold" if self.has_gold else "escaped_without_gold"

        if not self.terminated and (self.agent_pos in self.pits):
            self.terminated = True
            self.outcome = "fell_in_pit"
        if not self.terminated and self.agent_pos == self.wumpus_pos and self.wumpus_alive:
            self.terminated = True
            self.outcome = "eaten_by_wumpus"

        percept = self._compute_percept(bump=bump, scream=scream)
        return StepResult(percept=percept, terminated=self.terminated, outcome=self.outcome, has_gold=self.has_gold)

    def _shoot_arrow(self) -> bool:
        dx, dy = 0, 0
        if self.agent_dir == Direction.NORTH:
            dy = 1
        elif self.agent_dir == Direction.SOUTH:
            dy = -1
        elif self.agent_dir == Direction.EAST:
            dx = 1
        else:
            dx = -1

        x, y = self.agent_pos
        while True:
            x += dx
            y += dy
            if not self._inside((x, y)):
                return False
            if (x, y) == self.wumpus_pos and self.wumpus_alive:
                self.wumpus_alive = False
                return True

    def initial_percept(self) -> Percept:
        return self._compute_percept(bump=False, scream=False)

    def render_ascii(self) -> str:
        lines: List[str] = []
        for y in reversed(range(1, self.size + 1)):
            row: List[str] = []
            for x in range(1, self.size + 1):
                cell = "."
                if (x, y) == self.agent_pos:
                    cell = "A"
                elif (x, y) == self.start:
                    cell = "S"
                if (x, y) == self.gold_pos and not self.has_gold:
                    cell = "G"
                if (x, y) == self.wumpus_pos and self.wumpus_alive:
                    cell = "W"
                if (x, y) in self.pits:
                    cell = "P"
                row.append(cell)
            lines.append(" ".join(row))
        return "\n".join(lines)

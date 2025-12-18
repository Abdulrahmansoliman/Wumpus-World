from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

from .agent_base import Agent
from .bridge import PrologBridge
from .planning import shortest_path
from .world import Action, Direction, Percept, Position


class PrologAgent(Agent):
    """Agent that asks Prolog which squares are provably safe."""

    def __init__(self, grid_size: int, bridge: PrologBridge) -> None:
        self.grid_size = grid_size
        self.bridge = bridge
        self.reset()

    def reset(self) -> None:
        self.position: Position = (1, 1)
        self.direction: Direction = Direction.EAST
        self.visited: Dict[Position, Percept] = {}
        self.safe_known: Set[Position] = {self.position}
        self.plan: Deque[Action] = deque()
        self.wumpus_dead = False
        self.has_gold = False
        self.last_action: Optional[Action] = None

    def act(self, percept: Percept) -> Action:
        self._ingest_feedback(percept)
        self._remember_percept(percept)

        if percept.glitter:
            self.plan.clear()
            self.last_action = Action.GRAB
            self.has_gold = True
            return Action.GRAB

        safe_squares = self.bridge.compute_safe_squares(
            grid_size=self.grid_size, visited=self.visited, wumpus_dead=self.wumpus_dead
        )
        self.safe_known.update(safe_squares)
        self.safe_known.update(self.visited.keys())

        if self.has_gold:
            if not self.plan:
                self._queue_path_to((1, 1))
        else:
            if not self.plan:
                target = self._choose_frontier()
                if target:
                    self._queue_path_to(target)

        if not self.plan:
            if self.position == (1, 1):
                self.last_action = Action.CLIMB
                return Action.CLIMB
            self._queue_path_to((1, 1))
            if not self.plan:
                self.last_action = Action.CLIMB
                return Action.CLIMB

        action = self.plan.popleft()
        self.last_action = action
        return action

    def _ingest_feedback(self, percept: Percept) -> None:
        if self.last_action == Action.TURN_LEFT:
            self.direction = self.direction.turn_left()
        elif self.last_action == Action.TURN_RIGHT:
            self.direction = self.direction.turn_right()
        elif self.last_action == Action.FORWARD:
            forward_pos = self._forward(self.position, self.direction)
            if not percept.bump:
                self.position = forward_pos
        if percept.scream:
            self.wumpus_dead = True

    def _remember_percept(self, percept: Percept) -> None:
        self.visited[self.position] = percept

    def _neighbors(self, pos: Position) -> list[Position]:
        """Return valid grid neighbors of a position."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def _choose_frontier(self) -> Optional[Position]:
        candidates = [p for p in self.safe_known if p not in self.visited]
        if not candidates:
            return None
        
        # Priority 1: If we detected glimmer anywhere, explore unvisited neighbors of glimmer cells first
        glimmer_neighbors: Set[Position] = set()
        for pos, percept in self.visited.items():
            if percept.glimmer:
                for neighbor in self._neighbors(pos):
                    if neighbor not in self.visited and neighbor in self.safe_known:
                        glimmer_neighbors.add(neighbor)
        
        if glimmer_neighbors:
            # Pick the closest glimmer neighbor
            return sorted(glimmer_neighbors, key=lambda c: (abs(c[0] - self.position[0]) + abs(c[1] - self.position[1])))[0]
        
        # Priority 2: Otherwise pick closest unvisited safe cell
        return sorted(candidates, key=lambda c: (abs(c[0] - self.position[0]) + abs(c[1] - self.position[1])))[0]

    def _queue_path_to(self, target: Position) -> None:
        if target not in self.safe_known:
            return
        path = shortest_path(self.position, target, passable=self.safe_known)
        if not path or len(path) < 2:
            return
        actions: Deque[Action] = deque()
        current_dir = self.direction
        current_pos = self.position
        for step in path[1:]:
            step_actions, current_dir = self._actions_to_neighbor(current_pos, step, current_dir)
            actions.extend(step_actions)
            current_pos = step
        self.plan = actions

    def _actions_to_neighbor(
        self, current: Position, nxt: Position, facing: Direction
    ) -> Tuple[Deque[Action], Direction]:
        cx, cy = current
        nx, ny = nxt
        desired = facing
        if nx > cx:
            desired = Direction.EAST
        elif nx < cx:
            desired = Direction.WEST
        elif ny > cy:
            desired = Direction.NORTH
        else:
            desired = Direction.SOUTH

        turn_actions: Deque[Action] = deque()
        if desired != facing:
            turn_actions.extend(self._turn_sequence(facing, desired))
        turn_actions.append(Action.FORWARD)
        return turn_actions, desired

    def _turn_sequence(self, start: Direction, goal: Direction) -> Deque[Action]:
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        start_idx = order.index(start)
        goal_idx = order.index(goal)
        diff = (goal_idx - start_idx) % 4
        actions: Deque[Action] = deque()
        if diff == 1:
            actions.append(Action.TURN_RIGHT)
        elif diff == 2:
            actions.append(Action.TURN_RIGHT)
            actions.append(Action.TURN_RIGHT)
        elif diff == 3:
            actions.append(Action.TURN_LEFT)
        return actions

    def _forward(self, pos: Position, direction: Direction) -> Position:
        x, y = pos
        if direction == Direction.NORTH:
            return (x, y + 1)
        if direction == Direction.SOUTH:
            return (x, y - 1)
        if direction == Direction.EAST:
            return (x + 1, y)
        return (x - 1, y)

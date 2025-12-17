from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]


def shortest_path(
    start: Position,
    goal: Position,
    passable: Set[Position],
) -> Optional[List[Position]]:
    """Breadth-first search for the shortest path over known passable squares."""
    if start == goal:
        return [start]

    queue = deque([start])
    came_from: Dict[Position, Optional[Position]] = {start: None}

    while queue:
        current = queue.popleft()
        cx, cy = current
        neighbors = [
            (cx + 1, cy),
            (cx - 1, cy),
            (cx, cy + 1),
            (cx, cy - 1),
        ]
        for nxt in neighbors:
            if nxt not in passable or nxt in came_from:
                continue
            came_from[nxt] = current
            if nxt == goal:
                return _reconstruct_path(came_from, start, goal)
            queue.append(nxt)
    return None


def _reconstruct_path(
    came_from: Dict[Position, Optional[Position]],
    start: Position,
    goal: Position,
) -> List[Position]:
    path = [goal]
    while path[-1] != start:
        prev = came_from[path[-1]]
        if prev is None:
            break
        path.append(prev)
    path.reverse()
    return path

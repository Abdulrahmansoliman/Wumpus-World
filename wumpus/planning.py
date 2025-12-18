from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

Position = Tuple[int, int]


def shortest_path(start: Position, goal: Position, passable: Set[Position]) -> Optional[List[Position]]:
    """
    BFS shortest path from start to goal through passable cells.
    Returns the path as a list of positions, or None if no path exists.
    This is the standard PLAN-ROUTE algorithm from AIMA textbook.
    """
    if start == goal:
        return [start]
    if start not in passable or goal not in passable:
        return None

    # BFS with parent tracking
    queue: deque[Position] = deque([start])
    visited: Set[Position] = {start}
    parent: dict[Position, Position] = {}

    while queue:
        current = queue.popleft()
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path
        
        # Explore neighbors (4-directional)
        x, y = current
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in passable and neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    return None  # No path found


def direct_path(start: Position, goal: Position, passable: Set[Position]) -> Optional[List[Position]]:
    """
    Simple manhattan path builder (no search variants).
    Moves horizontally then vertically as long as all squares are passable.
    Returns None if any intermediate square is not passable.
    
    DEPRECATED: Use shortest_path() for correct pathfinding with detours.
    """
    if start == goal:
        return [start]
    if start not in passable or goal not in passable:
        return None

    path = [start]
    x, y = start
    gx, gy = goal

    step_x = 1 if gx > x else -1
    while x != gx:
        x += step_x
        if (x, y) not in passable:
            return None
        path.append((x, y))

    step_y = 1 if gy > y else -1
    while y != gy:
        y += step_y
        if (x, y) not in passable:
            return None
        path.append((x, y))

    return path

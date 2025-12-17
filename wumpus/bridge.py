from __future__ import annotations

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .world import Percept

Position = Tuple[int, int]


class PrologBridge:
    """Runs SWI-Prolog queries to compute provably safe squares."""

    def __init__(self, knowledge_base_path: Path) -> None:
        self.knowledge_base_path = knowledge_base_path

    def compute_safe_squares(
        self,
        grid_size: int,
        visited: Dict[Position, Percept],
        wumpus_dead: bool,
    ) -> Set[Position]:
        state_lines: List[str] = [f"grid_size({grid_size})."]
        if wumpus_dead:
            state_lines.append("wumpus_dead.")

        for (x, y), percept in visited.items():
            if percept.breeze:
                state_lines.append(f"breeze({x},{y}).")
            else:
                state_lines.append(f"no_breeze({x},{y}).")
            if percept.stench:
                state_lines.append(f"stench({x},{y}).")
            else:
                state_lines.append(f"no_stench({x},{y}).")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pl") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("\n".join(state_lines))
            tmp.flush()

        kb = self.knowledge_base_path.as_posix()
        state_file = tmp_path.as_posix()
        goal = f"consult('{state_file}'), provably_safe(S), write(S)"
        cmd = ["swipl", "-q", "-s", kb, "-g", goal, "-t", "halt"]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return set(visited.keys())
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        if proc.returncode != 0:
            return set(visited.keys())

        output = proc.stdout.strip()
        if not output:
            return set(visited.keys())
        try:
            parsed = ast.literal_eval(output)
        except (ValueError, SyntaxError):
            return set(visited.keys())

        safe: Set[Position] = set()
        for item in parsed:
            if isinstance(item, tuple) and len(item) == 2:
                safe.add((int(item[0]), int(item[1])))
        return safe

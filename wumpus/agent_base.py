from __future__ import annotations

from typing import Optional

from .world import Action, Percept


class Agent:
    """Interface for agents that choose actions from percepts."""

    def reset(self) -> None:
        raise NotImplementedError

    def act(self, percept: Percept) -> Action:
        raise NotImplementedError


class HumanAgent(Agent):
    """Simple interactive agent for manual play."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        prompt = (
            f"Percepts: Breeze={percept.breeze}, Stench={percept.stench}, Glitter={percept.glitter}, "
            f"Bump={percept.bump}, Scream={percept.scream}\n"
            "Choose action [f=forward, l=turn left, r=turn right, g=grab, s=shoot, c=climb]: "
        )
        key = input(prompt).strip().lower()[:1]
        mapping = {
            "f": Action.FORWARD,
            "l": Action.TURN_LEFT,
            "r": Action.TURN_RIGHT,
            "g": Action.GRAB,
            "s": Action.SHOOT,
            "c": Action.CLIMB,
        }
        return mapping.get(key, Action.CLIMB)

"""Define o objeto de Task."""

from pydantic.dataclasses import dataclass


@dataclass
class Task:
    """Define a task que será executada por um modelo ou agente."""

    backstory: str
    short_backstory: str
    role: str
    goal: str

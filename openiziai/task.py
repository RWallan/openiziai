from pydantic.dataclasses import dataclass


@dataclass
class Task:
    backstory: str
    short_backstory: str
    role: str
    goal: str

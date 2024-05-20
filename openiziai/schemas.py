import sys
from datetime import datetime
from typing import Any, Protocol

from pydantic import Field
from pydantic.dataclasses import dataclass

from openiziai.task import Task

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class DataDict(TypedDict):
    data: Any


class Pipeline(Protocol):
    def run(self): ...


@dataclass
class GPTModel:
    name: str
    task: Task
    base_model: str
    created_at: datetime = Field(init=False, default_factory=datetime.now)

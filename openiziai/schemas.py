"""Define os schemas utilizados."""

import sys
from datetime import datetime
from typing import Any, Literal, Optional, Protocol

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from openiziai.task import Task

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class DataDict(TypedDict):
    """Schema dos dados que serão utilizados para criar os treinos."""

    data: Any


class Pipeline(Protocol):
    """Define o método que deve ser implementado em um pipeline."""

    def run(self): ...


@dataclass
class GPTModel:
    """Entidade do modelo GPT treinado."""

    name: str
    task: Task
    base_model: Optional[str] = None
    created_at: datetime = Field(init=False, default_factory=datetime.now)


class Message(BaseModel):
    """Representa a interação de um Agente."""
    role: Literal['assistant', 'user'] = 'user'
    content: str

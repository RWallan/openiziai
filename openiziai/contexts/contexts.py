from pathlib import Path
from uuid import uuid4

from pydantic import Field, PositiveInt
from pydantic.dataclasses import dataclass

from openiziai.schemas import GPTModel, Message


@dataclass
class Context:
    agent_model: GPTModel
    history: list[Message]
    max_context_length: PositiveInt
    context_store: Path
    id: str = Field(init=False, default=f'context-{uuid4()}')

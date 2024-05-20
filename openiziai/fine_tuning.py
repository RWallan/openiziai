from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from openiziai.task import Task


class JobStatus(Enum):
    VALIDATING = 'validating_files'
    QUEUED = 'queued'
    COMPLETED = 'succeeded'
    RUNNING = 'running'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class FineTuning(BaseModel):
    client: OpenAI = Field(default=None, description='Client da OpenAI.')
    train_file: Path | str
    task: Task = Field(default=None)
    model: str = Field(default='gpt-3.5-turbo')
    _file_id: str
    _job_id: str = PrivateAttr(default=None)
    _job_status: JobStatus = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    @field_validator('train_file')
    @classmethod
    def size_must_be_minor_512(cls, v: Path | str) -> Path | str:
        MAX_FILE_SIZE = 512000000
        _v = Path(v) if isinstance(v, str) else v
        size = _v.stat().st_size
        if size > MAX_FILE_SIZE:
            raise ValueError(
                'max_file_size_error',
                f'Arquivo de treino deve conter no máximo 512MB. {size / 100000}MB',  # noqa
            )

        return v

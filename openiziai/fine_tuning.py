"""Disponibiliza o fine tuning do modelo."""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from openiziai.schemas import GPTModel
from openiziai.task import Task


class JobStatus(Enum):
    """Status do job de fine tuning."""

    VALIDATING = 'validating_files'
    QUEUED = 'queued'
    COMPLETED = 'succeeded'
    RUNNING = 'running'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class FineTuning(BaseModel):
    """Classe que gerencia os métodos necessários para construir um modelo de
    fine tuning.
    """

    client: OpenAI = Field(description='Client da OpenAI.')
    train_file: Path | str = Field(
        description='Caminho até o arquivo de treino.'
    )
    task: Task = Field(
        description='Task em que o modelo deve se especializar.'
    )
    base_model: str = Field(
        default='gpt-3.5-turbo', description='Modelo base que será refinado.'
    )
    _file_id: str = PrivateAttr(default=None)
    _job_id: str = PrivateAttr(default=None)
    _job_status: JobStatus = PrivateAttr(default=None)
    _model: GPTModel = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Cria um novo fine tuning analisando e validando o input.

        Args:
            client (OpenAI): Client da OpenAI.
            train_file (Path|str): Caminho até o arquivo de treino.
            task (Task): Task em que o modelo deve se especializar.
            base_model (str): Modelo base que será refinado.
        """
        super().__init__(**data)

    @field_validator('train_file')
    @classmethod
    def size_must_be_minor_512(cls, v: Path | str) -> Path | str:
        """Valida se o arquivo indicado possui tamanho menor que 512MB."""
        MAX_FILE_SIZE = 512000000
        _v = Path(v) if isinstance(v, str) else v
        size = _v.stat().st_size
        if size > MAX_FILE_SIZE:
            raise ValueError(
                'max_file_size_error',
                f'Arquivo de treino deve conter no máximo 512MB. {size / 100000}MB',  # noqa
            )

        return v

    def upload_file_to_openai(self) -> 'FineTuning':
        """Envia o arquivo de treino para a plataforma da OpenAI."""
        self._file_id = self.client.files.create(
            file=open(self.train_file, 'rb'), purpose='fine-tune'
        ).id

        return self

    @property
    def file_id(self) -> Optional[str]:
        """ID do arquivo de treino na plataforma OpenAI."""
        if not self._file_id:
            print('Nenhum dado foi enviado ainda.')
            return None
        return self._file_id

    def start(self) -> Optional['FineTuning']:
        """Inicia o job de fine tuning."""
        file_id = self.file_id
        if not file_id:
            return None

        job = self.client.fine_tuning.jobs.create(
            training_file=file_id, model=self.base_model
        )
        self._job_id = job.id
        print(
            f'Fine tuning started: {self._job_id}.',
            'Veja o status com `.status`.',
        )
        return self

    @property
    def job_id(self) -> Optional[str]:
        """ID do job de fine tuning na plataforma OpenAI."""
        if not getattr(self, '_job_id'):
            print('Nenhum fine tuning foi iniciado.')
            return None
        return self._job_id

    @property
    def status(self) -> Optional[str]:
        """Status do job de fine tuning."""
        job_id = self._job_id
        if not job_id:
            print('Nenhum fine tuning foi iniciado.')
            return None

        if self._job_status != JobStatus.COMPLETED:
            _job_status = self.client.fine_tuning.jobs.retrieve(job_id).status
            self._job_status = JobStatus(_job_status)

        return self._job_status.name

    @property
    def model(self) -> Optional[GPTModel]:
        """Modelo criado do fine tuning."""
        if self._model:
            return self._model

        job_id = self.job_id
        if not job_id:
            print('Nenhum fine tuning foi iniciado.')
            return None

        model_name = self.client.fine_tuning.jobs.retrieve(
            job_id
        ).fine_tuned_model
        if not model_name:
            print(f'Modelo não disponível. Status: {self.status}')
            return None

        self._model = GPTModel(
            name=model_name, base_model=self.base_model, task=self.task
        )

        return self._model

    def __repr__(self) -> str:
        return (
            'FineTuning('
            f'client={self.client}, '
            f'train_file={self.train_file}, '
            f'task={self.task}, '
            f'base_model="{self.base_model}", '
            f'model={self.model}'
            ')'
        )

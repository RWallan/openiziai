"""Módulo que define o principal Agente."""

from typing import Any, Optional, Self

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from openiziai.schemas import GPTModel
from openiziai.task import Task


@dataclass
class PromptResponse:
    """Informações do prompt construído."""

    id: str
    prompt: str
    response: str | None
    temperature: float
    total_tokens: int | None
    fine_tuned_model: str


class Agent(BaseModel):
    """Classe que constrói o Agente especializado utilizando um modelo GPT."""

    client: OpenAI = Field(description='Client da OpenAI.')
    model: Optional[GPTModel] = Field(
        default=None, description='A entidade do modelo fine tuned.'
    )
    fine_tuned_model: Optional[str] = Field(
        default=None,
        description='Nome do modelo que será utilizado no Agente.',
    )
    task: Optional[Task] = Field(
        default=None, description='Task a ser executada pelo Agente.'
    )
    _template: str
    _full_context: list[dict[str, Any]]
    _context: list[dict[str, Any]]
    _fine_tuned_model: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Cria um novo agente analisando e validando o input.

        Parameters:
            client (OpenAI): Client da OpenAI.
            model (GPTModel): A entidade do modelo fine tuned.
            fine_tuned_model (str): Modelo que será utilizado no Agente.
            task (Task): Task a ser executada pelo Agente.

        Examples:
            >>> from openai import OpenAI
            >>>
            >>>
            >>> client = OpenAI()
            >>> task = Task(
            ...    backstory='backstory',
            ...    short_backstory='short_backstory',
            ...    role='role',
            ...    goal='goal',
            ... )
            >>>
            >>> my_model = GPTModel(
            ...    name='my_model',
            ...    task=task,
            ...    base_model='gpt-3.5-turbo',
            ... )
            >>>
            >>> Agent(client=client, model=my_model)
            >>> # OU
            >>> Agent(client=client, fine_tuned_model='my_model', task=task)
        """
        super().__init__(**data)
        self._template = self._build_template()
        self._fine_tuned_model = (
            self.model.name if self.model else self.fine_tuned_model
        )  # pyright: ignore

    @model_validator(mode='after')
    def validate_model_info(self) -> Self:
        """Valida se o `model` ou se `fine_tuned_model` e `task` foram
        declarados.
        """
        if not self.model:
            if not self.fine_tuned_model and not self.task:
                raise ValueError(
                    'Precisa de um `model` ou um `fine_tuned_model` e `task`.'
                )

        return self

    def _build_template(self) -> str:
        backstory = (
            self.model.task.short_backstory
            if self.model
            else self.task.short_backstory  # pyright: ignore
        )
        role = self.model.task.role if self.model else self.task.role  # pyright: ignore
        goal = self.model.task.goal if self.model else self.task.goal  # pyright: ignore

        template = f"""{backstory}.
        {{'your_role': {role}, 'your_goal': {goal}}}
        """

        return template

    def prompt(
        self, prompt: str, temperature: float = 0.5, max_tokens: int = 1000
    ) -> PromptResponse:
        """Executa o prompt para o modelo de fine tuning.

        Args:
            prompt (str): Prompt.
            temperature (float): Temperatura que controla a criatividade ao
                construir a resposta
            max_tokens (int): Máximo de tokens que deve conter nas respostas.
                Valores maiores trarão respostas maiores porém terá maior
                custo.

        Returns:
            PromptResponse: Informações do prompt construído.
        """
        messages = [
            {
                'role': 'system',
                'content': self._template,
            },
            {
                'role': 'user',
                'content': prompt,
            },
        ]

        result = self.client.chat.completions.create(
            messages=messages,  # pyright: ignore
            model=self._fine_tuned_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = PromptResponse(
            id=result.id,
            prompt=prompt,
            response=result.choices[0].message.content,
            temperature=temperature,
            total_tokens=result.usage.total_tokens if result.usage else None,
            fine_tuned_model=self._fine_tuned_model,
        )

        return response

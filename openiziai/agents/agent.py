from typing import Any, Optional, Self

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from openiziai.schemas import GPTModel
from openiziai.task import Task


@dataclass
class PromptResponse:
    id: str
    prompt: str
    response: str | None
    temperature: float
    tokens: int | None
    fine_tuned_model: str


class Agent(BaseModel):
    client: OpenAI
    model: Optional[GPTModel] = None
    fine_tuned_model: Optional[str] = None
    task: Optional[Task] = None
    max_context_length: Optional[int] = None
    _template: str
    _full_context: list[dict[str, Any]]
    _context: list[dict[str, Any]]
    _fine_tuned_model: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._template = self._build_template()
        self._fine_tuned_model = (
            self.model.name if self.model else self.fine_tuned_model
        )  # pyright: ignore

    @model_validator(mode='after')
    def validate_model_info(self) -> Self:
        if not self.model:
            if not self.fine_tuned_model and not self.task:
                raise ValueError(
                    'Precisa de um `model` ou um `model_name` e `task`.'
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
            max_tokens=max_tokens
        )

        response = PromptResponse(
            id=result.id,
            prompt=prompt,
            response=result.choices[0].message.content,
            temperature=temperature,
            tokens = result.usage.total_tokens if result.usage else None,
            fine_tuned_model=self._fine_tuned_model,
        )

        return response

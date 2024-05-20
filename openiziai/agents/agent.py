from typing import Any, Optional, Self

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, model_validator

from openiziai.schemas import GPTModel
from openiziai.task import Task


class Agent(BaseModel):
    client: OpenAI
    model: Optional[GPTModel] = None
    fine_tuned_model: Optional[str] = None
    task: Optional[Task] = None
    max_context_length: Optional[int] = None
    _template: str
    _full_context: list[dict[str, Any]]
    _context: list[dict[str, Any]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._template = self._build_template()
        self.fine_tuned_model = (
            self.model.name if self.model else self.fine_tuned_model
        )

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

        template = f"""This is your backstory:
        ```backstory: {backstory}```
        Your role is: {role}.
        Your goal is: {goal}.
        Answer with the `backstory` native language.
        """

        return template

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, PositiveInt, PrivateAttr

from openiziai.schemas import GPTModel, Message

from .contexts import Context

Exporter = Callable[[list[Message]], Any]


class ContextHandler(BaseModel):
    """Implementa os métodos necessários para criar o contexto do Agente."""
    max_context_length: PositiveInt
    context_store: Path
    agent_model: Optional[GPTModel] = None
    _history: list[Message] = PrivateAttr(default=[])
    _context: Context = PrivateAttr(default=None)

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self._context = Context(
            agent_model=self.agent_model,  # pyright: ignore
            history=[],
            max_context_length=self.max_context_length,
            context_store=self.context_store,
        )

    def create(self, overwrite: bool = False):
        """Cria ou limpa o contexto de um agente."""
        if len(self._history) > 0 and not overwrite:
            raise Exception(
                (
                    'Já existe um histórico para esse contexto.'
                    'Caso queira sobescrever defina `overwrite=True`.'
                )
            )

        self._history = []

    @property
    def history(self) -> list[Message]:
        """Retorna o histórico de interações do Agente."""
        return self._history[-self.max_context_length :]

    def add(self, message: Message):
        """Adiciona uma interação no contexto."""
        self._history.append(message)
        self._context.history = self._history

    def save(self):
        """Salva o contexto como um pickle."""
        self._context.history = self._history
        self.context_store.mkdir(parents=True, exist_ok=True)
        with open(
            self.context_store / f'context_{self._context.id}.pkl', 'wb'
        ) as file:
            pickle.dump(asdict(self._context), file, protocol=-1)

    def export_history(self, func: Exporter):
        """Executa uma função exporter para salvar todo o histório de
        interações do Agente.
        """
        func(self._history)

    @property
    def context(self) -> Context:
        """Retorna o contexto do Agente."""
        return self._context

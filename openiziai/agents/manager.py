from pathlib import Path
from typing import Optional

from pydantic import BaseModel, PositiveInt, PrivateAttr

from openiziai.contexts import Context, ContextHandler, Exporter
from openiziai.schemas import Message

from .agent import Agent


class AgentManager(BaseModel):
    """Gerenciador de contexto capaz de gerenciar o contexto de interações do
    agente.
    """

    agent: Agent
    pre_context: Optional[Context] = None
    context_store: Path = Path().cwd() / 'data/contexts'
    max_context_length: PositiveInt = 10
    context_exporter: Optional[Exporter] = None
    _ctx_handler: ContextHandler = PrivateAttr(default=None)

    def __enter__(self) -> 'AgentManager':
        if not self.pre_context:
            self._ctx_handler = ContextHandler(
                agent_model=self.agent.model,
                max_context_length=self.max_context_length,
                context_store=self.context_store,
            )
            self._ctx_handler.create()
        else:
            self._ctx_handler = ContextHandler(
                max_context_length=self.pre_context.max_context_length,
                context_store=self.pre_context.context_store,
                agent_model=self.pre_context.agent_model,
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self._ctx_handler.save()
        if self.context_exporter:
            self._ctx_handler.export_history(self.context_exporter)

    def prompt(
        self,
        prompt: str,
        *,
        temperature: float = 0.5,
        max_tokens: int = 1000,
    ):
        """Executa o prompt para o Agente.

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

        self._ctx_handler.add(Message(content=prompt))
        response = self.agent.prompt(
            history=self._ctx_handler.history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.response:
            self._ctx_handler.add(
                Message(role='assistant', content=response.response)
            )

        return response

    @property
    def context(self) -> Context:
        """O contexto completo do Agente."""
        return self._ctx_handler.context

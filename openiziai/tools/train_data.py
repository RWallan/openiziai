from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)
from openiziai.schemas import DataDict
from openiziai.task import Task


class TrainDataTool(BaseModel):
    """Cria dados preparados para fine tuning.

    Cria os dados para construir um fine tuning de um modelo LLM centrado
    em uma task contendo: backstory, role e o goal.

    :param client: Client da OpenAI.
    :param data: Dados utilizados para construir o dado e treino.
    :param task: Descrição da task que o modelo deverá executar.
    :param model: Modelo GPT usado para criar os dados para treino. Padrão gpt-3.5-turbo-125.
    :type client: OpenAI
    :type data: DataDict
    :type task: Task
    :type model: str
    """  # noqa

    client: OpenAI = Field(default=None, description='Client da OpenAI.')
    data: DataDict = Field(
        default=None,
        description='Dados utilizados para construir o dado de treino.',
    )
    task: Task = Field(
        default=None,
        description='Descrição da task que o modelo deverá executar.',
    )
    model: str = Field(
        default='gpt-3.5-turbo-125',
        description='Modelo GPT que criará o dado de fine tuning.',
    )

    _n_examples: int = PrivateAttr(default=None)
    _n_batch: int = PrivateAttr(default=None)
    _root: Path = PrivateAttr(default_factory=Path.cwd)
    _train_data_dir: Path
    _template: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Cria um novo modelo analisando e validando o input.

        Raises [ValidationError][pydantic_core.ValidationError]
        se os inputs não podem ser validados para forma um modelo válido.
        """
        super().__init__(**data)
        self._template = """You are generating data which will be used to train a machine learning model.
        To describe how the model will be trained, you will be given a high-level description in a dict with:
        ```{{'backstory': $backstory_goes_here, 'role': $role_goes_here, 'goal': $goal_goes_here, 'data': $data_goes_here}}
        From that, you will generate data samples, each with a prompt/response pair. You must return the sample in jsonl format with the exact informations:
        ```{{'prompt': $prompt_goes_here, 'response': $response_goes_here}}```
        Only one prompt/response pair should be generated by turn. For each turn, make the example slightly more complex than the last, while ensuring diversity.
        Here is the high-level description to generate the samples:
        {description}
        """  # noqa

        self._train_data_dir = self._create_dir_if_not_exist(
            self._root / 'data' / 'train'
        )

    @property
    def train_data_dir(self) -> str:
        return str(self._train_data_dir)

    @staticmethod
    def _create_dir_if_not_exist(dir: Path) -> Path:
        dir.mkdir(parents=True, exist_ok=True)

        return dir

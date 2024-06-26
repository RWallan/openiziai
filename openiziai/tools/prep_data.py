import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openiziai.schemas import DataDict, Pipeline


@lru_cache()
def prep_data(pipeline: Pipeline, persist: bool = True) -> DataDict:
    """Aplica um pipeline e retorna os dados no formato DataDict.

    Args:
        pipeline (Pipeline): Pipeline a ser executado. Deve contem o método
            run() implementado.
        persist (bool): Se o dado criado deve ser persistido em um pickle.
            Padrão True.

    Returns:
        DataDict: Dados em um dicionário que contém a chave `data`.
    """
    _pipelined_data = pipeline.run()

    if persist:
        path = Path(__file__).cwd() / 'data' / 'pipelined_data'
        path.mkdir(parents=True, exist_ok=True)
        file = path / f'pipelined_data_{datetime.now().strftime("%Y%m%d")}.pkl'

        with open(file, 'wb') as f:
            pickle.dump(_pipelined_data, f, protocol=-1)

    return {'data': _pipelined_data}

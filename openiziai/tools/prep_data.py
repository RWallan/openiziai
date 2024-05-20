import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openiziai.schemas import DataDict, Pipeline


@lru_cache()
def prep_data(pipeline: Pipeline, persist: bool = True) -> DataDict:
    _pipelined_data = pipeline.run()

    if persist:
        path = Path(__file__).cwd() / 'data' / 'pipelined_data'
        path.mkdir(parents=True, exist_ok=True)
        file = path / f'pipelined_data_{datetime.now().strftime('%Y%m%d')}.pkl'

        with open(file, 'wb') as f:
            pickle.dump(_pipelined_data, f, protocol=-1)

    return {'data': _pipelined_data}

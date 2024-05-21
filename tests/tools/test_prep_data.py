import shutil
from pathlib import Path

from openiziai.tools import prep_data


def test_prep_data_with_persistant_data():
    class P:
        def __init__(self) -> None:
            self._data = ['TeStE', 'tes te', 'teste ']

        def run(self):
            self.data = [i.lower().replace(r'\s', '') for i in self._data]

            return self.data

    p = P()
    path = Path().cwd() / 'data' / 'pipelined_data'
    response = prep_data(p, persist=True)

    assert 'data' in response
    assert path.exists()
    shutil.rmtree(path)

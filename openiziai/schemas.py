import sys
from typing import Any, Protocol

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class DataDict(TypedDict):
    data: Any


class Pipeline(Protocol):
    def run(self): ...


import sys
from typing import Any

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class DataDict(TypedDict):
    data: Any

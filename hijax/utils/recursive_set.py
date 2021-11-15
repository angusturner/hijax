from typing import Union, Dict
from haiku._src.data_structures import FlatMap

T = Union[Dict, FlatMap]


def recursive_set(src: T, sink: T):
    """
    Sets all leaf values in `sink` to the corresponding value from `src`.
    """
    for k, v in src.items():
        if isinstance(v, Dict) or isinstance(v, FlatMap):
            recursive_set(v, sink[k])
        else:
            sink[k] = v

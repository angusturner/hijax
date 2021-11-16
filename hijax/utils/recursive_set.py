from jaxlib.xla_extension import DeviceArray
from typing import Any

from haiku._src.data_structures import FlatMap


def recursive_set(src: Any, sink: Any):
    """
    Sets all leaf values in `sink` to the corresponding value from `src`.
    """
    if isinstance(src, dict) or isinstance(src, FlatMap) or isinstance(src, list):
        iter_ = enumerate(src) if isinstance(src, list) else src.items()
        for k, v in iter_:
            if isinstance(v, dict) or isinstance(v, FlatMap) or isinstance(v, list):
                recursive_set(v, sink[k])
            elif isinstance(v, DeviceArray) or isinstance(sink[k], DeviceArray):
                sink[k].at[:].set(v)
            else:
                sink[k] = v
    else:
        raise TypeError(f"src must be a dict or a list, not {type(src)}")

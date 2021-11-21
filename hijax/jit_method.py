import jax
from collections import Callable
from typing import Union, Iterable


def jit_method(*args, static_argnums: Union[int, Iterable[int], None] = None, **kwargs) -> Callable:
    """
    `jax.jit` decorator for bound class methods, which always marks the first positional argument (`self`) as static.
    The static indices are remapped, so that static_argnums=0 refers to the first argument after `self`. Remaining
    kwargs are passed to `jax.jit`.

    For example,
    ```
    @hx.jit_method()
    def some_fn(self, x):
        ...
    ```
    is shorthand for,
    ```
    @functools.partial(jax.jit, static_argnums=(0,))
    def some_fn(self, x):
        ...
    ```
    OR
    ```
    @hx.jit_method(static_argnums=3)
    def some_fn(self, x):
        ...
    ```
    is shorthand for,
    ```
    @functools.partial(jax.jit, static_argnums=(0, 4))
    def some_fn(self, x):
        ...
    ```

    :param args:
    :param static_argnums:
    :param kwargs:
    :return:
    """
    if static_argnums is None:
        static_argnums = []
    elif isinstance(static_argnums, int):
        static_argnums = [static_argnums]
    static_argnums = [0] + [i + 1 for i in static_argnums]

    def wrapper(fun):
        return jax.jit(fun, *args, static_argnums=static_argnums, **kwargs)

    return wrapper

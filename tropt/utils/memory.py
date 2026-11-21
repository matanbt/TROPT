# `find_executable_batch_size` is copied from Hugging Face Accelerate v1.16.0
# (https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/memory.py),
# Copyright 2022 The HuggingFace Team, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).

import functools
import inspect
from typing import Callable, Optional

from accelerate.utils.memory import clear_device_cache, should_reduce_batch_size


def find_executable_batch_size(
    function: Optional[Callable] = None,
    starting_batch_size: int = 128,
    reduce_batch_size_fn: Optional[Callable] = None,
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is multiplied by 0.9 and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory
        reduce_batch_size_fn (`callable`, *optional*):
            A function to determine the new batch size after an out-of-memory error. If not
            provided, the batch size is multiplied by 0.9 on each failure. The function takes
            no arguments and should return the new (reduced) batch size as an `int`.

    Example:

    ```python
    >>> from tropt.utils.memory import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            reduce_batch_size_fn=reduce_batch_size_fn,
        )

    batch_size = starting_batch_size
    if reduce_batch_size_fn is None:

        def reduce_batch_size_fn():
            nonlocal batch_size
            batch_size = int(batch_size * 0.9)
            return batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        # clear_device_cache(garbage_collection=True)  # [tropt] skipped: cleared only after an OOM (below)
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    clear_device_cache(garbage_collection=True)
                    batch_size = reduce_batch_size_fn()
                else:
                    raise

    return decorator

import functools

from accelerate.utils.memory import clear_device_cache, should_reduce_batch_size


def find_executable_batch_size(function=None, starting_batch_size: int = 128):
    """Drop-in for accelerate's decorator that only clears the device cache on OOM.

    (Accelerate's version runs `gc.collect()` + `empty_cache()` on *every* call, which dominates fast loops.)
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        batch_size = starting_batch_size
        while batch_size > 0:
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if not should_reduce_batch_size(e):
                    raise
                clear_device_cache(garbage_collection=True)
                batch_size = int(batch_size * 0.9)
        raise RuntimeError("No executable batch size found, reached zero.")

    return decorator

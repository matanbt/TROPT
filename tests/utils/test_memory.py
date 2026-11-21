"""`tropt.utils.memory.find_executable_batch_size` (accelerate's, minus the cache clear before every call): on OOM it
shrinks the batch size (x0.9) and retries, clearing the device cache only then; other errors propagate at once;
passing the batch size by hand raises a TypeError."""

import pytest
import torch

import tropt.utils.memory as memory


def test_find_executable_batch_size(monkeypatch):
    clears, tried = [], []
    monkeypatch.setattr(memory, "clear_device_cache", lambda **kwargs: clears.append(1))

    def body(batch_size, tag):
        tried.append(batch_size)
        if batch_size > 37:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if tag == "fail":
            raise ValueError(tag)
        return batch_size, tag

    assert memory.find_executable_batch_size(starting_batch_size=128)(body)(tag="x") == (36, "x")
    assert tried == [128, 115, 103, 92, 82, 73, 65, 58, 52, 46, 41, 36]
    assert len(clears) == 11  # one per OOM, none otherwise

    clears.clear()
    tried.clear()
    assert memory.find_executable_batch_size(body, starting_batch_size=8)(tag="y") == (8, "y")
    with pytest.raises(ValueError):
        memory.find_executable_batch_size(body, starting_batch_size=16)(tag="fail")
    with pytest.raises(TypeError, match="Batch size was passed"):
        memory.find_executable_batch_size(body, starting_batch_size=16)(16, "z")
    assert tried == [8, 16] and not clears

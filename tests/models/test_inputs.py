import pytest
import torch

from tropt.models.inputs import TargetsDictPlus, TextInputsManager


# --- TargetsDictPlus Tests ---

def test_targets_dict_plus_init():
    targets = {"key1": [1, 2, 3]}
    tdp = TargetsDictPlus(targets, n_messages=3)
    assert tdp.n_messages == 3
    assert tdp["key1"] == [1, 2, 3]

def test_targets_dict_plus_infer_n_messages():
    targets = {"key1": [1, 2, 3]}
    tdp = TargetsDictPlus(targets)
    assert tdp.n_messages == 3

def test_targets_dict_plus_validation():
    targets = {"key1": [1, 2]}
    with pytest.raises(AssertionError):
        TargetsDictPlus(targets, n_messages=3)

def test_targets_dict_plus_setitem():
    targets = {"key1": [1, 2, 3]}
    tdp = TargetsDictPlus(targets)
    with pytest.raises(AssertionError):
        tdp["key2"] = [1, 2] # Wrong length
    
    tdp["key2"] = [4, 5, 6]
    assert tdp["key2"] == [4, 5, 6]

def test_to_device():
    t1 = torch.tensor([1, 2, 3])
    targets = {"t1": t1}
    tdp = TargetsDictPlus(targets)
    
    device = torch.device("cpu")
    tdp.to_device(device)
    assert tdp["t1"].device.type == "cpu"

def test_expanded_with_candidates_tensor():
    # Input: (n_messages, ...)
    # Output: (n_messages, n_candidates, ...)
    n_messages = 2
    n_candidates = 3
    feature_dim = 4
    
    targets = {"t1": torch.randn(n_messages, feature_dim)}
    tdp = TargetsDictPlus(targets, n_messages=n_messages)
    
    expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates)
    
    assert expanded["t1"].shape == (n_messages, n_candidates, feature_dim)
    # Check content: all candidates for a message should be identical
    assert torch.allclose(expanded["t1"][0, 0], targets["t1"][0])
    assert torch.allclose(expanded["t1"][0, 1], targets["t1"][0])
    assert torch.allclose(expanded["t1"][1, 0], targets["t1"][1])

def test_expanded_with_candidates_list_strings():
    # Input: List of strings (length n_messages)
    # Output: List of lists of strings (length n_messages, each list length n_candidates)
    targets = {"s1": ["msg1", "msg2"]}
    tdp = TargetsDictPlus(targets)
    n_candidates = 3
    
    expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates)
    
    assert len(expanded["s1"]) == 2
    assert len(expanded["s1"][0]) == 3
    assert expanded["s1"][0][0] == "msg1"
    assert expanded["s1"][1][0] == "msg2"

# ---------------------------------------------------------------------------
# --- TextInputsManager Tests ---

def test_text_inputs_manager_init():
    texts = ["Hello {{OPTIMIZED_TRIGGER}} World", "Start {{OPTIMIZED_TRIGGER}}"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{OPTIMIZED_TRIGGER}}")
    
    assert tim.n_messages == 2
    assert tim.before_texts == ["Hello ", "Start "]
    assert tim.after_texts == [" World", ""]

def test_text_inputs_manager_get_triggered_inputs():
    texts = ["A{{OPTIMIZED_TRIGGER}}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{OPTIMIZED_TRIGGER}}")
    
    triggers = ["1", "2"]
    res = tim.get_triggered_inputs(triggers)
    
    # inputs_texts should be list of lists: inputs[message_idx][candidate_idx]
    assert len(res["inputs_texts"]) == 1
    assert len(res["inputs_texts"][0]) == 2
    assert res["inputs_texts"][0][0] == "A1B"
    assert res["inputs_texts"][0][1] == "A2B"

def test_text_inputs_manager_get_triggered_inputs_chosen_message():
    texts = ["A{{T}}B", "C{{T}}D"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")
    
    triggers = ["1", "2"]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=1)
    
    # When chosen_message_idx is set: inputs is a list of strings (candidates for that message)
    assert len(res["inputs_texts"]) == 2
    assert res["inputs_texts"][0] == "C1D"
    assert res["inputs_texts"][1] == "C2D"


# ==============================================================================
# Edge Cases and Robustness Tests
# ==============================================================================

def test_targets_dict_plus_empty_dict():
    """Test TargetsDictPlus with empty targets."""
    targets = {}
    tdp = TargetsDictPlus(targets, n_messages=5)

    assert tdp.n_messages == 5
    assert len(tdp) == 0


def test_targets_dict_plus_mixed_types():
    """Test TargetsDictPlus with mixed tensor and list types."""
    targets = {
        "tensor_key": torch.randn(3, 10),
        "list_key": ["a", "b", "c"],
        "int_list_key": [1, 2, 3],
    }
    tdp = TargetsDictPlus(targets)

    assert tdp.n_messages == 3
    assert isinstance(tdp["tensor_key"], torch.Tensor)
    assert isinstance(tdp["list_key"], list)


def test_targets_dict_plus_nested_tensors():
    """Test TargetsDictPlus with higher-dimensional tensors."""
    targets = {
        "embeddings": torch.randn(4, 128),  # (n_messages, d_model)
        "logits": torch.randn(4, 50, 1000),  # (n_messages, seq_len, vocab_size)
    }
    tdp = TargetsDictPlus(targets)

    assert tdp.n_messages == 4

    # Test expansion
    expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates=3)
    assert expanded["embeddings"].shape == (4, 3, 128)
    assert expanded["logits"].shape == (4, 3, 50, 1000)


def test_targets_dict_plus_validation_mismatch():
    """Test that TargetsDictPlus validates lengths across all keys."""
    targets = {
        "key1": [1, 2, 3],
        "key2": torch.randn(3, 10),
        "key3": ["a", "b", "c"],
    }
    tdp = TargetsDictPlus(targets)
    assert tdp.n_messages == 3

    # Adding mismatched length should fail
    with pytest.raises(AssertionError):
        tdp["key4"] = [1, 2]  # Only length 2


def test_text_inputs_manager_no_placeholder():
    """Test TextInputsManager when no placeholder is found."""
    texts = ["Hello World", "No trigger here"]

    # Should raise an error or handle gracefully
    with pytest.raises(AssertionError):
        TextInputsManager(texts, optimized_trigger_placeholder="{{OPTIMIZED_TRIGGER}}")


def test_text_inputs_manager_multiple_placeholders():
    """Test TextInputsManager with multiple placeholders in one text."""
    texts = ["Start {{T}} middle {{T}} end"]

    # This should either work (replacing all) or raise an error
    # Checking actual behavior:
    with pytest.raises(AssertionError):
        TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")


def test_text_inputs_manager_empty_trigger():
    """Test TextInputsManager with empty trigger string."""
    texts = ["A{{T}}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    triggers = [""]  # Empty trigger
    res = tim.get_triggered_inputs(triggers)

    assert res["inputs_texts"][0][0] == "AB"


def test_text_inputs_manager_long_trigger():
    """Test TextInputsManager with very long trigger."""
    texts = ["Start {{T}} end"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    long_trigger = "x" * 1000
    triggers = [long_trigger]
    res = tim.get_triggered_inputs(triggers)

    assert res["inputs_texts"][0][0] == f"Start {long_trigger} end"


def test_text_inputs_manager_special_characters():
    """Test TextInputsManager with special characters in trigger."""
    texts = ["Before {{T}} after"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    special_trigger = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    triggers = [special_trigger]
    res = tim.get_triggered_inputs(triggers)

    assert res["inputs_texts"][0][0] == f"Before {special_trigger} after"


def test_text_inputs_manager_unicode():
    """Test TextInputsManager with unicode characters."""
    texts = ["Text: {{T}}"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    unicode_trigger = "Hello 世界 🌍"
    triggers = [unicode_trigger]
    res = tim.get_triggered_inputs(triggers)

    assert res["inputs_texts"][0][0] == f"Text: {unicode_trigger}"


def test_text_inputs_manager_many_messages():
    """Test TextInputsManager with many messages."""
    n_messages = 100
    texts = [f"Message {i}: {{T}}" for i in range(n_messages)]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    assert tim.n_messages == n_messages

    triggers = ["test"]
    res = tim.get_triggered_inputs(triggers)

    assert len(res["inputs_texts"]) == n_messages
    assert res["inputs_texts"][50][0] == "Message 50: test"


def test_text_inputs_manager_many_candidates():
    """Test TextInputsManager with many candidate triggers."""
    texts = ["A{{T}}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    n_candidates = 1000
    triggers = [str(i) for i in range(n_candidates)]
    res = tim.get_triggered_inputs(triggers)

    assert len(res["inputs_texts"][0]) == n_candidates
    assert res["inputs_texts"][0][0] == "A0B"
    assert res["inputs_texts"][0][999] == "A999B"


def test_targets_dict_plus_device_movement():
    """Test that to_device correctly moves all tensors."""
    targets = {
        "t1": torch.randn(2, 10),
        "t2": torch.randn(2, 5, 3),
        "list": ["a", "b"],  # Should be unaffected
    }
    tdp = TargetsDictPlus(targets)

    # Move to CPU (should already be there, but test the mechanism)
    tdp.to_device(torch.device("cpu"))

    assert tdp["t1"].device.type == "cpu"
    assert tdp["t2"].device.type == "cpu"
    assert isinstance(tdp["list"], list)  # Unchanged


def test_expanded_with_candidates_preserves_dtype():
    """Test that expansion preserves tensor dtype."""
    targets = {
        "float32": torch.randn(2, 10, dtype=torch.float32),
        "int64": torch.randint(0, 100, (2, 5), dtype=torch.int64),
    }
    tdp = TargetsDictPlus(targets)

    expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates=3)

    assert expanded["float32"].dtype == torch.float32
    assert expanded["int64"].dtype == torch.int64


def test_text_inputs_manager_whitespace_preservation():
    """Test that TextInputsManager preserves whitespace correctly."""
    texts = ["  Start  {{T}}  end  "]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{T}}")

    assert tim.before_texts == ["  Start  "]
    assert tim.after_texts == ["  end  "]

    triggers = ["X"]
    res = tim.get_triggered_inputs(triggers)
    assert res["inputs_texts"][0][0] == "  Start  X  end  "

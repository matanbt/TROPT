from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.models.inputs import TextInputsManager
from tropt.common import ModelInput

# --- TargetsDictPlus Tests (old and commented out) ---

# def test_targets_dict_plus_init():
#     targets = {"key1": [1, 2, 3]}
#     tdp = TargetsDictPlus(targets, n_messages=3)
#     assert tdp.n_messages == 3
#     assert tdp["key1"] == [1, 2, 3]

# def test_targets_dict_plus_infer_n_messages():
#     targets = {"key1": [1, 2, 3]}
#     tdp = TargetsDictPlus(targets)
#     assert tdp.n_messages == 3

# def test_targets_dict_plus_validation():
#     targets = {"key1": [1, 2]}
#     with pytest.raises(AssertionError):
#         TargetsDictPlus(targets, n_messages=3)

# def test_targets_dict_plus_setitem():
#     targets = {"key1": [1, 2, 3]}
#     tdp = TargetsDictPlus(targets)
#     with pytest.raises(AssertionError):
#         tdp["key2"] = [1, 2] # Wrong length
    
#     tdp["key2"] = [4, 5, 6]
#     assert tdp["key2"] == [4, 5, 6]

# def test_to_device():
#     t1 = torch.tensor([1, 2, 3])
#     targets = {"t1": t1}
#     tdp = TargetsDictPlus(targets)
    
#     device = torch.device("cpu")
#     tdp.to_device(device)
#     assert tdp["t1"].device.type == "cpu"

# def test_expanded_with_candidates_tensor():
#     # Input: (n_messages, ...)
#     # Output: (n_messages, n_candidates, ...)
#     n_messages = 2
#     n_candidates = 3
#     feature_dim = 4
    
#     targets = {"t1": torch.randn(n_messages, feature_dim)}
#     tdp = TargetsDictPlus(targets, n_messages=n_messages)
    
#     expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates)
    
#     assert expanded["t1"].shape == (n_messages, n_candidates, feature_dim)
#     # Check content: all candidates for a message should be identical
#     assert torch.allclose(expanded["t1"][0, 0], targets["t1"][0])
#     assert torch.allclose(expanded["t1"][0, 1], targets["t1"][0])
#     assert torch.allclose(expanded["t1"][1, 0], targets["t1"][1])

# def test_expanded_with_candidates_list_strings():
#     # Input: List of strings (length n_messages)
#     # Output: List of lists of strings (length n_messages, each list length n_candidates)
#     targets = {"s1": ["msg1", "msg2"]}
#     tdp = TargetsDictPlus(targets)
#     n_candidates = 3
    
#     expanded = TargetsDictPlus.get_expanded_with_candidates(tdp, n_candidates)
    
#     assert len(expanded["s1"]) == 2
#     assert len(expanded["s1"][0]) == 3
#     assert expanded["s1"][0][0] == "msg1"
#     assert expanded["s1"][1][0] == "msg2"

# ---------------------------------------------------------------------------
# --- TextInputsManager Tests ---

def test_text_inputs_manager_init():
    texts = ["Hello {{OPTIMIZED_TRIGGER}} World", "Start {{OPTIMIZED_TRIGGER}}"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder="{{OPTIMIZED_TRIGGER}}")
    
    assert tim.n_messages == 2
    assert tim.before_texts == ["Hello ", "Start "]
    assert tim.after_texts == [" World", ""]

def test_text_inputs_manager_get_triggered_inputs():
    texts = [f"A{OPTIMIZED_TRIGGER_PLACEHOLDER}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)
    
    triggers = ["1", "2"]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)
    
    assert isinstance(res, ModelInput)
    assert res.input_texts == ["A1B", "A2B"]


def test_text_inputs_manager_get_triggered_inputs_chosen_message():
    texts = [f"A{OPTIMIZED_TRIGGER_PLACEHOLDER}B", f"C{OPTIMIZED_TRIGGER_PLACEHOLDER}D"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)
    
    triggers = ["1", "2"]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=1)
    
    assert isinstance(res, ModelInput)
    assert res.input_texts == ["C1D", "C2D"]


# ==============================================================================
# Edge Cases and Robustness Tests
# ==============================================================================

def test_targets_dict_plus_empty_dict():
    """Test Targets works when empty (no fields set)."""
    tdp = Targets()
    assert tdp.n_messages == 0
    assert len(tdp.model_dump(exclude_none=True)) == 0



def test_targets_dict_plus_mixed_types():
    """Test TargetsDictPlus with mixed tensor and list types."""
    targets = Targets(
        target_vectors=torch.randn(3, 10),
        target_response_strs=["a", "b", "c"],
    )

    assert targets.n_messages == 3
    assert isinstance(targets.target_vectors, torch.Tensor)
    assert isinstance(targets.target_response_strs, list)


def test_targets_dict_plus_nested_tensors():
    """Test TargetsDictPlus with higher-dimensional tensors."""
    targets = Targets(
        target_vectors=torch.randn(4, 128),  # (n_messages, d_model)
    )

    assert targets.n_messages == 4


def test_targets_dict_plus_validation_mismatch():
    """Test that TargetsDictPlus validates lengths across all keys."""
    with pytest.raises(ValueError):
        Targets(
            target_response_strs=["a", "b", "c"],
            target_vectors=torch.randn(2, 10),
        )


def test_text_inputs_manager_no_placeholder():
    """Test TextInputsManager when no placeholder is found."""
    texts = ["Hello World", "No trigger here"]

    with pytest.raises(ValueError):
        TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)


def test_text_inputs_manager_multiple_placeholders():
    """Test TextInputsManager with multiple placeholders in one text."""
    texts = [f"Start {OPTIMIZED_TRIGGER_PLACEHOLDER} middle {OPTIMIZED_TRIGGER_PLACEHOLDER} end"]

    with pytest.raises(ValueError):
        TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)


def test_text_inputs_manager_empty_trigger():
    """Test TextInputsManager with empty trigger string."""
    texts = [f"A{OPTIMIZED_TRIGGER_PLACEHOLDER}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    triggers = [""]  # Empty trigger
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)

    assert res.input_texts[0] == "AB"


def test_text_inputs_manager_long_trigger():
    """Test TextInputsManager with very long trigger."""
    texts = [f"Start {OPTIMIZED_TRIGGER_PLACEHOLDER} end"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    long_trigger = "x" * 1000
    triggers = [long_trigger]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)

    assert res.input_texts[0] == f"Start {long_trigger} end"


def test_text_inputs_manager_special_characters():
    """Test TextInputsManager with special characters in trigger."""
    texts = [f"Before {OPTIMIZED_TRIGGER_PLACEHOLDER} after"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    special_trigger = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    triggers = [special_trigger]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)

    assert res.input_texts[0] == f"Before {special_trigger} after"


def test_text_inputs_manager_unicode():
    """Test TextInputsManager with unicode characters."""
    texts = [f"Text: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    unicode_trigger = "Hello 世界 🌍"
    triggers = [unicode_trigger]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)

    assert res.input_texts[0] == f"Text: {unicode_trigger}"


def test_text_inputs_manager_many_messages():
    """Test TextInputsManager with many messages."""
    n_messages = 100
    texts = [f"Message {i}: {OPTIMIZED_TRIGGER_PLACEHOLDER}" for i in range(n_messages)]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    assert tim.n_messages == n_messages

    triggers = ["test"]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=50)

    assert res.input_texts[0] == "Message 50: test"


def test_text_inputs_manager_many_candidates():
    """Test TextInputsManager with many candidate triggers."""
    texts = [f"A{OPTIMIZED_TRIGGER_PLACEHOLDER}B"]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    n_candidates = 1000
    triggers = [str(i) for i in range(n_candidates)]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)

    assert len(res.input_texts) == n_candidates
    assert res.input_texts[0] == "A0B"
    assert res.input_texts[999] == "A999B"


def test_targets_dict_plus_device_movement():
    """Test that to_device correctly moves all tensors."""
    targets = Targets(
        target_vectors=torch.randn(2, 10),
        target_response_strs=["a", "b"],
    )

    # Move to CPU (should already be there, but test the mechanism)
    targets_on_cpu = targets.to_device(torch.device("cpu"))

    assert targets_on_cpu.target_vectors.device.type == "cpu"
    assert isinstance(targets_on_cpu.target_response_strs, list)  # Unchanged


def test_text_inputs_manager_whitespace_preservation():
    """Test that TextInputsManager preserves whitespace correctly."""
    texts = [f"  Start  {OPTIMIZED_TRIGGER_PLACEHOLDER}  end  "]
    tim = TextInputsManager(texts, optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER)

    assert tim.before_texts == ["  Start  "]
    assert tim.after_texts == ["  end  "]

    triggers = ["X"]
    res = tim.get_triggered_inputs(triggers, chosen_message_idx=0)
    assert res.input_texts[0] == "  Start  X  end  "

import pytest
import torch

from tropt.models.inputs import TargetsDictPlus, TextInputsManager

# TODO review

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

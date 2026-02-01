"""Test introspection-based loss resolution and Pydantic validation."""

import torch
from tropt.models.outputs import ModelOutput
from tropt.models.inputs import ModelInput
from tropt.loss.base import PrefillCELoss, SimilarityLoss, SteeringActivationLoss
from tropt.loss.resolution import compute_loss_from_model_data, LossResolutionError

def test_pydantic_validation():
    """Test that Pydantic correctly validates shapes."""
    print("\n" + "="*60)
    print("Testing Pydantic Validation")
    print("="*60)

    # Test 1: Valid output_embeddings
    print("\n[Test 1] Valid 2D embeddings...")
    try:
        output = ModelOutput(output_embeddings=torch.randn(4, 768))
        print("[OK] Valid 2D embeddings accepted")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 2: Invalid output_embeddings (wrong ndim)
    print("\n[Test 2] Invalid 1D embeddings (should fail)...")
    try:
        output = ModelOutput(output_embeddings=torch.randn(768))
        print("[FAIL] Invalid shape accepted (should have raised error)")
        return False
    except ValueError as e:
        print(f"[OK] Correctly rejected: {e}")

    # Test 3: Valid output_logits
    print("\n[Test 3] Valid 3D logits...")
    try:
        output = ModelOutput(output_logits=torch.randn(2, 50, 32000))
        print("[OK] Valid 3D logits accepted")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 4: Invalid output_logits (wrong ndim)
    print("\n[Test 4] Invalid 2D logits (should fail)...")
    try:
        output = ModelOutput(output_logits=torch.randn(2, 50))
        print("[FAIL] Invalid shape accepted (should have raised error)")
        return False
    except ValueError as e:
        print(f"[OK] Correctly rejected: {e}")

    # Test 5: Valid input_trigger_ids
    print("\n[Test 5] Valid 2D trigger IDs...")
    try:
        input_data = ModelInput(input_trigger_ids=torch.randint(0, 100, (4, 20)))
        print("[OK] Valid 2D trigger IDs accepted")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 6: Invalid input_trigger_ids (wrong ndim)
    print("\n[Test 6] Invalid 1D trigger IDs (should fail)...")
    try:
        input_data = ModelInput(input_trigger_ids=torch.randint(0, 100, (20,)))
        print("[FAIL] Invalid shape accepted (should have raised error)")
        return False
    except ValueError as e:
        print(f"[OK] Correctly rejected: {e}")

    print("\n[PASS] All Pydantic validation tests passed!")
    return True


def test_introspection_resolution():
    """Test that introspection correctly resolves loss parameters."""
    print("\n" + "="*60)
    print("Testing Introspection-Based Loss Resolution")
    print("="*60)

    # Test 1: PrefillCELoss (LogitBasedLoss)
    print("\n[Test 1] PrefillCELoss with output_logits and target_outputs_toks...")
    bsz, seq_len, vocab_size = 2, 10, 500

    model_output = ModelOutput(
        output_logits=torch.randn(bsz, seq_len, vocab_size)
    )
    # Use TargetsDictPlus to avoid Pydantic validation issues
    from tropt.models.inputs import TargetsDictPlus
    targets_dict = {"target_outputs_toks": torch.randint(0, vocab_size, (bsz, seq_len))}
    targets = TargetsDictPlus(targets=targets_dict, n_messages=bsz)

    model_input = ModelInput(targets=targets)
    loss_func = PrefillCELoss()

    try:
        loss = compute_loss_from_model_data(model_output, model_input, loss_func)
        assert loss.shape == (bsz,), f"Expected shape (2,), got {loss.shape}"
        assert not torch.isnan(loss).any(), "Loss contains NaN"
        print(f"[OK] Computed loss with shape {loss.shape}")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 2: SimilarityLoss (EmbeddingBasedLoss)
    print("\n[Test 2] SimilarityLoss with output_embeddings and target_vectors...")
    bsz, d_model = 4, 768

    model_output = ModelOutput(
        output_embeddings=torch.randn(bsz, d_model)
    )
    targets_dict = {"target_vectors": torch.randn(bsz, d_model)}
    targets = TargetsDictPlus(targets=targets_dict, n_messages=bsz)

    model_input = ModelInput(targets=targets)
    loss_func = SimilarityLoss()

    try:
        loss = compute_loss_from_model_data(model_output, model_input, loss_func)
        assert loss.shape == (bsz,), f"Expected shape (4,), got {loss.shape}"
        assert not torch.isnan(loss).any(), "Loss contains NaN"
        print(f"[OK] Computed loss with shape {loss.shape}")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 3: SteeringActivationLoss (HiddenStateBased)
    print("\n[Test 3] SteeringActivationLoss with output_hidden_states...")
    bsz, n_layers, seq_len, d_model = 2, 4, 10, 768

    model_output = ModelOutput(
        output_hidden_states=torch.randn(bsz, n_layers, seq_len, d_model)
    )
    targets_dict = {"target_directions": torch.randn(bsz, d_model)}
    targets = TargetsDictPlus(targets=targets_dict, n_messages=bsz)

    model_input = ModelInput(
        input_slices=[
            {"adv": slice(3, 7)},
            {"adv": slice(2, 6)},
        ],
        targets=targets
    )
    loss_func = SteeringActivationLoss()

    try:
        loss = compute_loss_from_model_data(model_output, model_input, loss_func)
        assert loss.shape == (bsz,), f"Expected shape (2,), got {loss.shape}"
        assert not torch.isnan(loss).any(), "Loss contains NaN"
        print(f"[OK] Computed loss with shape {loss.shape}")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 4: Missing required parameter (should fail)
    print("\n[Test 4] Missing required parameter (should fail with LossResolutionError)...")
    model_output = ModelOutput()  # No fields populated
    model_input = ModelInput()
    loss_func = PrefillCELoss()

    try:
        loss = compute_loss_from_model_data(model_output, model_input, loss_func)
        print("[FAIL] Should have raised LossResolutionError")
        return False
    except LossResolutionError as e:
        print(f"[OK] Correctly raised LossResolutionError:\n  {str(e)[:100]}...")
    except Exception as e:
        print(f"[FAIL] Unexpected error type: {type(e).__name__}: {e}")
        return False

    print("\n[PASS] All introspection resolution tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INTROSPECTION & PYDANTIC VALIDATION TESTS")
    print("="*60)

    results = []

    # Run Pydantic validation tests
    results.append(test_pydantic_validation())

    # Run introspection resolution tests
    results.append(test_introspection_resolution())

    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/{len(results)} test groups passed")
    print("="*60)

    if all(results):
        print("\n[PASS] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

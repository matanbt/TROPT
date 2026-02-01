"""Validation script for model wrapper refactoring.

This script performs basic checks to ensure the refactoring is working correctly:
1. Import checks
2. Dataclass instantiation
3. Type compatibility
4. Basic integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all new classes can be imported."""
    print("Testing imports...")

    try:
        from tropt.models import ModelOutput, ModelInput
        print("[OK] ModelOutput and ModelInput imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import ModelOutput/ModelInput: {e}")
        return False

    try:
        from tropt.loss import compute_loss_from_model_data, LossResolutionError
        print("[OK] Loss resolution functions imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import loss resolution: {e}")
        return False

    return True


def test_dataclass_instantiation():
    """Test that dataclasses can be instantiated."""
    print("\nTesting dataclass instantiation...")

    try:
        from tropt.models import ModelOutput, ModelInput
        import torch

        # Test ModelOutput
        output = ModelOutput(
            output_embeddings=torch.randn(2, 768),
            output_logits=None,
        )
        assert output.output_embeddings is not None
        assert output.output_logits is None
        print("[OK] ModelOutput instantiation successful")

        # Test ModelInput
        input_data = ModelInput(
            input_texts=["text1", "text2"],
            targets={"target_outputs": ["target1", "target2"]}
        )
        assert input_data.input_texts == ["text1", "text2"]
        assert input_data.input_embeds is None
        print("[OK] ModelInput instantiation successful")

    except Exception as e:
        print(f"[FAIL] Dataclass instantiation failed: {e}")
        return False

    return True


def test_model_output_fields():
    """Test that ModelOutput has all expected fields."""
    print("\nTesting ModelOutput fields...")

    try:
        from tropt.models import ModelOutput

        # Pydantic models use model_fields instead of dataclasses.fields()
        field_names = set(ModelOutput.model_fields.keys())
        expected_fields = {
            'output_embeddings',
            'output_logits',
            'output_hidden_states',
            'output_attentions',
            'generated_response_ids',
            'generated_response_strs',
            'generated_response_logits',
            'full_template_ids',
            'full_template_strs',
        }

        if expected_fields == field_names:
            print(f"[OK] ModelOutput has all {len(expected_fields)} expected fields")
        else:
            missing = expected_fields - field_names
            extra = field_names - expected_fields
            if missing:
                print(f"[FAIL] Missing fields: {missing}")
            if extra:
                print(f"[FAIL] Extra fields: {extra}")
            return False

    except Exception as e:
        print(f"[FAIL] Field check failed: {e}")
        return False

    return True


def test_model_input_fields():
    """Test that ModelInput has all expected fields."""
    print("\nTesting ModelInput fields...")

    try:
        from tropt.models import ModelInput

        # Pydantic models use model_fields instead of dataclasses.fields()
        field_names = set(ModelInput.model_fields.keys())
        expected_fields = {
            'input_texts',
            'input_trigger_ids',
            'input_embeds',
            'input_attention_mask',
            'input_prefix_cache_kwargs',
            'input_slices',
            'targets',
        }

        if expected_fields == field_names:
            print(f"[OK] ModelInput has all {len(expected_fields)} expected fields")
        else:
            missing = expected_fields - field_names
            extra = field_names - expected_fields
            if missing:
                print(f"[FAIL] Missing fields: {missing}")
            if extra:
                print(f"[FAIL] Extra fields: {extra}")
            return False

    except Exception as e:
        print(f"[FAIL] Field check failed: {e}")
        return False

    return True


def test_loss_resolution_imports():
    """Test that loss resolution can import loss types from base module."""
    print("\nTesting loss resolution imports...")

    try:
        # Loss types should be imported from tropt.loss.base, not resolution.py
        from tropt.loss.base import (
            LogitBasedLoss,
            TriggerLogitBasedLoss,
            AttentionBasedLoss,
            SteeringActivationLoss,
            EmbeddingBasedLoss,
            TextBasedLoss,
            CombinedLoss,
        )
        from tropt.loss.resolution import compute_loss_from_model_data, LossResolutionError
        print("[OK] All loss types and resolution functions are accessible")
    except ImportError as e:
        print(f"[FAIL] Loss type import failed: {e}")
        return False

    return True


def test_type_hints():
    """Test that type hints are properly set."""
    print("\nTesting type hints...")

    try:
        from tropt.models.model_base import LMBaseModel, EncoderBaseModel
        import inspect
        from typing import get_type_hints

        # Check LMBaseModel.__call__ return type
        lm_hints = get_type_hints(LMBaseModel.__call__)
        lm_return = str(lm_hints.get('return', ''))
        if 'ModelOutput' in lm_return or 'List[str]' in lm_return:
            print("[OK] LMBaseModel.__call__ has correct return type hint")
        else:
            print(f"[FAIL] LMBaseModel.__call__ return type unexpected: {lm_return}")
            return False

        # Check EncoderBaseModel.__call__ return type
        enc_hints = get_type_hints(EncoderBaseModel.__call__)
        enc_return = str(enc_hints.get('return', ''))
        if 'ModelOutput' in enc_return or 'Tensor' in enc_return:
            print("[OK] EncoderBaseModel.__call__ has correct return type hint")
        else:
            print(f"[FAIL] EncoderBaseModel.__call__ return type unexpected: {enc_return}")
            return False

    except Exception as e:
        print(f"[FAIL] Type hint check failed: {e}")
        return False

    return True


def main():
    """Run all validation tests."""
    print("="*60)
    print("MODEL WRAPPER REFACTORING VALIDATION")
    print("="*60)

    tests = [
        test_imports,
        test_dataclass_instantiation,
        test_model_output_fields,
        test_model_input_fields,
        test_loss_resolution_imports,
        test_type_hints,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*60)

    if all(results):
        print("\n[PASS] ALL VALIDATION TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED - Please review above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

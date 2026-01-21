import pytest
import torch

from tropt.loss.base import BaseLoss
from tropt.models import BaseModel
from tropt.models.inputs import TargetsDict
from tropt.optimizer.base import BaseOptimizer, OptimizerResult

# TODO review & consider dropping the mocks

class ConcreteOptimizer(BaseOptimizer):
    def optimize_trigger(
        self,
        texts,
        initial_trigger=None,
        targets=None,
    ) -> OptimizerResult:
        return OptimizerResult(
            best_trigger=torch.tensor([1]),
            best_trigger_str="test",
            best_loss=0.0,
            trigger_strs=["test"],
            losses=[0.0]
        )

class MockModel(BaseModel):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass

class MockLoss(BaseLoss):
    def __call__(self, *args, **kwargs):
        pass

def test_base_optimizer_init_and_requirements():
    model = MockModel()
    loss = MockLoss()
    
    # Test initialization
    optimizer = ConcreteOptimizer(model, loss)
    assert optimizer.model == model
    assert optimizer.loss_func == loss
    assert optimizer.tracker is not None # Should default to DummyTracker

    # Test requirements check (empty requirements for BaseOptimizer default, but subclasses use it)
    class RequiringOptimizer(BaseOptimizer):
        model_requirements = (str,) # Impossible requirement for a BaseModel
        def optimize_trigger(self, *args, **kwargs): pass

    with pytest.raises(AssertionError):
        RequiringOptimizer(model, loss)

def test_base_optimizer_abstract_methods():
    model = MockModel()
    loss = MockLoss()
    
    # Try to instantiate BaseOptimizer directly
    with pytest.raises(TypeError):
        BaseOptimizer(model, loss)

def test_optimizer_result_dataclass():
    res = OptimizerResult(
        best_trigger=torch.tensor([1, 2]),
        best_trigger_str="test",
        best_loss=0.5,
        trigger_strs=["a", "b"],
        losses=[1.0, 0.5]
    )
    assert res.best_loss == 0.5
    assert res.best_trigger_str == "test"

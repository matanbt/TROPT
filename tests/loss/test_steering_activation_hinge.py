"""SteeringActivationLoss with a hinge stops pushing once the similarity passes the hinge."""

import torch

from tropt.common import SliceKey
from tropt.loss import SteeringActivationLoss


def _loss(similarities, **kwargs):
    # One layer, one position; the direction is e_0 (normalized inside the loss), so similarity = hidden[..., 0].
    d_model = 4
    hidden = torch.zeros(len(similarities), 1, 1, d_model)
    hidden[:, 0, 0, 0] = torch.tensor(similarities)
    direction = torch.zeros(d_model)
    direction[0] = 3.0
    loss = SteeringActivationLoss(slc_name=SliceKey.INPUT_LAST_TOKEN, **kwargs)
    return loss(hidden, direction, input_slices={SliceKey.INPUT_LAST_TOKEN: slice(0, 1)})


def test_hinge_steering_away_is_zero_below_the_hinge():
    out = _loss([-2.0, 0.5, 3.0], steer_away=True, hinge=1.0)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 2.0])), out


def test_hinge_steering_towards_is_zero_above_the_hinge():
    out = _loss([-2.0, 0.5, 3.0], steer_away=False, hinge=1.0)
    assert torch.allclose(out, torch.tensor([3.0, 0.5, 0.0])), out


def test_without_hinge_the_loss_is_the_dot_product():
    out = _loss([-2.0, 0.5, 3.0], steer_away=True)
    assert torch.allclose(out, torch.tensor([-2.0, 0.5, 3.0])), out

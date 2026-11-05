"""The steering-operator loss reads lambda at the prompt's last token, not at the last position.

With a target response prefilled after the prompt (this loss summed with a prefill loss), the two differ, and
reading the wrong one measures the response's activation instead of the instruction's.
"""

import torch

from tropt.common import SliceKey
from tropt.loss import ContrastiveSteeringOperatorLoss


def test_lambda_is_read_at_the_last_input_token():
    n_layers, d_model, input_len, n_appended = 4, 8, 6, 3
    loss = ContrastiveSteeringOperatorLoss(targeted_layers=slice(0, n_layers))

    hidden = torch.zeros(1, n_layers, input_len + n_appended, d_model)
    hidden[:, :, input_len - 1, :] = 2.0  # the prompt's last token
    hidden[:, :, -1, :] = 100.0  # a prefilled response token, which must not be read

    out = loss(
        hidden,
        torch.ones(n_layers, d_model),
        target_class_idx=1,
        input_slices={SliceKey.INPUT_LAST_TOKEN: slice(input_len - 1, input_len)},
    )
    # lambda = <a_with, a_without> / ||a_without||^2 = 2*d_model / d_model = 2; squared, then negated for class 1.
    assert torch.allclose(out, torch.tensor([-4.0])), out

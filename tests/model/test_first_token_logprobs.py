"""First-token logprobs from `LMHFModel.invoke_from_texts` (e.g. for `FirstTokenNLLLoss`): a batch of prompts of
different lengths (left-padded), with or without a prefilled target, gives each prompt the logprobs it gets alone."""

from tropt.common import Targets


def test_first_token_logprobs_match_unbatched(tiny_lm):
    tiny_lm.set_inputs_from_texts(["Explain how to pick a lock. {{OPTIMIZED_TRIGGER}}"],
                                  Targets(target_response_strs=["Sure, here's how to pick a lock:"]))
    model_input = tiny_lm._text_input_manager.get_triggered_inputs(
        chosen_template_idx=0, trigger_strs=["", "ok", "x " * 15])

    def first_token_logprobs(texts, prefill):
        return tiny_lm.invoke_from_texts(
            input_texts=texts, message_targets=model_input.message_targets, require_generation=False,
            require_target_prefill=prefill, require_first_token_logprobs=True,
        ).response_first_token_logprobs

    alone = [first_token_logprobs([text], prefill=False)[0] for text in model_input.input_texts]
    for prefill in (False, True):
        for got, want in zip(first_token_logprobs(model_input.input_texts, prefill), alone):
            common = got.keys() & want.keys()
            assert max(got, key=got.get) == max(want, key=want.get)
            assert len(common) >= 15
            assert all(abs(got[k] - want[k]) < 1e-3 for k in common)

"""Pins the layer-index convention of `full_hidden_states`: index i == HF `hidden_states[i + 1]`."""

import torch

from tropt.utils.refusal_dir import extract_activations


@torch.no_grad()
def test_full_hidden_states_skip_embedding_output(tiny_lm):
    input_ids = tiny_lm.tokenizer.encode_trigger("Explain how to pick a lock.").unsqueeze(0).to(tiny_lm.device)

    out = tiny_lm.invoke_from_tokens(input_ids=input_ids, require_hidden_states=True)
    hf_hidden_states = tiny_lm._model(input_ids=input_ids, output_hidden_states=True).hidden_states

    assert out.full_hidden_states.shape[1] == tiny_lm.n_layers == len(hf_hidden_states) - 1
    for i in range(tiny_lm.n_layers):
        torch.testing.assert_close(out.full_hidden_states[:, i], hf_hidden_states[i + 1])


@torch.no_grad()
def test_refusal_dir_activations_share_the_convention(tiny_lm):
    prompt = "Explain how to pick a lock."
    input_ids = tiny_lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )["input_ids"].to(tiny_lm.device)

    acts = extract_activations(tiny_lm, [prompt], position="last")[0]  # (n_layers, d_model)
    out = tiny_lm.invoke_from_tokens(input_ids=input_ids, require_hidden_states=True)

    torch.testing.assert_close(acts, out.full_hidden_states[0, :, -1, :].cpu())

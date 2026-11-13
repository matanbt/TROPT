"""Pins the layer-index convention of `full_hidden_states`: index i == HF `hidden_states[i]` (0 = embeddings)."""

import torch

from tropt.utils.refusal_dir import extract_activations


@torch.no_grad()
def test_full_hidden_states_match_hf_indexing(tiny_lm):
    input_ids = tiny_lm.tokenizer.encode_trigger("Explain how to pick a lock.").unsqueeze(0).to(tiny_lm.device)

    out = tiny_lm.invoke_from_tokens(input_ids=input_ids, require_hidden_states=True)
    hf_hidden_states = tiny_lm._model(input_ids=input_ids, output_hidden_states=True).hidden_states

    assert out.full_hidden_states.shape[1] == len(hf_hidden_states) == tiny_lm.n_layers + 1
    for i in range(len(hf_hidden_states)):
        torch.testing.assert_close(out.full_hidden_states[:, i], hf_hidden_states[i])


@torch.no_grad()
def test_refusal_dir_activations_share_the_convention(tiny_lm):
    prompt = "Explain how to pick a lock."
    input_ids = tiny_lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )["input_ids"].to(tiny_lm.device)

    acts = extract_activations(tiny_lm, [prompt], position="last")[0]  # (n_layers + 1, d_model)
    out = tiny_lm.invoke_from_tokens(input_ids=input_ids, require_hidden_states=True)

    assert acts.shape[0] == tiny_lm.n_layers + 1
    torch.testing.assert_close(acts, out.full_hidden_states[0, :, -1, :].cpu())

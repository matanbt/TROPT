import itertools
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelOutput,
    SliceKey,
    Targets,
    TextTemplates,
)
from tropt.model import (
    GradientTokenAccessMixin,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.model.huggingface.base import (
    HuggingFaceBackendModel,
    HuggingFaceTokenInputManager,
)
from tropt.model.model_mixins import GradientEmbedAccessMixin

logger = logging.getLogger(__name__)

_MAX_TOP_LOGPROBS = 20  # matches OpenAI/LiteLLM convention


# ======================= Input/Output Handlers logic =======================
class LMHFTokenInputManager(HuggingFaceTokenInputManager):
    targets: Targets
    """
    optioanlly includes `target_response_toks` (n_templates, target_seq_len) if target outputs are provided;
    these are used to prefill the response per message
    """

    @cached_property
    def _prefill_embeds(self) -> List[Float[Tensor, "target_seq_len embd_dim"]]:
        assert self.targets.target_response_toks is not None, "target_response_toks must be set"
        return [self.embed_func(target_output) for target_output in self.targets.target_response_toks]

    def get_triggered_inputs(
        self,
        do_append_embeds: bool = False,
        **kwargs,
    ):
        assert (
            kwargs.get("append_embeds", None) is None
        ), "append_embeds should not be passed directly to LM models. Use `target_embeds` property instead."
        if do_append_embeds:
            assert self.targets.target_response_toks is not None, "target_response_toks must be provided in targets to append prefill_embeds to inputs."

        return super().get_triggered_inputs(
            **kwargs,
            append_embeds=self._prefill_embeds if do_append_embeds else None,
            do_append_embeds=do_append_embeds,
        )


# ======================= Model logic =======================

class LMHFModel(
    # HF backend first so its `device`/`dtype` win MRO over `BaseModel`'s defaults:
    HuggingFaceBackendModel,
    LMBaseModel,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    LogitsTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        forward_pass_batch_size: int = 1024,
        backward_pass_batch_size: int = 32,
        # more args:
        use_prefix_cache: bool = True,
        set_model_to_train: bool = False,
        use_eager_attention: bool = False,
        loaded_model: Optional[AutoModelForCausalLM] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs,  # to be handed to HuggingFace model init
    ):
        if loaded_model is not None:
            logger.info(f"Using provided loaded model for {model_name}.")
            assert isinstance(loaded_model, transformers.PreTrainedModel)
            self._model = loaded_model
        else:
            if use_eager_attention:
                # required for to support attention-based losses
                model_kwargs["attn_implementation"] = "eager"
                logger.info(
                    f"Using eager attention for model {model_name} to support attention-based loss."
                )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device or "auto",
                dtype=dtype or "auto",
                **model_kwargs
            )

        # Set tokenizer:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert isinstance(_tokenizer, transformers.PreTrainedTokenizerBase)
        self._tokenizer = _tokenizer

        # Set embedding layer:
        embedding_layer = self._model.get_input_embeddings()
        assert embedding_layer is not None, f"Model {model_name} has no input embeddings"
        self._embedding_layer: torch.nn.Module = embedding_layer
        self._use_prefix_cache = use_prefix_cache
        self.chat_template_kwargs = chat_template_kwargs or {}


        if not self._tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            self._tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )
        if self._tokenizer.padding_side != "left":
            # Left padding is required for LM prefilling and generation
            logger.warning(
                "Tokenizer padding side is not 'left'. Overriding to 'left' (required for causal LM generation)."
            )
            self._tokenizer.padding_side = "left"

        if not self._tokenizer.pad_token:
            if self._tokenizer.eos_token:
                logger.warning(
                    "Tokenizer does not have a pad token. Setting pad token to eos token."
                )
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer does not have a pad token or an eos token. Please set a pad token."
                )

    def _update_targets_by_model(self, targets: Optional[Targets]) -> Targets:
        """
        An LM-specific logic to update the targets. Particularly, if target response strings are provided, we tokenize them and store in `target_response_toks`.
        """
        if targets is None:
            targets = Targets()

        # Encode target outputs, if provided
        if targets.target_response_strs is not None:
            tokenized_lists = self._tokenizer(
                targets.target_response_strs, add_special_tokens=False
            )["input_ids"]
            # convert to list of tensors
            targets.target_response_toks = [
                torch.tensor(ids, device=self._model.device) for ids in tokenized_lists
                # each of shape (target_seq_len,)
            ]

        # Move targets to device
        targets = targets.to_device(self._model.device)
        return targets

    # ======================= Token-access methods =======================

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """
        Prepares and stores the inputs manager for the model, including tokenization and target processing.

        Args:
            templates: List of input templates containing the trigger placeholder.
            targets: Optional Targets object containing target response strings to optimize towards.
        """
        assert isinstance(templates, list) and all(isinstance(t, str) for t in templates), "templates must be a list of strings."
        assert all(
            [t.count(OPTIMIZED_TRIGGER_PLACEHOLDER) == 1 for t in templates]
        ), f"`templates` must contain the `{OPTIMIZED_TRIGGER_PLACEHOLDER}` placeholder."

        # put in chat-template + special tokens & tokenizer
        template_tok_ids: List[List[int]] = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": template}],
                tokenize=True,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )["input_ids"]
            for template in templates
        ]  # `tokenize` returns List[List[int]]

        # Update targets (eg tokenize target response strs if toks not provided, move to device, etc.)
        targets = self._update_targets_by_model(targets)

        # Build the input manager, that will allow combining with different triggers
        self._token_input_manager = LMHFTokenInputManager(
            templates_ids=template_tok_ids,
            device=self._model.device,
            model=self._model,
            tokenizer=self._tokenizer,
            embed_func=self._embedding_layer,
            use_prefix_cache=self._use_prefix_cache,
            targets=targets,
        )

    @torch.no_grad()
    def compute_logits_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        keep_message_dim: bool = False,
        return_trigger_logits_only: bool = False,
        return_after_trigger_logits_only: bool = False,
    ) -> (
        Float[Tensor, "n_templates n_candidates seq_len vocab_size"]
        | Tuple[
            Float[Tensor, "n_templates n_candidates seq_len vocab_size"], List[slice]
        ]
    ):
        """
        Given a batch of candidate trigger token ids and inputs object, returns the logits for the next token after the input sequence (i.e., after the trigger + input text + target text, if provided).

        Args:
            candidate_trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                the token ids of the candidate trigger sequences to evaluate
            inputs: LMHFTokenInputManager
                the inputs object containing the input text and target text (if provided)
            return_slices: bool
                whether to return the slices corresponding to each input in the batch (default: False)
            keep_message_dim: bool
                whether to keep the message dimension in the output logits (default: False)
            return_trigger_logits_only: bool
                whether to return only the logits corresponding to the trigger tokens (default: False)
            return_after_trigger_logits_only: bool
                whether to return only the logits corresponding to predicting the next token after trigger (default: False)
        """
        assert int(return_trigger_logits_only) + int(return_after_trigger_logits_only) <= 1, "Cannot set both `return_trigger_logits_only` and `return_after_trigger_logits_only` to True."
        assert self._token_input_manager is not None, "Token input manager is not initialized. Please call set_inputs_from_tokens() first."

        input_manager = self._token_input_manager
        n_templates = input_manager.n_templates
        n_candidates, trigger_seq_len = candidate_trigger_ids.shape

        # Compute the logits (in batches)
        @find_executable_batch_size(starting_batch_size=self._forward_pass_batch_size)
        def _compute_logits_batched(batch_size):
            full_logits = [[] for _ in range(n_templates)]
            slices: List[Optional[Dict[SliceKey, Optional[slice]]]] = [None for _ in range(n_templates)]

            for template_idx, cand_idx in itertools.product(
                range(n_templates),
                range(0, n_candidates, batch_size),
            ):
                cand_idx_end = min(cand_idx + batch_size, n_candidates)
                batch_candidate_trigger_ids = candidate_trigger_ids[cand_idx:cand_idx_end]

                # Get inputs for this specific message
                model_input = input_manager.get_triggered_inputs(
                    chosen_template_idx=template_idx,
                    trigger_ids=batch_candidate_trigger_ids,
                )
                # Compute the logits
                logits_batch = self.invoke_from_tokens(
                    **model_input.to_dict(),
                ).full_logits

                full_logits[template_idx].append(logits_batch)
                slices[template_idx] = model_input.input_slices

            # Stack all logits per message
            logits_per_message = [torch.cat(msg_logits, dim=0) for msg_logits in full_logits]
            logits = torch.stack(logits_per_message, dim=0)  # (n_templates, n_candidates, seq_len, vocab_size)

            return logits, slices

        logits, slices = _compute_logits_batched()
        # (n_templates, n_candidates, seq_len, vocab_size)

        if return_trigger_logits_only or return_after_trigger_logits_only:
            # return only the logits for the trigger part
            trigger_logits = torch.zeros(
                (n_templates, n_candidates, (trigger_seq_len if return_trigger_logits_only else 1), logits.shape[-1]),
                device=logits.device,
            )  # (n_templates, n_candidates, trigger_seq_len, vocab_size)
            for i_template in range(n_templates):
                slc_trigger = slices[i_template][SliceKey.TRIGGER]  # trigger slice for this candidate

                # extract the relevant logits
                if return_trigger_logits_only:
                    slc = slice(slc_trigger.start - 1, slc_trigger.stop - 1)
                    assert slc.stop - slc.start == trigger_seq_len, "Trigger slice length does not match candidate trigger length."
                else:  # return_after_trigger_logits_only
                    # take the logits at last trigger token
                    slc = slice(slc_trigger.stop - 1, slc_trigger.stop)

                trigger_logits[i_template] = logits[i_template, :, slc, :]

                assert trigger_logits.shape[2] == slc.stop - slc.start, "Extracted trigger logits length does not match expected length."

            logits = trigger_logits

        if not keep_message_dim:
            if not (return_trigger_logits_only or return_after_trigger_logits_only):
                logger.warning(
                    "`keep_message_dim` is False but neither `return_trigger_logits_only` nor `return_after_trigger_logits_only` is True. " \
                    "Averaging over messages might mix logits from different slices if the trigger is not aligned across message templates."
                )
            logits = logits.mean(dim=0)  # (n_candidates, seq_len, vocab_size)

        return logits

    def invoke_from_tokens(
        self,
        input_embeds: Optional[Float[Tensor, "bsz seq_len embd_dim"]] = None,
        input_ids: Optional[Int[Tensor, "bsz seq_len"]] = None,
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        input_prefix_cache_kwargs: Optional[Dict[str, Any]] = None,
        input_slices: Optional[Dict[str, slice]] = None,

        # computation flags:
        require_target_prefill: bool = False,
        require_generation: bool = False,
        require_hidden_states: bool = False,
        require_attentions: bool = False,
        require_first_token_logprobs: bool = False,
        count_backward: bool = False,
        # generation kwargs (only used when require_generation=True):
        max_new_tokens: int = 128,
        greedy_decode: bool = True,
        **kwargs
    ) -> ModelOutput:
        """
        Performs a forward pass through the model given input embeddings and attention mask.

        Args:
            input_embeds: Input embeddings tensor of shape (bsz, seq_len, embd_dim). Primary input.
            input_ids: Token IDs tensor of shape (bsz, seq_len). Used as a fallback when
                `input_embeds` is not provided; embedded internally via the model's embedding layer.
            input_attention_mask: Attention mask tensor of shape (bsz, seq_len).
            input_prefix_cache_kwargs: Optional dict of prefix cache kwargs to pass to the model.
            input_slices: Optional dict mapping slice keys to slices for extracting specific parts of the output.

            require_target_prefill: Whether the input includes a prefixed target, of which indices are marked by the input_slices, and we should extract it logits.
            require_generation: Whether to perform generation, in addition to forward pass.
            require_hidden_states: Whether to return hidden states in the output.
            require_attentions: Whether to return attentions in the output.
            require_first_token_logprobs: Whether to return log-probabilities for the top-20 candidates for the first generated token.
            count_backward: Whether this forward pass will be back-propagated through (set by gradient methods).

        Returns:
            ModelOutput: The output of the model containing logits, hidden states, and attentions as applicable.
        """
        if require_attentions and self._model.config._attn_implementation != "eager":
            logger.warning(
                "AttentionBasedLoss is used but the model is not using eager attention. "
                "This may lead to incorrect attention outputs. Consider initializing the model with eager attention, by passing LMHFModel the flag `use_eager_attention=True`."
            )
        if require_attentions and input_prefix_cache_kwargs:
            raise ValueError(
                "Attention-based losses are incompatible with prefix caching; initialize model with `use_prefix_cache=False`. "
            )

        # Resolve input: input_embeds takes priority; fall back to input_ids.
        if input_embeds is None:
            assert input_ids is not None, "Either `input_embeds` or `input_ids` must be provided to HF's invoke_from_tokens."
            input_embeds = self._embedding_layer(input_ids)

        if input_attention_mask is None:
            input_attention_mask = torch.ones(input_embeds.shape[:2], device=input_embeds.device)

        outputs = self._model(
            inputs_embeds=input_embeds,
            attention_mask=input_attention_mask,
            output_attentions=require_attentions,
            output_hidden_states=require_hidden_states,
            **(input_prefix_cache_kwargs or {})
        )

        self._update_invoke_stats(
            n_tokens=int(input_attention_mask.sum().item()),
            n_samples=input_embeds.shape[0],
            count_backward=count_backward,
        )

        # Extract first-token logprobs, if requested
        response_first_token_logprobs = None
        if require_first_token_logprobs:
            last_token_indices = input_attention_mask.sum(dim=1).long() - 1  # (bsz,)
            batch_idx = torch.arange(input_embeds.shape[0], device=input_embeds.device)
            first_tok_logits = outputs.logits[batch_idx, last_token_indices]  # (bsz, vocab)
            first_tok_lps = torch.nn.functional.log_softmax(first_tok_logits, dim=-1)
            top_lps, top_ids = torch.topk(
                first_tok_lps, min(_MAX_TOP_LOGPROBS, first_tok_lps.shape[-1]), dim=-1
            )  # (bsz, top_k)
            response_first_token_logprobs = [
                {self._tokenizer.decode([top_ids[b, i].item()]): top_lps[b, i].item()
                 for i in range(top_ids.shape[-1])}
                for b in range(first_tok_lps.shape[0])
            ]

        # Extract prefill logits, if exist and requested
        prefill_response_logits = None
        if require_target_prefill:
            assert input_slices is not None, "input_slices must be provided to extract prefill logits when `require_target_prefill` is True."
            response_slc = input_slices[SliceKey.APPENDED]
            prefill_response_logits = outputs.logits[:, response_slc.start - 1 : response_slc.stop - 1, :]  # (bsz, response_seq_len, vocab_size)

        # Generate response, if requested
        generated_response_strs = None
        generated_response_ids = None
        generated_response_logits = None
        if require_generation:
            hf_gen_kwargs = {
                "do_sample": not greedy_decode,
                "output_logits": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
            }
            with torch.no_grad():  # generation doesn't require grad anyway
                generation_output = self._model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=input_attention_mask,
                    **hf_gen_kwargs,
                    **(input_prefix_cache_kwargs or {}),
                )
            # When inputs_embeds is used, sequences only contains generated token IDs (no prompt IDs)
            generated_response_ids = [generation_output.sequences[i] for i in range(input_embeds.shape[0])]
            generated_response_logits = torch.stack(generation_output.logits, dim=1)  # (bsz, gen_len, vocab)
            generated_response_strs = self._tokenizer.batch_decode(generated_response_ids, skip_special_tokens=True)
            self._update_invoke_stats(
                n_tokens=sum(len(t) for t in generated_response_ids),
                n_samples=input_embeds.shape[0],
            )

        return ModelOutput(
            full_logits=outputs.logits,
            prefill_response_logits=prefill_response_logits,
            full_attentions=torch.stack(outputs.attentions, dim=1) if require_attentions else None,
            full_hidden_states=torch.stack(outputs.hidden_states[1:], dim=1) if require_hidden_states else None,  # (skips input embedding (layer 0))
            generated_response_strs=generated_response_strs,
            generated_response_ids=generated_response_ids,
            generated_response_logits=generated_response_logits,
            response_first_token_logprobs=response_first_token_logprobs,
        )

    # ======================= Text-access methods =======================

    def set_inputs_from_texts(self, templates, targets=None):

        # Update targets (eg tokenize target response strs if toks not provided, move to device, etc.)
        targets = self._update_targets_by_model(targets)

        return super().set_inputs_from_texts(templates, targets)

    def invoke_from_texts(
        self,
        input_texts: Optional[List[str]] = None,
        message_targets: Optional[MessageTargets] = None,

        greedy_decode: bool = True,
        max_new_tokens: int = 128,

        require_target_prefill: bool = False,
        require_generation: bool = True,
        require_first_token_logprobs: bool = False,
    ) -> ModelOutput:
        """
        Generate text completions. Always returns a ModelOutput.
        - If self.do_prefill_response is True, and the relevant target response prefix is available, the generation starts after the prefilled response, and the returned logits will include the prefilled response portion.

        Args:
            input_texts: list of plain-text prompts.
            message_targets: Optional MessageTargets object. Mainly relevant if `require_target_prefill` is True, in which case the target responses will be prefixed to the model output.

            greedy_decode: Whether to use greedy decoding (vs. sampling) for generation.
            max_new_tokens: The maximum number of new tokens to generate.
            require_target_prefill: Whether to prefill the target response in the model input (if provided in `message_targets`) and return the corresponding logits.
            require_generation: Whether to perform generation. If False, performs only the forward pass.
            require_first_token_logprobs: Whether to return log-probabilities for the top-20 candidates for the first generated token.
        """

        assert input_texts is not None, "input_texts must be provided."
        if require_target_prefill:
            assert message_targets is not None, "message_targets must be provided if require_target_prefill is True."
            assert isinstance(message_targets, MessageTargets), "message_targets must be an instance of MessageTargets."
            assert message_targets.target_response_toks is not None and message_targets.target_response_strs is not None, "message_targets must include target_response_toks and target_response_strs if require_target_prefill is True."

        # 1. Apply chat template (user turn only; generation prompt adds assistant role marker)
        assert isinstance(input_texts, list), "input_texts must be a list of strings."
        template_tok_ids = []
        for text in input_texts:
            template_tok_ids.append(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )["input_ids"]
            )

        # 2. Append prefill tokens to the prompt
        prefill_len = 0
        if require_target_prefill:
            prefill_len = message_targets.target_response_toks.shape[0]
            prefill_list = message_targets.target_response_toks.tolist()
            for prompt_toks in template_tok_ids:
                prompt_toks.extend(prefill_list)

        # 3. Pad and prep inputs
        assert self._tokenizer.padding_side == "left", "Tokenizer must use left padding for correct prefiling and generation. Please set `tokenizer.padding_side = 'left'`."
        inputs = self._tokenizer.pad(
            {"input_ids": template_tok_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        padded_seq_len = inputs.input_ids.shape[1]
        n_prompt_tokens = inputs.input_ids.numel()

        # ---- Shared forward pass (prefill logits and/or first-token logprobs) ----
        prefill_response_logits = None
        first_token_logprobs = None

        fwd_out = self._model(**inputs, use_cache=False)

        if require_target_prefill and prefill_len > 0:
            start = padded_seq_len - prefill_len - 1
            end   = padded_seq_len - 1
            prefill_response_logits = torch.stack(
                [fwd_out.logits[i, start:end] for i in range(len(input_texts))],
                dim=0,
            )  # (bsz, prefill_len, vocab_size)

        if require_first_token_logprobs:
            # Logits at the last real token predict the first response token.
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1  # (bsz,)
            batch_idx = torch.arange(len(input_texts), device=self.device)
            first_tok_logits = fwd_out.logits[batch_idx, last_token_indices]  # (bsz, vocab)
            first_tok_lps = torch.nn.functional.log_softmax(first_tok_logits, dim=-1)
            top_lps, top_ids = torch.topk(
                first_tok_lps, min(_MAX_TOP_LOGPROBS, first_tok_lps.shape[-1]), dim=-1
            )  # (bsz, top_k)
            first_token_logprobs = [
                {self._tokenizer.decode([top_ids[b, i].item()]): top_lps[b, i].item()
                 for i in range(top_ids.shape[-1])}
                for b in range(first_tok_lps.shape[0])
            ]

        # ---- Early return if generation not requested ----
        if not require_generation:
            self._update_invoke_stats(
                n_tokens=n_prompt_tokens,
                n_samples=len(input_texts),
            )
            return ModelOutput(
                prefill_response_logits=prefill_response_logits,
                response_first_token_logprobs=first_token_logprobs,
            )

        # ---- Generation ----
        hf_gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy_decode,
            "pad_token_id": self._tokenizer.pad_token_id,
            "output_logits": True,
            "return_dict_in_generate": True,
        }
        generation_output = self._model.generate(
            **inputs,
            **hf_gen_kwargs,
        )

        generation_logits = torch.stack(generation_output.logits, dim=1)  # (bsz, gen_seq_len, vocab_size)

        # Slice generated toks
        # generate()'s `.sequences` is a list of tensors of shape (bsz, padded_seq_len [incl. prefill]+ gen_len);
        full_toks = generation_output.sequences
        generated_toks = [full_toks[i][padded_seq_len:] for i in range(len(full_toks))]

        # Post-processing & stats
        generation_strs = self._tokenizer.batch_decode(
            generated_toks,
            skip_special_tokens=True,
        )
        n_gen_tokens = sum(len(t) for t in generated_toks)

        self._update_invoke_stats(
            n_tokens=n_prompt_tokens + n_gen_tokens,
            n_samples=len(generated_toks),
        )

        full_strs = self._tokenizer.batch_decode(
            full_toks,
            skip_special_tokens=False,
        )

        return ModelOutput(
            prefill_response_logits=prefill_response_logits,
            generated_response_strs=generation_strs,
            generated_response_ids=generated_toks,
            generated_response_logits=generation_logits,
            full_strs=full_strs,
            full_ids=full_toks,
            response_first_token_logprobs=first_token_logprobs,
        )

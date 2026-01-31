import itertools
import logging
from functools import cached_property
from typing import List, Optional, Tuple

import torch
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from tropt.common import DEFAULT_INIT_TRIGGER, OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import (
    AttentionBasedLoss,
    BaseLoss,
    CombinedLoss,
    HiddenStateBased,
    LogitBasedLoss,
    SteeringActivationLoss,
    TriggerLogitBasedLoss,
)
from tropt.models import (
    GradientTokenAccessMixin,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
    MessageBatchedTargetsDict,
    TargetsDict,
    TargetsDictPlus,
)
from tropt.models.huggingface.base import _HFTokenInputsManager, _HuggingFaceModelMixins

logger = logging.getLogger(__name__)


# ======================= Input/Output Handlers logic =======================
class LMHFTokenInputsManager(_HFTokenInputsManager):
    targets: TargetsDictPlus | TargetsDict
    # includes `target_outputs_toks` (n_messages, target_seq_len) if target outputs are provided;
    # to optimize towards an output per message

    @property
    def _do_prefill_targets(self) -> bool:
        return "target_outputs_toks" in self.targets

    @cached_property
    def _prefill_embeds(self) -> List[Float[Tensor, "target_seq_len embd_dim"]]:
        if self._do_prefill_targets:
            return [self.embed_func(target_output) for target_output in self.targets["target_outputs_toks"]]
        return None

    def get_triggered_inputs(self, *args, **kwargs):
        assert (
            kwargs.get("append_embeds", None) is None
        ), "append_embeds should not be passed directly to LM models. Use `target_embeds` property instead."

        return super().get_triggered_inputs(
            *args,
            **kwargs,
            append_embeds=self._prefill_embeds if self._do_prefill_targets else None,
        )


# ======================= Model logic =======================

class LMHFModel(
    LMBaseModel,
    # adds implementation of common HF model methods
    _HuggingFaceModelMixins,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    LogitsTokenAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    def __init__(
        self,
        model_name: str,
        device: str = None,
        dtype: str = None,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 32,
        # more args:
        use_prefix_cache: bool = True,
        set_model_to_eval: bool = True,
        use_eager_attention: bool = False,
        **model_kwargs,  # to be handed to HuggingFace model init
    ):
        self.model_name = model_name
        self.forward_pass_batch_size = forward_pass_batch_size
        self.backward_pass_batch_size = backward_pass_batch_size

        if use_eager_attention:
            # required for to support attention-based losses
            model_kwargs["attn_implementation"] = "eager"
            logger.info(
                f"Using eager attention for model {model_name} to support attention-based loss."
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device or "auto",
            dtype=dtype or "auto",
            **model_kwargs
        )
        self.dtype = self.model.dtype
        logger.info(f"Loaded model {model_name} on device {self.device}, with dtype {self.dtype}.")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_layer = self.model.get_input_embeddings()
        self.use_prefix_cache = use_prefix_cache

        # Set model to eval mode
        if set_model_to_eval:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # To make sure the placeholder will be tokenizer as is
        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        ## warning and checks:
        if self.model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {self.model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if self.model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not self._tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            self._tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )
        if self._tokenizer.padding_side != "left":
            logger.warning(
                "Tokenizer padding side is not 'left'. Our code currenly assume left padding ."
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

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def device(self):
        return self.model.device

    def prepare_token_inputs(
        self,
        texts: List[str],
        targets: TargetsDict | TargetsDictPlus,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
    ) -> Tuple[LMHFTokenInputsManager, Int[Tensor, "1 trigger_seq_len"]]:
        """
        Prepares the inputs for the model, including tokenization and target processing.
        """
        # To make sure the placeholder will be tokenizer as is
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        assert isinstance(texts, list) and all(isinstance(t, str) for t in texts), "texts must be a string or a list of strings."
        assert all(
            [t.count(OPTIMIZED_TRIGGER_PLACEHOLDER) == 1 for t in texts]
        ), f"`texts` must contain the `{OPTIMIZED_TRIGGER_PLACEHOLDER}` placeholder."

        # put in chat template + special tokens & tokenizer
        template_tok_ids: List[List[int]] = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for text in texts
        ]

        # Encode target outputs, if provided
        if "target_outputs" in targets:
            tokenized_lists = self.tokenizer(
                targets["target_outputs"], add_special_tokens=False
            )["input_ids"]
            # convert to list of tensors
            targets["target_outputs_toks"] = [
                torch.tensor(ids, device=self.model.device) for ids in tokenized_lists
                # each of shape (target_seq_len,)
            ]

        # Build the input manager, that will allow combining with different triggers
        inputs = LMHFTokenInputsManager(
            tok_ids=template_tok_ids,
            model=self.model,
            tokenizer=self.tokenizer,
            embed_func=self.embedding_layer,
            optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
            use_prefix_cache=self.use_prefix_cache,
            targets=targets,
        )

        # Tokenizer trigger
        if not initial_trigger:
            # start with an empty trigger
            trigger_ids = torch.zeros((1, 0), dtype=torch.long, device=self.model.device)
        else:
            trigger_ids = (
                self.tokenizer.encode(
                    initial_trigger, add_special_tokens=False, return_tensors="pt"
                )
                .to(self.model.device, torch.int64)
            )

        return inputs, trigger_ids

    @torch.no_grad()
    def compute_logits_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        inputs: LMHFTokenInputsManager,
        keep_message_dim: bool = False,
        return_trigger_logits_only: bool = False,
        return_after_trigger_logits_only: bool = False,
    ) -> (
        Float[Tensor, "n_messages n_candidates seq_len vocab_size"]
        | Tuple[
            Float[Tensor, "n_messages n_candidates seq_len vocab_size"], List[slice]
        ]
    ):
        """
        Given a batch of candidate trigger token ids and inputs object, returns the logits for the next token after the input sequence (i.e., after the trigger + input text + target text, if provided).

        Args:
            candidate_trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                the token ids of the candidate trigger sequences to evaluate
            inputs: LMHFTokenInputsManager
                the inputs object containing the input text and target text (if provided)
            return_slices: bool
                whether to return the slices corresponding to each input in the batch (default: False)
            keep_message_dim: bool
                whether to keep the message dimension in the output logits (default: False)
            return_trigger_logits_only: bool
                whether to return only the logits corresponding to the trigger tokens (default: False)
            return_after_trigger_logits_only: bool
                whether to return only the logits corresponding to the final token of the trigger (default: False)
        """
        assert int(return_trigger_logits_only) + int(return_after_trigger_logits_only) <= 1, "Cannot set both `return_trigger_logits_only` and `return_after_trigger_logits_only` to True."

        ## TODO-2 the following logic should call `get_triggered_inputs()` with chosen_message_idx, as multi-message input is not supported here yet! For iplementating this, inspect the shapes returned by `get_triggered_inputs()` in the case of single message input, and adapt the following code accordingly (ie aggregate the required tensors correctly).
        ## START OF PREPARATION ##
        # Get the inputs with the candidate triggers inserted
        inputs_embeds_dict = inputs.get_triggered_inputs(
            trigger_ids=candidate_trigger_ids
            # chosen_message_idx=chosen_message_idx,  # TODO see note above
        )
        inputs_embeds, attention_mask, slices = (
            inputs_embeds_dict["inputs_embeds"],
            inputs_embeds_dict["attention_mask"],
            inputs_embeds_dict["targets"]["slices"], # n_messages lists of length n_candidates
        )
        n_messages, n_candidates = inputs_embeds.shape[:2]

        # Flatten first two dims: (M, C, ...) -> (M*C, ...)
        inputs_embeds = inputs_embeds.reshape(-1, *inputs_embeds.shape[2:])
        attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:])

        ## END OF PREPARATION ##

        # Compute the logits (in batches)
        @find_executable_batch_size(starting_batch_size=self.forward_pass_batch_size)
        def _compute_logits_batched(batch_size):
            n_samples = inputs_embeds.shape[0]
            logit_chunks = []

            # Process in chunks
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                # Get input batch
                inp_slice = inputs_embeds[i:end_i]
                attn_slice = attention_mask[i:end_i] if attention_mask is not None else None
                # Forward pass
                logits_slice = self.model(
                    inputs_embeds=inp_slice,
                    attention_mask=attn_slice,
                ).logits
                # Collect logits
                logit_chunks.append(logits_slice)

            # 3. Reassemble
            return torch.cat(logit_chunks, dim=0)
        logits = _compute_logits_batched()
        # (n_messages * n_candidates, seq_len, vocab_size)

        # un-flatten (n_messages * n_candidates, ..) -> (n_messages, n_candidates, ..)
        logits = logits.reshape(n_messages, n_candidates, *logits.shape[1:])

        if return_trigger_logits_only or return_after_trigger_logits_only:
            # return only the logits for the trigger part
            trigger_logits = torch.zeros(
                (n_messages, n_candidates, candidate_trigger_ids.shape[1] if return_trigger_logits_only else 1, logits.shape[-1]),
                device=logits.device,
            )  # (n_messages, n_candidates, trigger_seq_len, vocab_size)
            for i_message, i_cand in itertools.product(
                range(n_messages), range(n_candidates)
            ):
                slc_trigger = slices[i_message][i_cand]["adv"]  # trigger slice for this candidate
                if return_trigger_logits_only:
                    slc = slice(slc_trigger.start - 1, slc_trigger.stop)
                    assert slc.stop - slc.start == candidate_trigger_ids.shape[1], "Trigger slice length does not match candidate trigger length."
                else:  # return_after_trigger_logits_only
                    slc = slice(slc_trigger.stop, slc_trigger.stop + 1)
                trigger_logits[i_message, i_cand] = logits[i_message, i_cand, slc, :]
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

    def _loss_hook(
        self,
        inputs_embeds: Float[Tensor, "bsz seq_len embd_dim"],
        attention_mask: Float[Tensor, "bsz seq_len"],
        targets: MessageBatchedTargetsDict,
        loss_func: BaseLoss,
        prefix_cache_kwargs: dict = {},
        trigger_ids: Int[Tensor, "bsz trigger_seq_len"] = None,
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        """
        Hook for computing the loss on the given inputs, which are for *specific message* (for the inputs to be aligned).
        """
        if loss_func.contains_loss_type(AttentionBasedLoss) and self.model.config._attn_implementation != "eager":
            logger.warning(
                "AttentionBasedLoss is used but the model is not using eager attention. "
                "This may lead to incorrect attention outputs. Consider initializing the model with eager attention, by passing LMHFModel the flag `use_eager_attention=True`."
            )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=loss_func.contains_loss_type(AttentionBasedLoss),
            output_hidden_states=loss_func.contains_loss_type(HiddenStateBased),
            **prefix_cache_kwargs,
        )

        def _calc_loss_from_outputs(_outputs, _targets, _trigger_ids, _loss_func):
            if isinstance(_loss_func, LogitBasedLoss):
                logits = _outputs.logits
                response_target_ids = _targets["target_outputs_toks"]  # (bsz, target_seq_len)
                response_slcs = [slices['appended'] for slices in _targets["slices"]]  # bsz of `slice`

                # Check if slices are aligned across the batch
                first_slc = response_slcs[0]
                are_slcs_aligned = all(
                    s.start == first_slc.start and s.stop == first_slc.stop
                    for s in response_slcs
                )

                assert isinstance(response_target_ids, torch.Tensor) and response_target_ids.dim() == 2 and response_target_ids.shape[0] == logits.shape[0], \
                    "response_target_ids must be a tensor of shape (bsz, target_seq_len) matching the batch size of logits."
                assert first_slc.stop - first_slc.start == response_target_ids.shape[1], \
                    "Length of target sequences must match the length of the response slices."
                assert are_slcs_aligned, "Response slices are not aligned across the batch. Variable-length target sequences are not supported yet."

                # If slices are aligned, we can simply stack them
                start_idx = first_slc.start - 1
                end_idx = first_slc.stop - 1
                response_logits = logits[:, start_idx:end_idx, :]

                # Compute loss
                loss = _loss_func(
                    response_logits,
                    response_target_ids,
                )  # shape: (bsz,)

            elif isinstance(_loss_func, TriggerLogitBasedLoss):
                assert _trigger_ids is not None, "trigger_ids must be provided for TriggerLogitBasedLoss losses."
                logits = _outputs.logits
                trigger_slcs = [slices['adv'] for slices in _targets['slices']]  # bsz of `slice`

                # Check if slices are aligned across the batch
                first_slc = trigger_slcs[0]
                are_slcs_aligned = all(
                    s.start == first_slc.start and s.stop == first_slc.stop
                    for s in trigger_slcs
                )
                assert are_slcs_aligned, "Trigger slices are not aligned across the batch. Variable-length trigger sequences are not supported yet."
                assert first_slc.start >= 1, "Trigger slices should start at least one position (for feasible logits). It could be that prefix-caching is enabled and causing this; if so, disable prefix caching."

                # If slices are aligned, we can simply stack them
                start_idx = first_slc.start - 1
                end_idx = first_slc.stop - 1
                trigger_logits = logits[:, start_idx:end_idx, :]
                assert _trigger_ids.shape[1] == trigger_logits.shape[1], "Trigger ids length must match the length of the trigger slices."

                # Compute loss
                loss = _loss_func(
                    trigger_logits,
                    _trigger_ids,
                )  # shape: (bsz,)

            elif isinstance(_loss_func, AttentionBasedLoss):
                attentions = torch.stack(
                    _outputs.attentions, dim=1
                )  # (bsz, n_layers, n_heads, seq_len[dst], seq_len[src])
                loss = _loss_func(
                    attentions,
                    slices=_targets["slices"],
                )  # shape: (bsz,)

            elif isinstance(_loss_func, SteeringActivationLoss):
                hidden_states = torch.stack(
                    _outputs.hidden_states, dim=1
                )  # (bsz, n_layers, seq_len, embd_dim)

                target_directions = _targets["target_directions"]  # (bsz, d_model)

                loss = _loss_func(
                    hidden_states,
                    target_directions=target_directions,
                    slices=_targets["slices"],
                )  # shape: (bsz,)

            elif isinstance(_loss_func, CombinedLoss):
                losses = []
                for _nested_loss_func in _loss_func:
                    # Recursive call to compute each loss
                    losses.append(
                        _calc_loss_from_outputs(_outputs, _targets, _trigger_ids, _nested_loss_func)
                    ) # shape: (bsz,)

                # combine the losses (by calling the loss function on them)
                losses = torch.stack(losses, dim=0)  # (n_losses, bsz)
                loss = _loss_func(losses)  # shape: (bsz,)

            else:
                raise NotImplementedError(
                    f"Loss function {loss_func} not supported for HuggingFace models yet."
                )
            return loss

        return _calc_loss_from_outputs(outputs, targets, trigger_ids, loss_func)

    @torch.no_grad()
    def __call__(
        self,
        texts: List[str],
        greedy_decode: bool = True,
        max_new_tokens: int = 128,
        return_full_template: bool = False,
    ) -> List[str]:
        """Get the embeddings for the given texts."""
        # TODO this doubles the BOS - resolve!  -> TODO-claude-code -- make sure it's fixed and the logic of the tokenization here works across MULTIPLE modles (eg gemma, qwen, llama, etc)
        assert isinstance(texts, list), "texts must be a string or a list of strings."

        # Add chat template and tokenize
        template_tok_ids: List[List[int]] = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for text in texts
        ]

        # Use tokenizer's pad method for cleaner handling (avoids double BOS)
        inputs = self.tokenizer.pad(
            {"input_ids": template_tok_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Keep prompt lengths
        prompt_lengths = [len(toks) for toks in inputs.input_ids]

        # Generate responses
        generation_toks = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy_decode,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if not return_full_template:
            # Extract only the generated part
            generation_toks = [
                toks[prompt_lengths[i]:] for i, toks in enumerate(generation_toks)
            ]

        # Decode to strings
        generation_strs = self.tokenizer.batch_decode(
            generation_toks,
            skip_special_tokens=not return_full_template
        )

        # Track usage
        prompt_tokens = inputs.input_ids.numel()
        gen_tokens = sum(len(t) for t in generation_toks)
        self._update_usage_stats(
            tokens=prompt_tokens + gen_tokens,
            forward_calls=1,
            forward_samples=len(texts)
        )

        return generation_strs

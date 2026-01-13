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

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import AttentionBasedLoss, BaseLoss, CombinedLoss, LogitBasedLoss
from tropt.models.base import (
    GradientTokenAccessMixin,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
    MessageBatchedTargetsDict,
    TargetsDict,
    TargetsDictPlus,
)
from tropt.models.huggingface.base import HFTokenInputsManager, HuggingFaceModelMixins

logger = logging.getLogger(__name__)


# ======================= Input/Output Handlers logic =======================
class LMHFInputsManager(HFTokenInputsManager):
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

# Some models (e.g., gemma) require eager attention to enable some losses  # TODO should decouple such exceptions / let user control/override them
MODELS_TO_EAGER_ATTENTION = ["gemma"]


class LMHFModel(
    LMBaseModel,
    # adds implementation of common HF model methods
    HuggingFaceModelMixins,
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
        use_prefix_cache: bool = True,  # TODO make sure this works as intended
        set_model_to_eval: bool = True,
        **model_kwargs,  # to be handed to HuggingFace model init
    ):
        self.model_name = model_name
        self.forward_pass_batch_size = forward_pass_batch_size
        self.backward_pass_batch_size = backward_pass_batch_size

        if any(m in model_name.lower() for m in MODELS_TO_EAGER_ATTENTION):
            # TODO this sub-optimal option should be triggered only if an attention-based loss is used
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
        self.device = self.model.device
        self.dtype = self.model.dtype
        logger.info(f"Loaded model {model_name} on device {self.device}, with dtype {self.dtype}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_layer = self.model.get_input_embeddings()
        self.use_prefix_cache = use_prefix_cache

        # Set model to eval mode
        if set_model_to_eval:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # To make sure the placeholder will be tokenizer as is
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        ## warning and checks:
        if self.model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {self.model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if self.model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not self.tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            self.tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )
        if self.tokenizer.padding_side != "left":
            logger.warning(
                "Tokenizer padding side is not 'left'. Our code currenly assume left padding ."
            )
            self.tokenizer.padding_side = "left"

        if not self.tokenizer.pad_token:
            if self.tokenizer.eos_token:
                logger.warning(
                    "Tokenizer does not have a pad token. Setting pad token to eos token."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer does not have a pad token or an eos token. Please set a pad token."
                )

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def prepare_token_inputs(
        self,
        texts: List[str],
        targets: TargetsDict | TargetsDictPlus,
        initial_trigger: Optional[str] = "! " * 20,
    ) -> Tuple[LMHFInputsManager, Int[Tensor, "1 trigger_seq_len"]]:
        """
        Prepares the inputs for the model, including tokenization and target processing.
        """
        # To make sure the placeholder will be tokenizer as is
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        assert isinstance(texts, list), "texts must be a string or a list of strings."
        n_messages = len(texts)
        targets = TargetsDictPlus(targets, n_messages=n_messages)

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
        inputs = LMHFInputsManager(
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
        inputs: LMHFInputsManager,
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
            inputs: LMHFInputsManager
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

        # Get the inputs with the candidate triggers inserted
        inputs_embeds_dict = inputs.get_triggered_inputs(
            trigger_ids=candidate_trigger_ids
            # TODO: choose get input per message
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
                    slc = slc_trigger
                else:  # return_after_trigger_logits_only
                    slc = slice(slc_trigger.stop, slc_trigger.stop + 1)
                trigger_logits[i_message, i_cand] = logits[i_message, i_cand, slc, :]
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
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        """
        Hook for computing the loss on the given inputs, which are for *specific message* (for the inputs to be aligned).
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=loss_func.contains_loss_type(AttentionBasedLoss),
            **prefix_cache_kwargs,
        )
        # hidden_states = outputs.hidden_states [TODO]

        def _calc_loss_from_outputs(_outputs, _targets, _loss_func):
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

                ## [DISABLED] Currently not supported: variable-length target sequences
                ## Otherwise, we fallback to padding of the variable lengths
                # if not are_slcs_aligned:
                # response_logits = [
                #     logits[i, response_slcs[i].start - 1 : response_slcs[i].stop - 1, :]
                #     for i in range(logits.shape[0])
                # ]  # bsz of (target_seq_len[i], vocab_size)
                # # Pad logits to the same length
                # response_logits = pad_sequence(
                #     response_logits, batch_first=True, padding_value=0.0
                # )
                # # Pad targets and place -100 where we have padding (to ignore in loss)
                # response_target_ids = pad_sequence(
                #     response_target_ids, batch_first=True, padding_value=-100
                # )

                # Compute loss
                loss = _loss_func(
                    response_logits,
                    response_target_ids,
                )  # shape: (bsz,)

            elif isinstance(_loss_func, AttentionBasedLoss):
                attentions = torch.stack(
                    _outputs.attentions, dim=1
                )  # (bsz, n_layers, n_heads, seq_len[dst], seq_len[src])
                loss = _loss_func(
                    attentions,
                    slices=_targets['slices'],  # optionally contains slices
                )  # shape: (bsz,)

            elif isinstance(_loss_func, CombinedLoss):
                losses = []
                for _nested_loss_func in _loss_func.loss_funcs:
                    # Recursive call to compute each loss
                    losses.append(
                        _calc_loss_from_outputs(_outputs, _targets, _nested_loss_func)
                    ) # shape: (bsz,)

                # combine the losses (by calling the loss function on them)
                losses = torch.stack(losses, dim=0)  # (n_losses, bsz)
                loss = _loss_func(losses)  # shape: (bsz,)

            else:
                raise NotImplementedError(
                    f"Loss function {loss_func} not supported for HuggingFace models yet."
                )
            return loss

        return _calc_loss_from_outputs(outputs, targets, loss_func)

    @torch.no_grad()
    def __call__(
        self,
        texts: List[str],
        greedy_decode: bool = True,
        max_new_tokens: int = 128,
        return_full_template: bool = False,
    ) -> List[str]:
        """Get the embeddings for the given texts."""
        # TODO this doubles the BOS - resolve!
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

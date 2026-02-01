from __future__ import annotations

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
from tropt.models.inputs import SliceKey, TextInputsManager

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

        n_messages = inputs.n_messages
        n_candidates, trigger_seq_len = candidate_trigger_ids.shape

        # Compute the logits (in batches)
        @find_executable_batch_size(starting_batch_size=self.forward_pass_batch_size)
        def _compute_logits_batched(batch_size):
            all_logits = [[] for _ in range(n_messages)]
            all_slices = [[] for _ in range(n_messages)]

            for message_idx, cand_idx in itertools.product(
                range(n_messages),
                range(0, n_candidates, batch_size),
            ):
                cand_idx_end = min(cand_idx + batch_size, n_candidates)
                batch_candidate_trigger_ids = candidate_trigger_ids[cand_idx:cand_idx_end]

                # Get inputs for this specific message
                model_input = inputs.get_triggered_inputs(
                    trigger_ids=batch_candidate_trigger_ids,
                    chosen_message_idx=message_idx,
                )

                # Forward pass
                logits_batch = self.model(
                    inputs_embeds=model_input.input_embeds,
                    attention_mask=model_input.input_attention_mask,
                    **(model_input.input_prefix_cache_kwargs or {})
                ).logits  # (n_cand_batch, seq_len, vocab_size)

                all_logits[message_idx].append(logits_batch)
                all_slices[message_idx].extend(
                    inputs_dict["targets"]["slices"]  # list of n_cand_batch dicts
                )

            # Stack all logits per message
            logits_per_message = [torch.cat(msg_logits, dim=0) for msg_logits in all_logits]
            logits = torch.stack(logits_per_message, dim=0)  # (n_messages, n_candidates, seq_len, vocab_size)
            slices = all_slices

            return logits, slices

        logits, slices = _compute_logits_batched()
        # (n_messages, n_candidates, seq_len, vocab_size)

        if return_trigger_logits_only or return_after_trigger_logits_only:
            # return only the logits for the trigger part
            trigger_logits = torch.zeros(
                (n_messages, n_candidates, (trigger_seq_len if return_trigger_logits_only else 1), logits.shape[-1]),
                device=logits.device,
            )  # (n_messages, n_candidates, trigger_seq_len, vocab_size)
            for i_message, i_cand in itertools.product(
                range(n_messages), range(n_candidates)
            ):
                slc_trigger = slices[i_message][i_cand][SliceKey.TRIGGER]  # trigger slice for this candidate
                if return_trigger_logits_only:
                    slc = slice(slc_trigger.start - 1, slc_trigger.stop)
                    assert slc.stop - slc.start == trigger_seq_len, "Trigger slice length does not match candidate trigger length."
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

        # Create ModelOutput from HuggingFace forward pass
        from tropt.models.outputs import ModelOutput
        from tropt.models.inputs import ModelInput
        from tropt.loss.resolution import compute_loss_from_model_data

        model_output = ModelOutput(
            output_logits=outputs.logits,
            output_attentions=torch.stack(outputs.attentions, dim=1) if outputs.attentions else None,
            output_hidden_states=torch.stack(outputs.hidden_states, dim=1) if outputs.hidden_states else None,
        )

        model_input = ModelInput(
            input_trigger_ids=trigger_ids,
            input_slices=targets["slices"] if "slices" in targets else None,
            targets=targets,
        )

        # Use unified loss resolution
        return compute_loss_from_model_data(model_output, model_input, loss_func)

    @torch.no_grad()
    def __call__(
        self,
        texts: List[str],
        greedy_decode: bool = True,
        max_new_tokens: int = 128,
        return_full_output: bool = False,
    ) -> List[str] | "ModelOutput":
        """
        Generate text completions for the given input texts.
        """
        assert isinstance(texts, list), "texts must be a string or a list of strings."

        # Add chat template and tokenize
        # Note: apply_chat_template handles special tokens (BOS, EOS) according to the model's template
        template_tok_ids: List[List[int]] = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for text in texts
        ]

        # Use tokenizer's pad method for consistent padding behavior
        inputs = self.tokenizer.pad(
            {"input_ids": template_tok_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Keep prompt lengths
        prompt_lengths = [len(toks) for toks in inputs.input_ids]

        # Generate responses
        generation_output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy_decode,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=return_full_output,
            return_dict_in_generate=return_full_output,
        )

        if return_full_output:
            full_toks = generation_output.sequences
            generation_logits = torch.stack(generation_output.scores, dim=1)
        else:
            full_toks = generation_output

        # Extract only the generated part
        generated_toks = [
            full_toks[i][prompt_lengths[i]:] for i in range(len(full_toks))
        ]

        # Decode to strings
        generation_strs = self.tokenizer.batch_decode(
            generated_toks,
            skip_special_tokens=True
        )

        # Track usage
        prompt_tokens = inputs.input_ids.numel()
        gen_tokens = sum(len(t) for t in generated_toks)
        self._update_usage_stats(
            tokens=prompt_tokens + gen_tokens,
            forward_calls=1,
            forward_samples=len(texts)
        )

        if return_full_output:
            # Trim logits per-sample to match actual generated length
            # (scores are already prompt-excluded, but samples may differ due to EOS)
            generation_logits = [
                generation_logits[i, :len(generated_toks[i])]
                for i in range(len(generated_toks))
            ]

            full_strs = self.tokenizer.batch_decode(
                full_toks,
                skip_special_tokens=False
            )

            from tropt.models.outputs import ModelOutput
            return ModelOutput(
                generated_response_strs=generation_strs,
                generated_response_ids=generated_toks,
                generated_response_logits=generation_logits,
                full_template_strs=full_strs,
                full_template_ids=full_toks,
            )

        return generation_strs

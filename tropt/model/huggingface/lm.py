import itertools
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelInput,
    ModelOutput,
    SliceKey,
    Targets,
    TextTemplates,
)
from tropt.loss import (
    AttentionBasedLoss,
    BaseLoss,
    HiddenStateBasedLoss,
    LogitBasedLoss,
)
from tropt.model import (
    GradientTokenAccessMixin,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.model.huggingface.base import _HFTokenInputManager, _HuggingFaceModelMixins
from tropt.model.model_mixins import GradientEmbedAccessMixin

logger = logging.getLogger(__name__)


# ======================= Input/Output Handlers logic =======================
class LMHFTokenInputManager(_HFTokenInputManager):
    targets: Targets
    # includes `target_response_toks` (n_templates, target_seq_len) if target outputs are provided;
    # to optimize towards an output per message

    @property
    def _do_prefill_targets(self) -> bool:
        return self.targets.target_response_toks is not None

    @cached_property
    def _prefill_embeds(self) -> List[Float[Tensor, "target_seq_len embd_dim"]]:
        if self._do_prefill_targets:
            return [self.embed_func(target_output) for target_output in self.targets.target_response_toks]
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
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    def __init__(
        self,
        model_name: str,
        device: str = None,
        dtype: str = None,
        _forward_pass_batch_size: int = 512,
        _backward_pass_batch_size: int = 32,
        # more args:
        use_prefix_cache: bool = True,
        set_model_to_eval: bool = True,
        use_eager_attention: bool = False,
        **model_kwargs,  # to be handed to HuggingFace model init
    ):
        self._model_name = model_name
        self._forward_pass_batch_size = _forward_pass_batch_size
        self._backward_pass_batch_size = _backward_pass_batch_size

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
        self.dtype = self._model.dtype
        logger.info(f"Loaded model {model_name} on device {self.device}, with dtype {self.dtype}.")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._embedding_layer = self._model.get_input_embeddings()
        self._use_prefix_cache = use_prefix_cache

        # Set model to eval mode
        if set_model_to_eval:
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

        # To make sure the placeholder will be tokenizer as is
        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        ## warning and checks:
        if self._model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {self._model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if self._model.device == torch.device("cpu"):
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
                "Tokenizer padding side is not 'left'. Our code currently assumes left padding."
            )
            # TODO is it true that we need it? where do we assume it?? maybe it's not needed anymore?
            # !!!!!!!!!!!!!!!!!!!!!!!!
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
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._model.device

    # ======================= Token-access methods =======================

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """
        Prepares and stores the inputs manager for the model, including tokenization and target processing.
        """
        # To make sure the placeholder will be tokenizer as is
        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

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
            )
            for template in templates
        ]

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

        # Build the input manager, that will allow combining with different triggers
        self._token_input_manager = LMHFTokenInputManager(
            tok_ids=template_tok_ids,
            model=self._model,
            tokenizer=self._tokenizer,
            embed_func=self._embedding_layer,
            optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
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
            all_logits = [[] for _ in range(n_templates)]
            slices: List[Dict[str, slice]] = [None for _ in range(n_templates)]

            for template_idx, cand_idx in itertools.product(
                range(n_templates),
                range(0, n_candidates, batch_size),
            ):
                cand_idx_end = min(cand_idx + batch_size, n_candidates)
                batch_candidate_trigger_ids = candidate_trigger_ids[cand_idx:cand_idx_end]

                # Get inputs for this specific message
                model_input = input_manager.get_triggered_inputs(
                    trigger_ids=batch_candidate_trigger_ids,
                    chosen_template_idx=template_idx,
                )
                # Compute the logits
                logits_batch = self.invoke_from_tokens(
                    **model_input.to_dict(),
                ).output_logits

                all_logits[template_idx].append(logits_batch)
                slices[template_idx] = model_input.input_slices

            # Stack all logits per message
            logits_per_message = [torch.cat(msg_logits, dim=0) for msg_logits in all_logits]
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
        input_embeds: Float[Tensor, "bsz seq_len embd_dim"],
        input_attention_mask: Float[Tensor, "bsz seq_len"],
        input_prefix_cache_kwargs: Optional[Dict[str, Any]] = None,
        input_slices: Optional[Dict[str, slice]] = None,
        reference_loss_func: BaseLoss = None,
    ) -> ModelOutput:
        """
        Performs a forward pass through the model given input embeddings and attention mask.

        Args:
            input_embeds: Input embeddings tensor of shape (bsz, seq_len, embd_dim).
            input_attention_mask: Attention mask tensor of shape (bsz, seq_len).
            input_prefix_cache_kwargs: Optional dict of prefix cache kwargs to pass to the model.
            input_slices: Optional dict mapping slice keys to slices for extracting specific parts of the output.
            reference_loss_func: Optional loss function used to determine which outputs to compute
                (e.g., attentions for AttentionBasedLoss, hidden states for HiddenStateBasedLoss).

        Returns:
            ModelOutput: The output of the model containing logits, hidden states, and attentions as applicable.
        """
        if reference_loss_func is not None and reference_loss_func.contains_loss_type(AttentionBasedLoss) and self._model.config._attn_implementation != "eager":
            logger.warning(
                "AttentionBasedLoss is used but the model is not using eager attention. "
                "This may lead to incorrect attention outputs. Consider initializing the model with eager attention, by passing LMHFModel the flag `use_eager_attention=True`."
            )

        assert input_embeds is not None, "input_embeds must be provided in HF's invoke_from_tokens."

        outputs = self._model(
            inputs_embeds=input_embeds,
            attention_mask=input_attention_mask,
            output_attentions=reference_loss_func.contains_loss_type(AttentionBasedLoss) if reference_loss_func else False,
            output_hidden_states=reference_loss_func.contains_loss_type(HiddenStateBasedLoss) if reference_loss_func else False,
            **(input_prefix_cache_kwargs or {})
        )
        self._update_usage_stats(
            forward_calls=1,
            forward_samples=input_embeds.shape[0],
            tokens=input_attention_mask.sum().item(),
        )

        response_logits = None
        if reference_loss_func is not None and reference_loss_func.contains_loss_type(LogitBasedLoss):
            response_slc = input_slices[SliceKey.APPENDED]
            response_logits = outputs.logits[:, response_slc.start - 1 : response_slc.stop - 1, :]  # (bsz, response_seq_len, vocab_size)

        return ModelOutput(
            output_logits=outputs.logits,
            response_logits=response_logits,
            output_attentions=torch.stack(outputs.attentions, dim=1) if outputs.attentions else None,
            output_hidden_states=torch.stack(outputs.hidden_states[1:], dim=1) if outputs.hidden_states else None,  # (skips input embedding (layer 0)
        )

    # ======================= Text-access methods =======================

    def invoke_from_texts(
        self,
        input_texts: Optional[List[str]] = None,

        # [Optional] Embedding input:
        inputs_embeds: Optional[Float[Tensor, "bsz seq_len embd_dim"]] = None,
        attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,

        greedy_decode: bool = True,
        max_new_tokens: int = 128,
    ) -> ModelOutput:
        """
        Generate text completions. Always returns a ModelOutput.

        Accepts either plain texts or input embeddings.

        Args:
            input_texts: list of plain-text prompts.  Mutually exclusive with ``inputs_embeds``.
            inputs_embeds: pre-built prompt embeddings (bsz, seq_len, embd_dim).
                Note that in the case of input_embedding the full_template_strs and full_template_ids will not be returned in the output, as we don't have access to the text/tokenized input.
            attention_mask: Only relevant if ``inputs_embeds`` is provided. Attention mask matching ``inputs_embeds``.
        """
        assert (input_texts is None) ^ (inputs_embeds is None), \
            "Exactly one of `input_texts` or `inputs_embeds` must be provided."

        hf_gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy_decode,
            "pad_token_id": self._tokenizer.pad_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        if inputs_embeds is not None:
            # --- Embed flow ----------------------------------
            n = inputs_embeds.shape[0]
            generation_output = self._model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **hf_gen_kwargs
            )

            full_toks = generation_output.sequences
            generation_logits = torch.stack(generation_output.scores, dim=1)

            # HF returns only generated token IDs when inputs_embeds is used
            generated_toks = [full_toks[i] for i in range(n)]
            n_prompt_tokens = inputs_embeds.shape[0] * inputs_embeds.shape[1]

        else:
            # --- Text flow ----------------------------------
            # Note: apply_chat_template handles special tokens (BOS, EOS) according to the model's template
            assert isinstance(input_texts, list), "input_texts must be a list of strings."
            template_tok_ids: List[List[int]] = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for text in input_texts
            ]

            inputs = self._tokenizer.pad(
                {"input_ids": template_tok_ids},
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            prompt_lengths = [len(toks) for toks in inputs.input_ids]

            generation_output = self._model.generate(
                **inputs,
                **hf_gen_kwargs
            )

            full_toks = generation_output.sequences
            generation_logits = torch.stack(generation_output.scores, dim=1)

            generated_toks = [
                full_toks[i][prompt_lengths[i]:] for i in range(len(full_toks))
            ]
            n_prompt_tokens = inputs.input_ids.numel()

        # --- Shared post-processing ------------------------------------
        generation_strs = self._tokenizer.batch_decode(
            generated_toks,
            skip_special_tokens=True
        )

        n_gen_tokens = sum(len(t) for t in generated_toks)
        self._update_usage_stats(
            tokens=n_prompt_tokens + n_gen_tokens,
            forward_calls=1,
            forward_samples=len(generated_toks),
        )

        # Trim logits per-sample to match actual generated length
        # (scores are already prompt-excluded, but samples may differ due to EOS)
        generation_logits = [
            generation_logits[i, :len(generated_toks[i])]
            for i in range(len(generated_toks))
        ]

        if inputs_embeds is None:
            full_strs = self._tokenizer.batch_decode(full_toks, skip_special_tokens=False)
            full_toks_out = full_toks
        else:
            # Prompt was given as embeddings; no prompt token IDs to reconstruct
            full_strs = None
            full_toks_out = None

        return ModelOutput(
            generated_response_strs=generation_strs,
            generated_response_ids=generated_toks,
            generated_response_logits=generation_logits,
            full_template_strs=full_strs,
            full_template_ids=full_toks_out,
        )

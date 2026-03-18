import itertools
import logging
from abc import abstractmethod
from functools import cached_property
from typing import Annotated, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from accelerate.utils.memory import clear_device_cache, find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelInput,
    ModelOutput,
    SliceKey,
    Targets,
)
from tropt.loss import BaseLoss
from tropt.loss.resolution import resolve_and_compute_loss
from tropt.model import (
    TokenInputManager,
)

logger = logging.getLogger(__name__)

# ======================= Input/Output Handlers logic =======================


class _HFTokenInputManager(TokenInputManager):
    before_ids: Annotated[List[Float[Tensor, "bef_len"]], "n_templates"]
    after_ids: Annotated[List[Float[Tensor, "aft_len"]], "n_templates"]
    embed_func: torch.nn.Embedding
    targets: Targets
    padding_side: str
    pad_token_id: int
    tokenizer: transformers.PreTrainedTokenizer

    # Optional prefix cache (for models that support it)
    prefix_cache: Optional[List[tuple]] = None

    @torch.no_grad()
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        tok_ids: List[List[int]],
        embed_func: torch.nn.Embedding,
        optimized_trigger_placeholder: Optional[str] = OPTIMIZED_TRIGGER_PLACEHOLDER,
        use_prefix_cache: Optional[bool] = False,
        targets: Targets = None,
    ):
        self.padding_side = tokenizer.padding_side
        self.pad_token_id = tokenizer.pad_token_id

        # Split texts into before/after optimized trigger parts
        before_texts, after_texts = [], []
        for text in tokenizer.batch_decode(tok_ids):
            bef, aft = text.split(optimized_trigger_placeholder)
            before_texts.append(bef)
            after_texts.append(aft)

        # Tokenize & Tensorize everything
        # We save the tensor ids in lists, as they may have different lengths
        before_ids = tokenizer(before_texts, add_special_tokens=False)["input_ids"]
        before_ids = [
            torch.tensor(ids, device=model.device, dtype=torch.int64)
            for ids in before_ids
        ]
        after_ids = tokenizer(after_texts, add_special_tokens=False)["input_ids"]
        after_ids = [
            torch.tensor(ids, device=model.device, dtype=torch.int64)
            for ids in after_ids
        ]

        self.before_ids = before_ids
        self.before_texts = before_texts
        self.after_ids = after_ids
        self.after_texts = after_texts
        self.embed_func = embed_func
        self.tokenizer = tokenizer

        # Prepare targets
        self.targets = targets.to_device(model.device)

        # Compute the KV Cache for tokens that appear before the optimized tokens
        prefix_cache: List[ # per message
            Tuple[ # n_layers of these:
                Tuple[
                    Float[Tensor, "1 n_head seq_len head_dim"],  # keys
                    Float[Tensor, "1 n_head seq_len head_dim"],  # values
                ]
            ]
        ] = []

        if use_prefix_cache:
            for i in range(self.n_templates):
                # (seq, emb) -> (1, seq, emb)
                curr_embeds = self.before_embeds[i].unsqueeze(0)
                curr_attn_mask = torch.ones(
                    curr_embeds.shape[:2], device=model.device, dtype=torch.int64
                )
                output = model(
                    inputs_embeds=curr_embeds,
                    attention_mask=curr_attn_mask,
                    use_cache=True,
                )
                curr_prefix = output.past_key_values.to_legacy_cache()
                # tuple(layers) of tuple(k, v) where k,v are (1, n_head, seq_len, head_dim)
                prefix_cache.append(curr_prefix)

        self.prefix_cache = prefix_cache if use_prefix_cache else None
        # Memory for formatted prefix cache kwargs, keyed by (batch_size, template_idx)
        self._prefix_cache_kwargs_mem: dict = {}

    @property
    def vocab_size(self):
        # TODO some models might have slightly different effecive vocab size in weight (?)
        #      it's possible that each caller need to receive different vocab size; go over these.
        return self.tokenizer.vocab_size

    @property
    def n_templates(self):
        """Number of strored templates/messages."""
        return len(self.before_ids)

    @property
    def device(self):
        return self.before_ids[0].device

    @property
    def float_dtype(self):
        return self.before_embeds[0].dtype

    @property
    def use_prefix_cache(self):
        return self.prefix_cache is not None

    @cached_property
    def before_embeds(self) -> Float[Tensor, "n_templates bef_len embd_dim"]:
        return [self.embed_func(ids) for ids in self.before_ids]

    @cached_property
    def after_embeds(self) -> Float[Tensor, "n_templates aft_len embd_dim"]:
        return [self.embed_func(ids) for ids in self.after_ids]

    @cached_property
    def pad_token_embeds(self) -> Float[Tensor, "1 embd_dim"]:
        return self.embed_func(torch.tensor([self.pad_token_id], device=self.device))

    def get_triggered_inputs(
        self,
        # trigger options:
        trigger_ids: Float[Tensor, "n_candidates trigger_seq_len"] = None,
        trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embd_dim"] = None,
        append_embeds: List[Float[Tensor, "n_app_ids embd_dim"]] = None,  # of length n_templates
        chosen_template_idx: Optional[int] = None,
    ) -> ModelInput:
        """
        Returns the input embeddings with the given trigger merged in for a specific message.

        Notes:
        - We do not support varying trigger lengths in the same candidate batch
        (they must share `trigger_seq_len`).
        - for specific use cases, the following method can be optimized; however,
            currently generality and support for different input types/shapes are prioritized.

        Args:
            trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                the token ids of the trigger(s) to insert.
                If trigger_embeds is also provided, this is used for reference only.
            trigger_embeds: Tensor, shape = (n_candidates, trigger_seq_len, embd_dim)
                an optional alternative to `trigger_ids`, where the trigger embeddings
                are provided directly (useful for gradient computation).
                If provided, it is used for input computation instead of `trigger_ids`.
            append_embeds: n_templates-long List of tensors, each of shape = (n_app_ids, embd_dim)
                optional embeddings to append at the end of each message (e.g., for planting response in LMs)
            chosen_template_idx: int (required)
                the index of the message to process. Must be provided; multi-message is not supported by this method.

        Returns: A ModelInput object containing:
                - input_trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                    the token ids of the trigger(s) inserted (detached, for reference)
                - inputs_embeds: Tensor, shape = (n_candidates, seq_len, embd_dim)
                    the input embeddings with the trigger merged in;
                    if the provided input_embds required grad, then this tensor will also require grad.
                - attention_mask: Tensor, shape = (n_candidates, seq_len)
                    the attention mask matching the input embeddings
                - targets: MessageTargets
                    the targets dict for the chosen message, expanded to match n_candidates dimension
        """
        # assert trigger_ids is not None, "`trigger_ids` must be provided to `get_triggered_inputs()`."
        assert chosen_template_idx is not None, "`chosen_template_idx` must be provided to `get_triggered_inputs()`. Multi-message calls should loop over messages."
        # TODO re-read and test this critical code !!!!!!!!!!!

        if trigger_embeds is None:
            # embed the trigger-ids, if trigger embeddings are not provided
            trigger_embeds = self.embed_func(trigger_ids)

        n_candidates = trigger_embeds.shape[0]
        template_idx = chosen_template_idx

        ## Construct the parts of the input for this message:
        curr_before = self.before_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
        curr_trigger = trigger_embeds
        curr_after = self.after_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
        curr_append = (
            append_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
            if append_embeds is not None
            else None
        )

        # Build embeddings and attention mask
        embeds_parts = []
        attn_parts = []

        if not self.use_prefix_cache:
            # only add 'before' part if not using prefix cache
            embeds_parts.append(curr_before)
        # always add 'before' part attention (even with prefix cache)
        attn_parts.append(torch.ones((n_candidates, curr_before.shape[-2])))

        embeds_parts.extend([curr_trigger, curr_after])
        attn_parts.extend([
            torch.ones((n_candidates, curr_trigger.shape[-2])),
            torch.ones((n_candidates, curr_after.shape[-2])),
        ])

        if curr_append is not None:
            embeds_parts.append(curr_append)
            attn_parts.append(torch.ones((n_candidates, curr_append.shape[-2])))

        # Concatenate parts
        inputs_embeds = torch.cat(embeds_parts, dim=-2)  # (n_candidates, seq_len, embd_dim)
        attention_mask = torch.cat(attn_parts, dim=-1)  # (n_candidates, seq_len)
        attention_mask = attention_mask.to(self.device, torch.int64)

        # Calculate slices for different regions
        # Since prefix-caching removes the 'before' part from the input, we need to adjust the slices accordingly
        # (this "removal" will be reflected in the model outputs, which is where we use the slicing info)
        before_offset = curr_before.shape[-2] if not self.use_prefix_cache else 0

        input_slices = {
            SliceKey.INPUT_BEFORE: slice(
                0,
                before_offset,
            ),
            SliceKey.TRIGGER: slice(
                before_offset,
                before_offset + curr_trigger.shape[-2],
            ),
            SliceKey.INPUT_AFTER: slice(
                before_offset + curr_trigger.shape[-2],
                before_offset + curr_trigger.shape[-2] + curr_after.shape[-2],
            ),
            SliceKey.INPUT_LAST_TOKEN: slice(
                before_offset + curr_trigger.shape[-2] + curr_after.shape[-2] - 1,
                before_offset + curr_trigger.shape[-2] + curr_after.shape[-2],
            ),
            SliceKey.APPENDED: slice(
                before_offset + curr_trigger.shape[-2] + curr_after.shape[-2],
                before_offset + curr_trigger.shape[-2] + curr_after.shape[-2] + curr_append.shape[-2],
            ) if curr_append is not None else None,
        }

        ## Prepare the targets repeated for each candidate
        targets: MessageTargets = self.targets.select_message(chosen_template_idx)

        ## Prepare prefix cache kwargs (only if both message and batching are provided)
        prefix_cache_kwargs = {}
        if self.use_prefix_cache:
            prefix_cache_kwargs = self._get_prefix_cache_kwargs(
                batch_size=n_candidates,
                template_idx=chosen_template_idx,
            )

        return ModelInput(
            # input_texts=TODO
            # input_trigger_strs=TODO   !!!!!!!!!!!!!
            input_trigger_ids=trigger_ids,  # detached triggers for reference
            input_embeds=inputs_embeds.to(self.device, self.float_dtype),
            input_attention_mask=attention_mask.to(self.device, torch.int64),
            input_slices=input_slices,
            targets=targets,
            input_prefix_cache_kwargs=prefix_cache_kwargs,
        )

    def _get_prefix_cache_kwargs(
        self, batch_size: int = 1, template_idx: int = None
    ) -> List[Dict[str, transformers.DynamicCache | bool]] | Dict[str, transformers.DynamicCache | bool]:
        """Returns kwargs for model forward pass to use the prefix cache, if available."""
        if not self.use_prefix_cache:
            return dict()

        if template_idx is None:
            raise ValueError("Prefix cache without specific template_idx is not supported.")

        # Check memory first
        mem_key = (batch_size, template_idx)
        if mem_key in self._prefix_cache_kwargs_mem:
            # Retrieve from memory (stored as legacy tuple on CPU) and convert/move to device
            saved_kv = self._prefix_cache_kwargs_mem[mem_key]
            # Move tensors to device
            saved_kv = tuple(
                (k.to(self.device), v.to(self.device)) for k, v in saved_kv
            )
            past_key_values = transformers.DynamicCache.from_legacy_cache(saved_kv)
            return dict(
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Compute if not in memory
        past_key_values = self.prefix_cache[template_idx]
        # Structure: tuple(layers) of tuple(k, v) where k,v are (1, heads, seq, dim)

        if batch_size != 1:
            batch_prefix_cache = []
            for k, v in past_key_values:
                # Expand batch dimension
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
                batch_prefix_cache.append((k, v))
            past_key_values = tuple(batch_prefix_cache)

        # Store in memory on CPU as legacy format (tuple of tensors)
        # This avoids issues with DynamicCache not having a .to() method
        cache_cpu = tuple(
            (k.cpu(), v.cpu()) for k, v in past_key_values
        )
        self._prefix_cache_kwargs_mem[mem_key] = cache_cpu

        # Convert to DynamicCache and return with values on device
        past_key_values = transformers.DynamicCache.from_legacy_cache(past_key_values)
        return dict(
            past_key_values=past_key_values,
            use_cache=True,
        )


# ======================= Model logic =======================


class _HuggingFaceModelMixins:
    """Implementation of common methods for HuggingFace models."""

    _model: transformers.PreTrainedModel
    _embedding_layer: torch.nn.Embedding

    @property
    def n_layers(self) -> int:
        """Number of hidden layers in the model."""
        return self._model.config.num_hidden_layers

    @cached_property
    def effective_embedding_matrix(self) -> Float[Tensor, "vocab_size embd_dim"]:
        """
        Compuates the effective embedding matrix used by the model.

        Explanation:
            Sometimes the input embedding function is not a simple matmul embedding layer, but rather it's
            enriched with some additional logic (e.g. scaling the embeddings).
            See Gemma3 for example: https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/gemma3/modular_gemma3.py#L348
            Since our (gradient) calculation operates on the matrix *directly*, we need to take this into account.
            In this non-matmul case, merely multiplying the one-hot encoding with the matrix may provide
            an incorrect embedding. One possible fix is to require the "effective" embedding matrix, and
            operate on it instead, as we do here.
            Naturally, this assumes that the embedding function works position-wise, which is usually the case.

        Returns:
            Tensor of shape (vocab_size, embd_dim)
        """
        all_token_ids = torch.arange(self._embedding_layer.num_embeddings, device=self._model.device)
        effective_embedding_matrix = self._embedding_layer(all_token_ids)  # (vocab_size, dim)
        return effective_embedding_matrix  # shape: (vocab_size, embd_dim)

    def compute_grad_from_tokens(
        self,
        loss_func: BaseLoss,

        # Hard trigger input:
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"]=None,

        # Semi-Soft trigger input:
        candidate_trigger_probs: Float[Tensor, "n_candidates trigger_seq_len vocab_size"] = None,
        do_gumbel_softmax: bool = False,
        gumbel_softmax_temp: Optional[float] = None,

        # Additional config:
        normalize_grads: bool = True,
        return_loss: bool = False,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len vocab_size"]:
        """Compute gradients of loss w.r.t. one-hot token representations for gradient-based optimization.

        This method is the core of white-box, gradient-based trigger optimization (e.g., GCG, GASLITE).
        It computes the gradient of the loss with respect to the one-hot token matrix for each candidate
        trigger, enabling gradient-guided token selection.

        Args:
            inputs: Token inputs manager containing templates and tokenization info.
                Created by `prepare_token_inputs()`.
            loss_func: Loss function to optimize. Must be compatible with model outputs
                (e.g., LogitBasedLoss for LMs, EmbeddingBasedLoss for encoders).

            candidate_trigger_ids: Discrete token IDs for hard triggers.
                Shape: (n_candidates, trigger_seq_len)
                Mutually exclusive with `candidate_trigger_probs`.

            candidate_trigger_probs: Probability distributions over vocabulary for semi-soft triggers.
                Shape: (n_candidates, trigger_seq_len, vocab_size)
                Mutually exclusive with `candidate_trigger_ids`.
                Used for continuous optimization methods.

            do_gumbel_softmax: If True, apply Gumbel-softmax to `candidate_trigger_probs`
                before embedding. Adds stochastic exploration for soft optimization.
                Requires `gumbel_softmax_temp` to be set.

            gumbel_softmax_temp: Temperature for Gumbel-softmax sampling.
                Lower values → more discrete (sharper), higher values → more uniform.
                Only used when `do_gumbel_softmax=True`.
            
            normalize_grads: If True, L2-normalize the gradients along the vocab dimension.
                Defaults to True, as this is usually desirable for fair comparison across token positions.

            return_loss: If True, also return the computed detached loss values for each candidate. Useful for debugging.

        Returns:
            Normalized gradients w.r.t. one-hot token matrix.
            Shape: (n_candidates, trigger_seq_len, vocab_size)

            Gradients are L2-normalized along the vocab dimension (dim=-1) to enable
            fair comparison across different token positions.

        Raises:
            AssertionError: If not exactly one of the trigger input modes is provided.
            AssertionError: If effective embedding matrix doesn't match embed function
                (indicates non-standard model embedding logic).

        Example:
            >>> # GCG-style optimization with discrete token candidates
            >>> inputs = model.set_inputs_from_tokens(texts, initial_trigger, targets)
            >>> candidate_ids = torch.randint(0, vocab_size, (128, 20))  # 128 candidates
            >>> grads = model.compute_grad_from_tokens(
            ...     inputs=inputs,
            ...     loss_func=PrefillCELoss(),
            ...     candidate_trigger_ids=candidate_ids
            ... )
            >>> # grads shape: (128, 20, vocab_size)
            >>> # Use grads to select top-k tokens per position for next iteration

        """
        assert (candidate_trigger_ids is not None) ^ (candidate_trigger_probs is not None), \
            "Exactly one of `candidate_trigger_ids` or `candidate_trigger_probs` must be provided."
        assert self._token_input_manager is not None, "Token input manager is not initialized. Please call set_inputs_from_tokens() first."

        model = self._model
        embedding_layer = self._embedding_layer
        input_manager = self._token_input_manager
        n_templates = input_manager.n_templates

        # Get shape from whichever input is provided
        if candidate_trigger_ids is not None:
            n_candidates, trigger_seq_len = candidate_trigger_ids.shape
        else: # candidate_trigger_probs is not None:
            n_candidates, trigger_seq_len = candidate_trigger_probs.shape[:2]

        @find_executable_batch_size(starting_batch_size=self._backward_pass_batch_size)
        def _compute_grad__batched(
            batch_size: int,
        ) -> Float[Tensor, "n_templates n_candidates"]:

            # --- Update backward batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self._backward_pass_batch_size:
                logger.info(f"OOM detected. Reducing _backward_pass_batch_size from {self._backward_pass_batch_size} to {batch_size}")
                self._backward_pass_batch_size = batch_size
            # --------------------

            all_grads = []  # of len `n_candidate // batch_size`
            all_losses = []  # per-batch mean losses (detached)

            # Prepare the one-hot encoding matrix
            # (n_candidates, trigger_seq_len, vocab_size)
            if candidate_trigger_probs is None:
                candidate_ids_onehot_detached = torch.nn.functional.one_hot(
                    candidate_trigger_ids,
                    num_classes=embedding_layer.num_embeddings,
                ).to(model.device, model.dtype)
            else:
                candidate_ids_onehot_detached = candidate_trigger_probs.to(model.device, model.dtype)

            # Prepare the effective embedding matrix:
            embedding_matrix = self.effective_embedding_matrix  # (vocab_size, embd_dim)

            for cand_idx_start in range(0, n_candidates, batch_size):
                # for each batch we calculate its gradients, through the per-message loss
                batch_losses = []  # of len n_templates
                cand_idx_end = min(cand_idx_start + batch_size, n_candidates)

                # we compute the loss per template in the batch
                # (we avoid mixing templates, per a potentially different objective)
                for template_idx in range(0, n_templates):
                    # 1. Enable gradients on the one-hot input
                    # (bsz_triggers, trigger_seq_len, vocab_size)
                    candidate_ids_onehot = candidate_ids_onehot_detached[cand_idx_start:cand_idx_end].clone()
                    candidate_ids_onehot.requires_grad_()

                    # 1'. optionally apply gumbel-softmax to the trigger probs
                    if do_gumbel_softmax:
                        assert gumbel_softmax_temp is not None, "gumbel_softmax_temp must be provided if do_gumbel_softmax is True."
                        candidate_ids_onehot = torch.nn.functional.gumbel_softmax(
                            logits=candidate_ids_onehot,
                            tau=gumbel_softmax_temp,
                            hard=False,
                            dim=-1,
                        )

                    # 2. Apply embedding to get trigger_embeds
                    # (n_candidates, trigger_seq_len, vocab_size) @ (vocab_size, embed_dim) -> (n_candidates, trigger_seq_len, embed_dim)
                    candidate_embeds = candidate_ids_onehot @ embedding_matrix

                    # TODO move to this check to the tests, to avoid slowing down this function
                    # Only check when using discrete tokens (not soft probabilities)
                    # TODO wrap in "MORE_CHECKS" flag or something, to avoid slowing down in prod  !!!!!!!!!!!!!!!!
                    if candidate_trigger_ids is not None:
                        assert torch.allclose(
                            candidate_embeds,
                            input_manager.embed_func(
                                candidate_trigger_ids[cand_idx_start:cand_idx_end]
                            )
                        ), ("Mismatch between effective embedding matrix and embed-func. It could be that you use " \
                        "a model with non-standard embedding logic. Please report this issue!")

                    # 3. Get batched inputs & compute loss:
                    logger.debug(f"from grad [msg={template_idx}]: {candidate_embeds.shape}")

                    # Get trigger IDs for reference (if using discrete tokens)
                    # For soft triggers, compute argmax from probabilities
                    if candidate_trigger_ids is not None:
                        ref_trigger_ids = candidate_trigger_ids[cand_idx_start:cand_idx_end]
                    else:
                        # Compute discrete tokens from soft probabilities (before Gumbel-softmax)
                        ref_trigger_ids = candidate_ids_onehot_detached[cand_idx_start:cand_idx_end].argmax(dim=-1)

                    model_input = input_manager.get_triggered_inputs(
                        trigger_embeds=candidate_embeds,
                        chosen_template_idx=template_idx,

                        # Also pass trigger ids as a reference
                        trigger_ids=ref_trigger_ids,
                    )
                    model_output = self.invoke_from_tokens(
                        **model_input.to_dict(),
                        reference_loss_func=loss_func,
                    )
                    loss = resolve_and_compute_loss(model_output, model_input, loss_func)
                    batch_losses.append(loss)

                # collect losses for the batch & take avg over texts
                batch_losses = torch.stack(
                    batch_losses, dim=0
                )  # (n_templates, bsz_triggers)
                batch_losses = batch_losses.mean(dim=0)  # Shape: (bsz_triggers,)

                # Update usage stats
                self._update_usage_stats(
                    grad_calls=1,
                    grad_samples=len(batch_losses) * n_templates
                )

                # Compute the gradient of each trigger's loss w.r.t. its one-hot input
                candidate_onehot_grad = torch.autograd.grad(
                    outputs=batch_losses,
                    inputs=[candidate_ids_onehot],
                    grad_outputs=torch.ones_like(batch_losses, device=model.device),
                )[0]  # (bsz_triggers, trigger_seq_len, vocab_size)
                all_grads.append(candidate_onehot_grad)
                all_losses.append(batch_losses.detach())
                # clear_device_cache()  # clear unused GPU memory

            return (
                torch.cat(all_grads, dim=0),  # (n_candidates, trigger_seq_len, vocab_size)
                torch.cat(all_losses, dim=0),  # (n_candidates,)
            )

        # get the candidates' gradients; (n_candidates, trigger_seq_len, vocab_size)
        all_grads, all_losses = _compute_grad__batched()

        # normalize each token's gradient vector (over the vocab_size dim)
        if normalize_grads:
            all_grads = all_grads / (all_grads.norm(dim=-1, keepdim=True) + 1e-10)

        if return_loss:
            return all_grads, all_losses

        return all_grads

    def compute_grad_from_embeds(
        self,
        loss_func: BaseLoss,
        candidate_trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embed_dim"],
        return_loss: bool = False,
        normalize_grads: bool = True,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len embed_dim"]:
        """Compute gradients of loss w.r.t. trigger embeddings.

        This variant optimizes directly in the continuous embedding space, with no constraints.

        Args:
            inputs: Token inputs manager.
            loss_func: Loss function to optimize.
            candidate_trigger_embeds: Continuous embedding vectors for triggers.
                Shape: (n_candidates, trigger_seq_len, embed_dim)

        Returns:
            Gradients w.r.t. the input embeddings.
            Shape: (n_candidates, trigger_seq_len, embed_dim)
        """
        assert self._token_input_manager is not None, "Token input manager is not initialized. Please call set_inputs_from_tokens() first."

        model = self._model
        input_manager = self._token_input_manager
        n_templates = input_manager.n_templates
        n_candidates = candidate_trigger_embeds.shape[0]

        @find_executable_batch_size(starting_batch_size=self._backward_pass_batch_size)
        def _compute_grad__batched(
            batch_size: int,
        ) -> Float[Tensor, "n_templates n_candidates"]:

            # --- Update backward batch size ---
            if batch_size < self._backward_pass_batch_size:
                logger.info(f"OOM detected. Reducing _backward_pass_batch_size from {self._backward_pass_batch_size} to {batch_size}")
                self._backward_pass_batch_size = batch_size
            # --------------------

            all_grads = []
            all_losses = []

            for cand_idx_start in range(0, n_candidates, batch_size):
                batch_losses = []
                cand_idx_end = min(cand_idx_start + batch_size, n_candidates)

                for template_idx in range(0, n_templates):
                    # 1. Enable gradients on the embedding input directly
                    candidate_embeds = candidate_trigger_embeds[cand_idx_start:cand_idx_end].clone().detach()
                    candidate_embeds.requires_grad_()

                    # 2. Get batched inputs
                    model_input = input_manager.get_triggered_inputs(
                        trigger_embeds=candidate_embeds,
                        chosen_template_idx=template_idx,
                    )

                    # 3. Forward pass
                    model_output = self.invoke_from_tokens(
                        **model_input.to_dict(),
                        reference_loss_func=loss_func,
                    )

                    # 4. Compute Loss
                    loss = resolve_and_compute_loss(model_output, model_input, loss_func)
                    batch_losses.append(loss)

                # Collect losses & average over messages
                batch_losses = torch.stack(batch_losses, dim=0) # (n_templates, bsz_triggers)
                batch_losses = batch_losses.mean(dim=0)

                # Update usage stats
                self._update_usage_stats(
                    grad_calls=1,
                    grad_samples=len(batch_losses) * n_templates
                )

                # 5. Compute gradients w.r.t. the embeddings
                candidate_embeds_grad = torch.autograd.grad(
                    outputs=batch_losses,
                    inputs=[candidate_embeds],
                    grad_outputs=torch.ones_like(batch_losses, device=model.device),
                )[0] # (bsz_triggers, trigger_seq_len, embed_dim)

                all_grads.append(candidate_embeds_grad)
                all_losses.extend(batch_losses.detach().cpu().tolist())

            return torch.cat(all_grads, dim=0), torch.tensor(all_losses, device=model.device).mean().item()

        # Execute batched computation
        all_grads, avg_loss = _compute_grad__batched()

        # normalize each token's gradient vector (over the embed_dim dim)
        if normalize_grads:
            all_grads = all_grads / (all_grads.norm(dim=-1, keepdim=True) + 1e-10)

        if return_loss:
            return all_grads, avg_loss

        return all_grads


    @torch.no_grad()
    def compute_loss_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"] | Float[Tensor, "n_templates n_candidates"]:
        """Computes the loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            inputs_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            keep_message_dim : bool
                whether to return the loss per message (shape = (n_templates, n_candidates))
        Returns:
            Tensor, shape = (n_candidates,), or (n_templates, n_candidates) if keep_message_dim=True
                the loss for each candidate sequence
        """
        assert self._token_input_manager is not None, "Token input manager is not initialized. Please call set_inputs_from_tokens() first."

        input_manager = self._token_input_manager
        n_templates = input_manager.n_templates
        n_candidates, trigger_seq_len = candidate_trigger_ids.shape

        @find_executable_batch_size(starting_batch_size=self._forward_pass_batch_size)
        def _compute_candidates_loss__batched(
            batch_size: int,
        ) -> Float[Tensor, "n_templates n_candidates"]:

            # --- Update forward batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self._forward_pass_batch_size:
                logger.info(f"OOM detected. Reducing _forward_pass_batch_size from {self._forward_pass_batch_size} to {batch_size}")
                self._forward_pass_batch_size = batch_size
            # --------------------

            all_loss = [
                [] for _ in range(n_templates)
            ]  # list of list of tensors, to be concatenated later

            for template_idx, cand_idx in itertools.product(
                # we avoid mixing messages, per a potentially different objective
                range(0, n_templates),
                range(0, n_candidates, batch_size),
            ):
                cand_idx_end = min(cand_idx + batch_size, n_candidates)
                batch_candidate_trigger_ids = candidate_trigger_ids[cand_idx:cand_idx_end]

                logger.debug(f"from loss [msg={template_idx}]: {(cand_idx_end - cand_idx)}")
                model_input = input_manager.get_triggered_inputs(
                    trigger_ids=batch_candidate_trigger_ids,
                    chosen_template_idx=template_idx,
                )
                model_output = self.invoke_from_tokens(
                        **model_input.to_dict(),
                        reference_loss_func=loss_func,
                    )
                loss = resolve_and_compute_loss(model_output, model_input, loss_func)
                all_loss[template_idx].append(loss)

            return torch.stack([torch.cat(_l, dim=0) for _l in all_loss], dim=0)

        losses = _compute_candidates_loss__batched()
        # clear_device_cache()  # clear unused GPU memory

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # reduce message dim -> (n_candidates,)
        return losses

    @abstractmethod
    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_model"],
        input_attention_mask: Int[Tensor, "bsz seq_len"],
        reference_loss_func: BaseLoss = None,
        **kwargs,
    ) -> ModelOutput:
        """Performs a forward pass with the given token-based model input. Forward pass is expected to be done on `input_embeds`.

        Args:
            input_embeds: Float[Tensor, "bsz seq_len d_model"]
                the input embeddings with the trigger merged in; if provided, used instead of any other potential input.
            input_attention_mask: Int[Tensor, "bsz seq_len"]
                the attention mask matching the input embeddings
            reference_loss_func: BaseLoss
                the loss function to use for reference (some models may need it for special handling)

        Returns:
            ModelOutput
                the model output containing logits, embeddings, attentions, etc.
        """
        pass

    @staticmethod
    def cast_to_model_tokenizer(
        old_ids: Float[Tensor, "bsz seq_len"],
        model_from: "_HuggingFaceModelMixins",
        model_to: "_HuggingFaceModelMixins",
    ) -> Tuple[Float[Tensor, "bsz len_old"], Float[Tensor, "bsz len_new"]]:
        """
        Given `ids` in the `model_from` tokenizer, heurisically casts them to the
        `model_to` tokenizer, while filtering out mismatches.
        """
        # a. decode w/ util-model tokenizer
        strs = model_from.tokenizer.batch_decode(old_ids)

        # b. encode w/ model tokenizer
        new_ids = [
            model_to.tokenizer.encode(s, return_tensors="pt", add_special_tokens=False)
            .to(model_to.device)
            .squeeze(0)
            for s in strs
        ]

        # c'. pick the maximal length with which most triggers fit (to avoid cutting too much)
        lengths = [ids.shape[-1] for ids in new_ids]
        counts = np.bincount(lengths)
        _min_len = np.argmax(counts)  # so most ids will be kept as fully
        # smaller than min -> drop
        to_drop_indices = set([i for i, l in enumerate(lengths) if l < _min_len])
        old_ids = old_ids[[i for i in range(len(new_ids)) if i not in to_drop_indices]]
        new_ids = [ids for i, ids in enumerate(new_ids) if i not in to_drop_indices]
        # longer than min -> trim
        new_ids = [ids[..., :_min_len] for ids in new_ids]
        new_ids = torch.stack(new_ids, dim=0)  # (<= bsz, min_len)

        return old_ids, new_ids.to(model_to.device, torch.int64)

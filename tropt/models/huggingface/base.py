import itertools
import logging
from abc import abstractmethod
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from accelerate.utils.memory import clear_device_cache, find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import BaseLoss
from tropt.models import (
    BatchedTargetsDict,
    MessageBatchedTargetsDict,
    TargetsDict,
    TargetsDictPlus,
    TokenInputsManager,
)

logger = logging.getLogger(__name__)

# ======================= Input/Output Handlers logic =======================


class _HFTokenInputsManager(TokenInputsManager):
    before_ids: List[Float[Tensor, "bef_len"]]
    after_ids: List[Float[Tensor, "aft_len"]]  # of length n_messages
    embed_func: torch.nn.Embedding
    targets: TargetsDictPlus
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
        targets: TargetsDict | TargetsDictPlus = None,
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
        # make sure it's a TargetsDictPlus, we use its utils later
        targets = TargetsDictPlus(targets, n_messages=self.n_messages)
        targets = targets.to_device(model.device)
        self.targets = targets

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if self.n_messages > 1 and use_prefix_cache:
            # Prefix cache is currently disabled for multiple messages until analyzing different edge cases [TODO]
            logger.warning("Prefix cache is currently unsupported: prefix cahce is now manually set to disabled, since multiple messages are used.")
            use_prefix_cache = False
        prefix_cache: List[ # per message
            Tuple[ # n_layers of these:
                Tuple[
                    Float[Tensor, "1 n_head seq_len head_dim"],  # keys
                    Float[Tensor, "1 n_head seq_len head_dim"],  # values
                ]
            ]
        ] = []

        if use_prefix_cache:
            for i in range(self.n_messages):
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

    @property
    def vocab_size(self):
        # TODO some models might have slightly different effecive vocab size in weight (?)
        #      it's possible that each caller need to receive different vocab size; go over these.
        return self.tokenizer.vocab_size

    @property
    def n_messages(self):
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
    def before_embeds(self) -> Float[Tensor, "n_messages bef_len embd_dim"]:
        return [self.embed_func(ids) for ids in self.before_ids]

    @cached_property
    def after_embeds(self) -> Float[Tensor, "n_messages aft_len embd_dim"]:
        return [self.embed_func(ids) for ids in self.after_ids]

    @cached_property
    def pad_token_embeds(self) -> Float[Tensor, "1 embd_dim"]:
        return self.embed_func(torch.tensor([self.pad_token_id], device=self.device))

    def get_triggered_inputs(
        self,
        # trigger options:
        trigger_ids: Float[Tensor, "n_candidates trigger_seq_len"] = None,
        trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embd_dim"] = None,
        append_embeds: List[Float[Tensor, "n_app_ids embd_dim"]] = None,  # of length n_messages
        chosen_message_idx: Optional[int] = None,
    ) -> dict[str, Tensor | BatchedTargetsDict | MessageBatchedTargetsDict]:
        """
        Returns the input embeddings with the given trigger merged in. That is, a dict,
        including `inputs_embeds` of shape (n_messages, n_candidates, seq_len, embd_dim).

        Notes:
        (I) Note that for specific use cases, the following method can be optimized; however,
            currently generality and support for different input types/shapes are prioritized.
        (II) We do not support varying trigger lengths in the same candidate batch (they must
             share `trigger_seq_len`).

        Args:
            trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                the token ids of the trigger(s) to insert
            trigger_embeds: Tensor, shape = (n_candidates, trigger_seq_len, embd_dim)
                an optional alternative to `trigger_ids`, where the trigger embeddings
                are provided directly (useful for gradient computation)
            include_after: bool
                whether to include the after sequence (useful for some attacks calculating the trigger logits)
            append_embeds: n_messages-long List of tensors, each of shape = (n_app_ids, embd_dim)
                optional embeddings to append at the end of each message (e.g., for planting response in LMs)
            batch_slice: slice
                slice to apply on the n_candidates dimension for batching
            chosen_message_idx: Optional[int]
                if provided, selects only the given message (useful for batching); if None, all messages are returned.

        Returns:
            dict with keys:
                - inputs_embeds: Tensor, shape = (n_messages, n_candidates, seq_len, embd_dim)
                    the input embeddings with the trigger merged in; padded to the same length, if needed;
                    shape depends on `batch_slice`/`chosen_message_idx` options.
                - attention_mask: Tensor, shape = (n_messages, n_candidates, seq_len)
                    the attention mask matching the input embeddings;
                    shape depends on `batch_slice`/`chosen_message_idx` options.
                - targets: BatchedTargetsDict | MessageBatchedTargetsDict
                    the targets dict, expanded to match the n_candidates dimension;
                    inclusion of all the n_messages depends on `chosen_message_idx` option.
        """
        # TODO to simplify the flow (and possible shapes), allow this function only to handle a specific message_idx at a time
        #     (i.e., `assert chosen_message_idx is not None`)
        assert [trigger_ids, trigger_embeds].count(None) == 1, \
            "Exactly one of `trigger_ids` or `trigger_embeds` must be provided."

        if trigger_ids is not None:
            trigger_embeds = self.embed_func(trigger_ids)

        # add message dim to triggers -> (curr_n_messages, n_candidates, trigger_seq_len, embd_dim)
        messages = [chosen_message_idx] if chosen_message_idx is not None else range(self.n_messages)
        curr_n_messages = len(messages)
        trigger_embeds = trigger_embeds.unsqueeze(0).repeat(curr_n_messages, 1, 1, 1)
        n_candidates = trigger_embeds.shape[1]

        ## Construct the parts of the inputs:
        inputs_embeds_lst_parts: List[
            List[Float[Tensor, "n_candidates part_len embd_dim"]]
        ] = [ [] for _ in messages ]
        attention_mask_lst_parts: List[
            List[Float[Tensor, "n_candidates part_len"]]
        ] = [ [] for _ in messages ]
        # keep track of the slices of each part, per message
        slices: List[dict[str, slice]] = []

        # we iterate over messages here, as we may have different lengths for each message
        for i, message_idx in enumerate(messages):
            curr_before, curr_trigger, curr_after, curr_append = (
                 # seq, emb -> n_candidates, seq, embd
                self.before_embeds[message_idx].unsqueeze(0).repeat(n_candidates, 1, 1),
                trigger_embeds[i],
                self.after_embeds[message_idx].unsqueeze(0).repeat(n_candidates, 1, 1),
                (
                    append_embeds[message_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
                    if append_embeds is not None
                    else None
                ),
            )

            # concatenate all parts:
            curr_embeds, curr_attns = [], []

            if not self.use_prefix_cache:
                # only add 'before' part if not using prefix cache
                curr_embeds.append(curr_before)
            # as required, we add 'before' part attention even with prefix cache
            curr_attns.append(torch.ones((n_candidates, curr_before.shape[-2])))

            curr_embeds.extend([curr_trigger, curr_after])
            curr_attns.extend([
                torch.ones((n_candidates, curr_trigger.shape[-2])),
                torch.ones((n_candidates, curr_after.shape[-2])),
            ])
            if append_embeds is not None:
                curr_embeds.append(curr_append)
                curr_attns.append(torch.ones((n_candidates, curr_append.shape[-2])))

            inputs_embeds_lst_parts[i] = curr_embeds
            attention_mask_lst_parts[i] = curr_attns

            before_offset = curr_before.shape[-2] if not self.use_prefix_cache else 0
            curr_slices = dict(
                adv=slice(
                    before_offset,
                    before_offset + curr_trigger.shape[-2],
                ),
                chat_template_after=slice(
                    # TODO this is currently only correct for LMs and suffix attacks (otherwise there might be more token in the "curr_after" other than the chat ones)-- need to generalize!
                    before_offset + curr_trigger.shape[-2],
                    before_offset + curr_trigger.shape[-2] + curr_after.shape[-2],  # noqa
                ),
                appended=slice(
                    before_offset + curr_trigger.shape[-2] + curr_after.shape[-2],
                    before_offset + curr_trigger.shape[-2] + curr_after.shape[-2] + curr_append.shape[-2],  # noqa
                ) if curr_append is not None else None,
            )
            slices.append(curr_slices)

        ## Add padding if needed, while matching the attention mask
        inputs_embeds_lst: List[Float[Tensor, "n_candidates seq_len embd_dim"]] = []
        attention_mask_lst: List[Float[Tensor, "n_candidates seq_len"]] = []
        max_seq_len = max(
            sum(part.shape[-2] for part in parts) for parts in inputs_embeds_lst_parts
        )
        for i, message_idx in enumerate(messages):
            curr_embeds = inputs_embeds_lst_parts[i]
            curr_embeds_len = sum(part.shape[-2] for part in curr_embeds)
            curr_attention_mask = torch.cat(attention_mask_lst_parts[i], dim=-1)
            curr_attention_mask = curr_attention_mask.to(self.device, torch.int64)

            # pad to max_seq_len, according to padding_side
            if curr_embeds_len < max_seq_len:
                pad_len = max_seq_len - curr_embeds_len
                pad_embeds = self.pad_token_embeds.unsqueeze(0).repeat(
                    n_candidates, pad_len, 1
                )  # (1, embd_dim) -> (n_candidates, pad_len, embd_dim)
                if self.padding_side == "right":
                    curr_embeds.append(pad_embeds)
                    curr_attention_mask = torch.cat(
                        [
                            curr_attention_mask,
                            torch.zeros((n_candidates, pad_len), device=self.device, dtype=torch.int64),  # noqa
                        ],
                        dim=-1,
                    )
                else:  # left padding
                    if self.use_prefix_cache:
                        raise ValueError("Active left padding with prefix cache is not supported. Either use input that does " \
                        "not require padding, or disable prefix cache.")
                    curr_embeds = [pad_embeds] + curr_embeds
                    curr_attention_mask = torch.cat(
                        [
                            torch.zeros((n_candidates, pad_len), device=self.device, dtype=torch.int64),  # noqa
                            curr_attention_mask,
                        ],
                        dim=-1,
                    )
                    # also need to shift the slices
                    slices[i] = {
                        k: slice(v.start + pad_len, v.stop + pad_len)
                        for k, v in slices[i].items()
                    }

            inputs_embeds_lst.append(
                torch.cat(curr_embeds, dim=-2)
            )  # cat on seq length dim
            attention_mask_lst.append(curr_attention_mask)

        inputs_embeds = torch.stack(
            inputs_embeds_lst, dim=0
        )  # (curr_n_messages, n_candidates, seq_len, embd_dim)
        attention_mask = torch.stack(
            attention_mask_lst, dim=0
        )  # (curr_n_messages, n_candidates, seq_len)

        ## expand slices for candidates
        slices = [[msg_slices] * n_candidates for msg_slices in slices]

        ## If a message is selected, discard the message dim
        if chosen_message_idx is not None:
            inputs_embeds = inputs_embeds.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            slices = slices[0]

        ## Also prepare the targets repeated for each candidate, if any
        targets: TargetsDictPlus = self.targets.copy()
        targets: BatchedTargetsDict = TargetsDictPlus.get_expanded_with_candidates(targets, n_candidates)
        if chosen_message_idx is not None:
            targets: MessageBatchedTargetsDict = TargetsDictPlus.get_message_from_batched_targets(
                targets, chosen_message_idx
            )

        # add the slices info for loss computation
        targets['slices'] = slices

        # TODO move the batching and message_idx to another function, to avoid repeated calls of the _whole_ function
        ## Apply batching options:
        # if batch_slice != slice(None, None, None):
        #     inputs_embeds = inputs_embeds[:, batch_slice]
        #     attention_mask = attention_mask[:, batch_slice]
        #     targets: BatchedTargetsDict = TargetsDictPlus.get_candidate_batch_from_batched_targets(targets, batch_slice)

        ## Prepare prefix cache kwargs (only if both message and batching are provided)
        prefix_cache_kwargs = {}
        if self.use_prefix_cache:
            prefix_cache_kwargs = self._get_prefix_cache_kwargs(
                batch_size=n_candidates,
                message_idx=chosen_message_idx,
            )

        return dict(
            inputs_embeds=inputs_embeds.to(self.device, self.float_dtype),
            attention_mask=attention_mask.to(self.device, torch.int64),
            targets=targets,
            prefix_cache_kwargs=prefix_cache_kwargs,
        )

    # TODO add get_triggered_texts()  ???

    def _get_prefix_cache_kwargs(
        self, batch_size: int = 1, message_idx: int = None
    ) -> List[Dict[str, transformers.DynamicCache | bool]] | Dict[str, transformers.DynamicCache | bool]:
        """Returns kwargs for model forward pass to use the prefix cache, if available."""
        # TODO optimization: keep a dict of these for different batch sizes, to avoid recomputing them every time

        if not self.use_prefix_cache:
            return dict()

        curr_prefix_caches = []
        messages = [message_idx] if message_idx is not None else range(self.n_messages)
        if message_idx is None:
            # TODO to support multi-message with prefix cache, we need to make sure it's well-defined as the model input
            raise ValueError("Prefix cache without specific message_idx is not supported yet.")

        for message_idx in messages:
            # Retrieve the cache for this specific message
            past_key_values = self.prefix_cache[message_idx]
            # Structure: tuple(layers) of tuple(k, v) where k,v are (1, heads, seq, dim)

            if batch_size != 1:
                batch_prefix_cache = []
                for k, v in past_key_values:
                    # Expand batch dimension
                    k = k.expand(batch_size, -1, -1, -1)
                    v = v.expand(batch_size, -1, -1, -1)
                    batch_prefix_cache.append((k, v))
                past_key_values = tuple(batch_prefix_cache)

            past_key_values = transformers.DynamicCache.from_legacy_cache(past_key_values)

            curr_prefix_caches.append(dict(
                past_key_values=past_key_values,
                use_cache=True,
            ))

        if message_idx is not None:
            return curr_prefix_caches[0]
        return curr_prefix_caches


# ======================= Model logic =======================


class _HuggingFaceModelMixins:
    """Implementation of common methods for HuggingFace models."""

    model: transformers.PreTrainedModel
    embedding_layer: torch.nn.Embedding

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
        all_token_ids = torch.arange(self.embedding_layer.num_embeddings, device=self.model.device)
        effective_embedding_matrix = self.embedding_layer(all_token_ids)  # (vocab_size, dim)
        return effective_embedding_matrix  # shape: (vocab_size, embd_dim)

    def compute_grad_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        inputs: _HFTokenInputsManager,
        loss_func: BaseLoss,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len vocab_size"]:
        """
        Computes the gradient of the loss w.r.t the one-hot token matrix
        for a batch of triggers. Uses dynamic batch size.
        """
        model = self.model
        embedding_layer = self.embedding_layer
        n_messages = inputs.n_messages
        n_candidates, trigger_seq_len = candidate_trigger_ids.shape
        # [TODO: allow second order grads] make it another function

        @find_executable_batch_size(starting_batch_size=self.backward_pass_batch_size)
        def _compute_grad__batched(
            batch_size: int,
        ) -> Float[Tensor, "n_messages n_candidates"]:

            # --- Update backward batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self.backward_pass_batch_size:
                logger.info(f"OOM detected. Reducing backward_pass_batch_size from {self.backward_pass_batch_size} to {batch_size}")
                self.backward_pass_batch_size = batch_size
            # --------------------

            all_grads = []  # of len `n_candidate // batch_size`

            # Prepare the one-hot encoding matrix
            # (n_candidates, trigger_seq_len, vocab_size)
            candidate_ids_onehot_detached = torch.nn.functional.one_hot(
                candidate_trigger_ids,
                num_classes=embedding_layer.num_embeddings,
            ).to(model.device, model.dtype)

            # Prepare the effective embedding matrix:
            embedding_matrix = self.effective_embedding_matrix  # (vocab_size, embd_dim)

            for cand_idx_start in range(0, n_candidates, batch_size):
                # for each batch we calculate its gradients, through the per-message loss
                batch_losses = []  # of len n_messages
                cand_idx_end = min(cand_idx_start + batch_size, n_candidates)

                # we compute the loss per message in the batch
                # (we avoid mixing messages, per a potentially different objective)
                for message_idx in range(0, n_messages):
                    # 1. Enable gradients on the one-hot input
                    # (bsz_triggers, trigger_seq_len, vocab_size)
                    candidate_ids_onehot = candidate_ids_onehot_detached[cand_idx_start:cand_idx_end].clone()
                    candidate_ids_onehot.requires_grad_()
                    # [TODO: allow second order grads] accept the `candidate_ids_onehot` as input (so the user can use the non-detached gradients later)

                    # 2. Apply embedding to get trigger_embeds
                    # (n_candidates, trigger_seq_len, vocab_size) @ (vocab_size, embed_dim) -> (n_candidates, trigger_seq_len, embed_dim)
                    candidate_embeds = candidate_ids_onehot @ embedding_matrix

                    # TODO move to this check to the tests, to avoid slowing down this function
                    assert torch.allclose(
                        candidate_embeds,
                        inputs.embed_func(
                            candidate_trigger_ids[cand_idx_start:cand_idx_end]
                        )
                    ), ("Mismatch between effective embedding matrix and embed-func. It could be that you use " \
                    "a model with non-standard embedding logic. Please report this issue!")

                    # 3. Get batched inputs & compute loss:
                    logger.debug(f"from grad [msg={message_idx}]: {candidate_embeds.shape}")
                    loss = self._loss_hook(
                        **inputs.get_triggered_inputs(
                            trigger_embeds=candidate_embeds,
                            chosen_message_idx=message_idx,
                        ),
                        loss_func=loss_func,
                    )
                    batch_losses.append(loss)

                # collect losses for the batch & take avg over messages
                batch_losses = torch.stack(
                    batch_losses, dim=0
                )  # (n_messages, bsz_triggers)
                batch_losses = batch_losses.mean(dim=0)  # Shape: (bsz_triggers,)
                # logger.debug(f"\tgrad: {batch_losses.mean().item()}")

                # Update usage stats
                self._update_usage_stats(
                    grad_calls=1,
                    grad_samples=len(batch_losses) * n_messages
                )

                # Compute the gradient of each trigger's loss w.r.t. its one-hot input
                candidate_onehot_grad = torch.autograd.grad(
                    outputs=batch_losses,
                    inputs=[candidate_ids_onehot],
                    grad_outputs=torch.ones_like(batch_losses, device=model.device),
                    # create_graph=True  # [TODO: allow second order grads] <-- This tells PyTorch to make grads differentiable
                )[0]  # (bsz_triggers, trigger_seq_len, vocab_size)
                all_grads.append(candidate_onehot_grad)
                # clear_device_cache()  # clear unused GPU memory

            return torch.cat(
                all_grads, dim=0
            )  # (n_candidates, trigger_seq_len, vocab_size)

        # get the candidates' gradients; (n_candidates, trigger_seq_len, vocab_size)
        all_grads = _compute_grad__batched()

        # normalize each token's gradient vector (over the vocab_size dim)
        all_grads = all_grads / (all_grads.norm(dim=-1, keepdim=True) + 1e-10)

        return all_grads

    @torch.no_grad()
    def compute_loss_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        inputs: _HFTokenInputsManager,
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"] | Float[Tensor, "n_messages n_candidates"]:
        """Computes the loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            inputs_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            keep_message_dim : bool
                whether to return the loss per message (shape = (n_messages, n_candidates))
        Returns:
            Tensor, shape = (n_candidates,), or (n_messages, n_candidates) if keep_message_dim=True
                the loss for each candidate sequence
        """

        n_messages = inputs.n_messages
        n_candidates, trigger_seq_len = candidate_trigger_ids.shape

        @find_executable_batch_size(starting_batch_size=self.forward_pass_batch_size)
        def _compute_candidates_loss__batched(
            batch_size: int,
        ) -> Float[Tensor, "n_messages n_candidates"]:

            # --- Update forward batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self.forward_pass_batch_size:
                logger.info(f"OOM detected. Reducing forward_pass_batch_size from {self.forward_pass_batch_size} to {batch_size}")
                self.forward_pass_batch_size = batch_size
            # --------------------

            all_loss = [
                [] for _ in range(n_messages)
            ]  # list of list of tensors, to be concatenated later

            for message_idx, cand_idx in itertools.product(
                # we avoid mixing messages, per a potentially different objective
                range(0, n_messages),
                range(0, n_candidates, batch_size),
            ):
                cand_idx_end = min(cand_idx + batch_size, n_candidates)
                batch_candidate_trigger_ids = candidate_trigger_ids[cand_idx:cand_idx_end]

                logger.debug(f"from loss [msg={message_idx}]: {(cand_idx_end - cand_idx)}")
                loss = self._loss_hook(
                    **inputs.get_triggered_inputs(
                        trigger_ids=batch_candidate_trigger_ids,
                        chosen_message_idx=message_idx,
                    ),
                    loss_func=loss_func,
                )  # shape: (bsz,)
                all_loss[message_idx].append(loss)

                self._update_usage_stats(
                    forward_calls=1,
                    forward_samples=len(batch_candidate_trigger_ids)
                )

            return torch.stack([torch.cat(_l, dim=0) for _l in all_loss], dim=0)

        losses = _compute_candidates_loss__batched()
        # clear_device_cache()  # clear unused GPU memory
        # logger.debug(f"\tloss: {losses.mean().item()}")

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # reduce message dim -> (n_candidates,)
        return losses

    @abstractmethod
    def _loss_hook(
        self,
        inputs_embeds: Float[Tensor, "bsz seq_len embd_dim"],
        attention_mask: Optional[Float[Tensor, "bsz seq_len"]],
        targets: MessageBatchedTargetsDict,
        loss_func: BaseLoss,
        prefix_cache_kwargs: dict = {},
        loss_kwargs: dict = {},
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        """
        Hook for computing the loss on the given inputs, which are for *specific message* (for the input to be aligned).
        Must be implemented in subclasses.
        """
        raise NotImplementedError("_loss_hook must be implemented in subclasses.")

    @staticmethod
    def cast_to_model_tokenizer(
        old_ids: Float[Tensor, "bsz seq_len"],
        model_from: "_HuggingFaceModelMixins",
        model_to: "_HuggingFaceModelMixins",
    ):
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

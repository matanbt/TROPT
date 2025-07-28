import logging
from abc import ABC, abstractmethod
from typing import List

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.loss.utils import masked_mean

logger = logging.getLogger(__name__)

## ------- Loss ------- ##
class BaseLoss(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Float[Tensor, "bsz 1"]:
        pass


############################
class LogitBasedLoss(BaseLoss):
    """Loss is computed based on model output logits."""

    TARGET_KEY: str = "target_outputs_toks"

    def __call__(
        self,
        logits: Float[Tensor, "bsz target_seq_len vocab_size"],
        target_ids: Float[Tensor, "bsz target_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError()


class PrefillCELoss(LogitBasedLoss):
    """
    Encourages (=maximize likelihood) the model to produce the target output (mostly an affirmative response).
    """

    temperature: float = 1.0  # for softmax

    def __call__(
        self,
        logits: Float[Tensor, "bsz target_seq_len vocab_size"],
        target_ids: Int[Tensor, "bsz target_seq_len"],
        ignore_index: int = -100,
    ) -> Float[Tensor, "bsz"]:
        logits = logits / self.temperature
        assert (
            logits.ndim == 3
            and target_ids.ndim == 2
            and logits.shape[:2] == target_ids.shape[:2]
        ), f"Shape mismatch: logits {logits.shape}, target_ids {target_ids.shape}"

        loss = torch.nn.functional.cross_entropy(
            logits.transpose(-1, -2),  # move vocab size (= # classes) to 2nd dim
            target_ids,
            reduction="none",
            ignore_index=ignore_index,
        )  # (bsz, seq_len)

        return masked_mean(loss, (target_ids != ignore_index).float())


class PrefillMellowMaxLoss(LogitBasedLoss):
    """
    Encourages the model to produce the target output by maximizing the mellowmax of the target logits.
    https://arxiv.org/pdf/1612.05628, http://confirmlabs.org/posts/TDC2023
    """

    mellowmax_alpha: float = 1.0
    temperature: float = 1.0

    def __call__(
        self,
        logits: Float[Tensor, "bsz target_seq_len vocab_size"],
        target_ids: Int[Tensor, "bsz target_seq_len"],
        ignore_index: int = -100,
    ) -> Float[Tensor, "bsz 1"]:
        logits = logits / self.temperature
        assert logits.shape[:-1] == target_ids.shape, "Shape mismatch"

        # 1. Create mask
        mask = target_ids != ignore_index
        # replace ignore index with 0 to avoid index error (will be masked later anyway)
        target_ids = target_ids.masked_fill(~mask, 0)

        # 2. Gather the logits corresponding to the target IDs
        target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        # Mellowmax maximizes its input, so to maximize the target_logits,
        # we minimize the negative of the target_logits.
        target_logits = -target_logits

        # 3. Prepare inputs for LogSumExp
        # We want to ignore padded tokens in the sum, so set them to -inf (exp(-inf) = 0)
        val_for_lse = self.mellowmax_alpha * target_logits
        val_for_lse = val_for_lse.masked_fill(~mask, float('-inf'))

        # 4. Calculate valid tokens per sequence
        n_valid = mask.sum(dim=-1).float().clamp(min=1.0)

        # Calculate Loss
        loss = (
            1.0
            / self.mellowmax_alpha
            * (
                torch.logsumexp(val_for_lse, dim=-1)
                - torch.log(n_valid)
            )
        )

        return loss  # (bsz,)


class PrefillCWLoss(LogitBasedLoss):
    """
    Encourages (=maximize likelihood) the model to produce the target output (mostly an affirmative response).
    CW-inspired hinge loss on the difference between the largest and the target logits.
    https://arxiv.org/abs/2402.09674
    """

    cw_margin: float = 1e-3

    def __call__(
        self,
        logits: Float[Tensor, "bsz target_seq_len vocab_size"],
        target_ids: Int[Tensor, "bsz target_seq_len"],
        ignore_index: int = -100,
    ) -> Float[Tensor, "bsz 1"]:
        assert logits.shape[:2] == target_ids.shape, (logits.shape, target_ids.shape)
        vocab_dim: int = -1  # dimension of vocab size

        # Create mask and safe indices
        mask = target_ids != ignore_index
        target_ids = target_ids.masked_fill(~mask, 0)  # replace ignore index with 0 to avoid index error (will be masked later anyway)

        # extract the target's logits (using the target ids as indices)
        tgt_logits = logits.gather(vocab_dim, target_ids.unsqueeze(-1)).squeeze(-1)

        # Set logits of target tok to -inf so it cannot be the largest
        tmp_logits = logits.clone()
        tmp_logits.scatter_(vocab_dim, target_ids.unsqueeze(-1), -torch.inf)

        # pick the largest logit among the non-target tokens
        largest_non_tgt_logits = tmp_logits.max(vocab_dim).values

        # calculate the CW loss:
        loss = largest_non_tgt_logits - tgt_logits
        loss = loss.clamp_min(-self.cw_margin)

        # Zero out loss for padding tokens
        loss = loss * mask.float()

        return masked_mean(loss, mask.float())

class PerplexityLoss(LogitBasedLoss):
    """
    Calculates perplexity, which is exp(cross_entropy).
    Useful for penalizing non-fluent triggers.
    """

    temperature: float = 1.0

    def __call__(
        self,
        logits: Float[Tensor, "bsz seq_len vocab_size"],
        target_ids: Int[Tensor, "bsz seq_len"],
        ignore_index: int = -100,
    ) -> Float[Tensor, "bsz 1"]:
        logits = logits / self.temperature
        # targets[targets == self.tokenizer.pad_token_id] = -100  # ignore padding  [TODO add support for target ids masking]
        assert (
            logits.ndim == 3 and logits.shape[:2] == target_ids.shape[:2]
        ), "Shape mismatch"

        ce_loss = torch.nn.functional.cross_entropy(
            logits.transpose(-1, -2),  # move vocab size (= # classes) to 2nd dim
            target_ids,
            reduction="none",
            ignore_index=ignore_index,
        )  # (bsz, seq_len)

        mean_ce_loss = masked_mean(ce_loss, (target_ids != ignore_index).float())  # (bsz,)

        perplexity = torch.exp(mean_ce_loss)

        return perplexity

#############################
class AttentionBasedLoss(BaseLoss):
    """Loss is computed based on model attention weights."""

    def __call__(
        self,
        attentions: Float[Tensor, "bsz n_layers n_heads seq_len[dst] seq_len[src]"],
        slices: List[dict[str, slice]] = {},  # of length bsz
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError()


class AttentionEnhLoss(AttentionBasedLoss):
    """
    Encourages attention from the trigger tokens to the chat template after the adversarial trigger.
    https://arxiv.org/abs/2506.12880, https://arxiv.org/abs/2410.09040
    """

    targeted_layers: slice = slice(None)
    src_slc_name: str = "adv"
    dst_slc_name: str = "chat_template_after"

    def __call__(
        self,
        attentions: Float[Tensor, "bsz n_layers n_heads seq_len[dst] seq_len[src]"],
        slices: List[dict[str, slice]] = {},  # of length bsz
    ) -> Float[Tensor, "bsz"]:
        if "chat_template_after" in (self.src_slc_name, self.dst_slc_name):
            logger.warning("`chat_template_after` is currently only correct for LMs and on suffix attacks.")
        slc_src = [
            message_slices.get(self.src_slc_name, slice(None))
            for message_slices in slices
        ]
        slc_dst = [
            message_slices.get(self.dst_slc_name, slice(None))
            for message_slices in slices
        ]

        # TODO vectorize this process:
        loss = torch.zeros(attentions.shape[0], device=attentions.device)  # (bsz,)
        for i, (curr_slc_src, curr_slc_dst) in enumerate(zip(slc_src, slc_dst)):
            loss[i] = attentions[
                i, self.targeted_layers, :, curr_slc_dst, curr_slc_src
            ].mean()

        # loss = attentions.mean(dim=(-1, -2, -3, -4))  # average over heads, src, dst, layers

        loss *= -1  # maximize attention

        return loss


############################
class EmbeddingBasedLoss(BaseLoss):
    """Loss is computed based on model embeddings, compared to given target vectors."""

    TARGET_KEY = "target_vectors"


class SimilarityLoss(EmbeddingBasedLoss):
    """
    Encourages given representation(s) to align (cos-sim) with the given target vectors.
    """


    def __call__(
        self,
        vectors: Float[Tensor, "bsz d_model"],
        target_vectors: Float[Tensor, "bsz d_model"],
    ) -> Float[Tensor, "bsz"]:
        assert vectors.ndim == target_vectors.ndim == 2, "Shape mismatch"
        target_vectors = target_vectors.to(vectors.device)

        # normalize:
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        target_vectors = target_vectors / target_vectors.norm(dim=-1, keepdim=True)

        # cosine similarity via normalized dot product:
        cos_sim = (vectors * target_vectors).sum(dim=-1, keepdim=True)
        loss = -1 * cos_sim  # maximize cos-sim <=> minimize (-1 * cos-sim)

        return loss.squeeze(-1)


############################


class TextBasedLoss(BaseLoss):
    """Mixin for models that can compute losses based on model outputs (embedding for encoder, text response for LMs); fits query access."""

    pass


class ResponseLMScoreLoss(TextBasedLoss):
    """A loss based on an LM-as-a-judge score of the model's response."""

    pass


############################
class SteeringEnhLoss(BaseLoss):
    def __call__(
        self,
        hidden_states: Float[Tensor, "bsz n_layers seq_len d_model"],
        target_directions: Float[Tensor, "bsz d_model"],
        targeted_layers: slice = slice(None),
        slc: slice = slice(None),  # TODO list as it may vary across elements?
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError("TODO")
        # TODO implement


############################
# from PAL:
#     @torch.no_grad()
#     def _compute_loss(
#         self,
#         batch_input_ids: BatchTokenIds,
#         batch_targets: torch.Tensor,
#         loss_slice: slice | torch.Tensor,
#         num_samples: int | None = None,
#         temperature: float = 1.0,
#         loss_func: str = "ce-all",
#         cw_margin: float = 1e-3,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         num_samples = num_samples or len(batch_input_ids)
#         input_embeds = self.embed_layer(batch_input_ids)

#         # logits: [batch_size, seq_len, vocab_size]
#         logits = self.model(
#             inputs_embeds=input_embeds,
#             past_key_values=self._get_batch_prefix_cache(len(batch_input_ids)),
#         ).logits[:num_samples]

#         # loss_logits: [batch_size, loss_len, vocab_size]
#         if isinstance(loss_slice, slice):
#             loss_logits = logits[:, loss_slice]
#         else:
#             loss_logits = logits.gather(1, loss_slice)

#         if batch_targets.dtype == torch.long:
#             # Hard-label target usually used for computing loss on target
#             if "ce" in loss_func:
#                 loss = F.cross_entropy(
#                     loss_logits.permute(0, 2, 1) / temperature,
#                     batch_targets,
#                     reduction="none",
#                 ).mean(dim=1)
#             elif "cw" in loss_func:
#                 loss = _cw_loss(loss_logits, batch_targets, cw_margin=cw_margin)
#             else:
#                 raise ValueError(f"Unknown loss_func: {loss_func}!")
#         else:
#             # Soft-label target usually used for training proxy model
#             loss = F.kl_div(
#                 (loss_logits / temperature).log_softmax(dim=-1),
#                 batch_targets / temperature,
#                 reduction="none",
#             )
#             loss = loss.sum(dim=-1).mean(dim=1)
#         assert loss.shape == (num_samples,), loss.shape
#         return loss_logits, loss

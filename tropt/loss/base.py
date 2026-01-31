import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    def __call__(self, *args, **kwargs) -> Float[Tensor, "bsz"]:
        pass

    def contains_loss_type(self, loss_type: type) -> bool:
        """
        Check if the loss is of the specified type.
        Complicated losses (e.g., CombinedLoss) may override this method with different logic.
        """
        return isinstance(self, loss_type)


############################
@dataclass
class LogitBasedLoss(BaseLoss):
    """
    Loss is computed based on model output logits.
    These losses required target tokens (i.e. `target_outputs_toks`); commonly automatically derived from `target_outputs` strings.
    """

    TARGET_KEY: str = "target_outputs_toks"

    def __call__(
        self,
        logits: Float[Tensor, "bsz target_seq_len vocab_size"],
        target_ids: Float[Tensor, "bsz target_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError()


@dataclass
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


@dataclass
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
    ) -> Float[Tensor, "bsz"]:
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


@dataclass
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
    ) -> Float[Tensor, "bsz"]:
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


############################
@dataclass
class TriggerLogitBasedLoss(BaseLoss):
    """
    Loss is computed based on model output logits *on the trigger tokens*.
    Useful for optimizing properties of the triggers directly.
    """

    def __call__(
        self,
        trigger_logits: Float[Tensor, "bsz trigger_seq_len vocab_size"],
        trigger_ids: Float[Tensor, "bsz trigger_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError()


@dataclass
class TriggerPerplexityLoss(TriggerLogitBasedLoss):
    """
    Calculates perplexity, which is exp(cross_entropy).
    Useful for penalizing non-fluent triggers.
    """

    temperature: float = 1.0

    def __call__(
        self,
        trigger_logits: Float[Tensor, "bsz seq_len vocab_size"],
        trigger_ids: Int[Tensor, "bsz seq_len"],
        ignore_index: int = -100,
    ) -> Float[Tensor, "bsz"]:
        trigger_logits = trigger_logits / self.temperature

        assert (
            trigger_logits.ndim == 3 and trigger_logits.shape[:2] == trigger_ids.shape[:2]
        ), "Shape mismatch"

        # Reuse the exact logic from PrefillCELoss
        ce_loss_fn = PrefillCELoss(temperature=self.temperature)
        ce_loss = ce_loss_fn(
            trigger_logits,
            trigger_ids,
            ignore_index=ignore_index
        )  # (bsz,)

        return torch.exp(ce_loss)

#############################
@dataclass
class AttentionBasedLoss(BaseLoss):
    """Loss is computed based on model attention weights."""

    def __call__(
        self,
        attentions: Float[Tensor, "bsz n_layers n_heads seq_len[dst] seq_len[src]"],
        slices: List[dict[str, slice]] = {},  # of length bsz
    ) -> Float[Tensor, "bsz"]:
        raise NotImplementedError()


@dataclass
class AttentionEnhLoss(AttentionBasedLoss):
    """
    Encourages attention from the trigger tokens to the chat template after the adversarial trigger.
    Note: the sign of the loss is set such that minimizing the loss maximizes the attention.

    Enable to instantiate the (different) losses from:
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
            logger.debug("Note: `chat_template_after` is currently only correct for LMs and on suffix attacks. If the usage is different, somethings may break, or worse -- be wrong.")
        slc_src = [
            message_slices.get(self.src_slc_name, slice(None))
            for message_slices in slices
        ]
        slc_dst = [
            message_slices.get(self.dst_slc_name, slice(None))
            for message_slices in slices
        ]

        # TODO vectorize this process
        loss = torch.zeros(attentions.shape[0], device=attentions.device)  # (bsz,)
        for i, (curr_slc_src, curr_slc_dst) in enumerate(zip(slc_src, slc_dst)):
            loss[i] = attentions[
                i, self.targeted_layers, :, curr_slc_dst, curr_slc_src
            ].mean()

        loss *= -1  # maximize attention

        return loss


############################
@dataclass
class EmbeddingBasedLoss(BaseLoss):
    """Loss is computed based on model embeddings, compared to given target vectors.

    Requires the target vectors (shape: (n_messages, d_model)) to be provided in the targets dict.
    """

    TARGET_KEY = "target_vectors"  # shape: (n_messages, d_model)


@dataclass
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

@dataclass  # TODO implement
class TextBasedLoss(BaseLoss):
    """Mixin for models that can compute losses based on model outputs (embedding for encoder, text response for LMs); fits query access."""

    pass


@dataclass
class ResponseLMScoreLoss(TextBasedLoss):
    """A loss based on an LM-as-a-judge score of the model's response."""

    pass

############################

@dataclass  # TODO implement
class TriggerTextBasedLoss(BaseLoss):
    """Mixin for models that can compute losses based on the trigger text directly."""

    pass


############################
@dataclass
class HiddenStateBased(BaseLoss):
    """Loss computed based on model hidden states."""

    pass


@dataclass
class SteeringActivationLoss(HiddenStateBased):
    """
    Encourages hidden activations at specific layers/positions to align with a target direction.
    - Each message has a target direction vector (optionally its own unique one).
        - target_directions: (n_messages, d_model)
        - Note that the direction will be applied to the whole target positions and layers.
    - Default is steering *towards* a direction (maximizing alignment).
        - Here, minimizing the loss maximizes alignment (dot product) with the target direction.
        - Set steer_away=True to steer *away* (e.g., for refusal suppression).

    References:
    - Was proposed as 'refusal direction suppression' combined with GCG:
        https://aclanthology.org/2025.naacl-long.302/
    - Was proposed for adapting attacks (e.g., GCG) for evading probe-based classifiers.
        https://arxiv.org/abs/2412.09565

    Args:
        targeted_layers: Which layers to apply steering on (default: all layers)
        steer_away: Whether to minimize alignment instead of maximizing (default: False = steer towards)
        slc_name: Which token positions to apply steering on (default: "last_input_token")
        do_cosine_sim: Whether to use cosine similarity instead of dot product (default: False)
    """

    TARGET_KEY = "target_directions"  # shape: (n_messages, d_model)

    targeted_layers: slice = slice(None)
    steer_away: bool = False
    slc_name: str = "last_input_token"
    do_cosine_sim: bool = False

    def __call__(
        self,
        hidden_states: Float[Tensor, "bsz n_layers seq_len d_model"],
        target_directions: Float[Tensor, "bsz d_model"],
        slices: List[dict[str, slice]] = None,  # of length bsz
    ) -> Float[Tensor, "bsz"]:
        """
        Compute steering loss by measuring cosine similarity between hidden states and target directions.

        Args:
            hidden_states: Model hidden states from all layers and positions (bsz, n_layers, seq_len, d_model)
            target_directions: Direction vectors to align with (bsz, d_model)
            slices: Position slices for each batch element (e.g., {"adv": slice(10, 30)})

        Returns:
            Loss tensor of shape (bsz,).
        """
        target_directions = target_directions.to(hidden_states.device)

        # Normalize target directions
        target_directions = target_directions / target_directions.norm(dim=-1, keepdim=True)

        # Extract slices for the tokens we want to steer
        if slices is None:
            slices = [{}] * hidden_states.shape[0]

        slc_list = [
            message_slices.get(self.slc_name, slice(None))
            for message_slices in slices
        ]

        # Compute cosine similarity for each batch element
        loss = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        for i, curr_slc in enumerate(slc_list):
            # Extract hidden states for targeted layers and positions
            h = hidden_states[i, self.targeted_layers, curr_slc, :]  # (n_layers, slc_seq_len, d_model)

            if self.do_cosine_sim:
                # Normalize hidden states
                h = h / h.norm(dim=-1, keepdim=True)

            # Compute dot product with target direction
            # (n_layers, slc_seq_len, d_model) * (1, 1, d_model)
            #  mult -> (n_layers, slc_seq_len, d_model)
            #  sum  -> (n_layers, slc_seq_len)
            res = (h * target_directions[i].unsqueeze(0).unsqueeze(0)).sum(dim=-1)

            # Average over layers and positions
            loss[i] = res.mean()

        # Apply sign based on steering direction
        if not self.steer_away:
            # Default: steer towards (maximize alignment)
            # Negate so minimizing loss maximizes dot product
            loss = -loss
        # else: steer away (minimize alignment)
        # Keep positive so minimizing loss minimizes dot product

        return loss

############################

@dataclass
class CombinedLoss(BaseLoss):
    """Combines multiple losses with given weights."""

    def __init__(self, loss_funcs: List[BaseLoss], weights: List[float] = None) -> None:
        assert weights is None or len(loss_funcs) == len(weights), "Length mismatch"
        assert all(isinstance(loss, BaseLoss) for loss in loss_funcs), "All elements in losses must be instances of BaseLoss"
        assert all(not isinstance(loss, CombinedLoss) for loss in loss_funcs), "CombinedLoss cannot contain another CombinedLoss"
        self.loss_funcs: List[BaseLoss] = loss_funcs

        if weights is None:
            # if weights not provided, set equal weights
            self.weights: Float[Tensor, "n_losses"] = torch.ones(len(loss_funcs)) / len(loss_funcs)
        else:
            self.weights: Float[Tensor, "n_losses"] = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, losses: Float[Tensor, "n_losses bsz"]) -> Float[Tensor, "bsz"]:
        """
        Compute the combined loss (weighted sum).
        Args:
            losses: Tensor of shape (n_losses, bsz), each row corresponds to the loss values from each loss function.
        Returns:
            Tensor of shape (bsz,), the combined loss for each element in the batch.
        """
        weights = self.weights.to(losses).unsqueeze(-1)  # shape: (n_losses, 1)
        loss = losses * weights
        loss = loss.sum(dim=0)  # recude over n_losses

        return loss  # shape: (bsz,)

    def contains_loss_type(self, loss_type: type) -> bool:
        """Check if the CombinedLoss contains a loss of the specified type."""
        return any(isinstance(loss, loss_type) for loss in self.loss_funcs)

    def __iter__(self):
        """Allows iterating over the nested loss functions."""
        return iter(self.loss_funcs)



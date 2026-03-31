"""
General loss functions.

Important note: The losses arguments must match the fields in ModelOutput and ModelInput
for unified loss resolution to work properly.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, ClassVar, List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import SliceKey
from tropt.loss.base import BaseLoss
from tropt.loss.utils import IGNORE_INDEX, masked_mean

logger = logging.getLogger(__name__)

############################
@dataclass
class PrefillBasedLoss(BaseLoss):
    """
    Loss computed on prefilled response logits (`prefill_response_logits`).
    Requires target tokens (`target_response_toks`); commonly derived from `target_response_strs`.
    Using this loss usually implies that the model will prefill the response with these target tokens.
    """

    require_target_prefill: ClassVar[bool] = True

    @abstractmethod
    def __call__(
        self,
        prefill_response_logits: Float[Tensor, "bsz response_seq_len vocab_size"],
        target_response_toks: Int[Tensor, "response_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        pass


@dataclass
class PrefillCELoss(PrefillBasedLoss):
    """
    Encourages (=maximize likelihood) the model to produce the target output (mostly an affirmative response).
    """

    temperature: float = 1.0  # for softmax

    def __call__(
        self,
        prefill_response_logits: Float[Tensor, "bsz response_seq_len vocab_size"],
        target_response_toks: Int[Tensor, "response_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        target_response_toks = target_response_toks.unsqueeze(0).expand(
            prefill_response_logits.shape[0], -1
        )  # (bsz, response_seq_len)
        prefill_response_logits = prefill_response_logits / self.temperature
        assert (
            prefill_response_logits.ndim == 3
            and target_response_toks.ndim == 2
            and prefill_response_logits.shape[1] == target_response_toks.shape[1]
        ), f"Shape mismatch: prefill_response_logits {prefill_response_logits.shape}, target_response_toks {target_response_toks.shape}"

        loss = torch.nn.functional.cross_entropy(
            prefill_response_logits.transpose(-1, -2),  # move vocab size (= # classes) to 2nd dim
            target_response_toks,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )  # (bsz, seq_len)

        return masked_mean(loss, (target_response_toks != IGNORE_INDEX).float())


@dataclass
class PrefillMellowMaxLoss(PrefillBasedLoss):
    """
    Encourages the model to produce the target output by maximizing the mellowmax of the target logits.
    https://arxiv.org/pdf/1612.05628, http://confirmlabs.org/posts/TDC2023
    """

    mellowmax_alpha: float = 1.0
    temperature: float = 1.0

    def __call__(
        self,
        prefill_response_logits: Float[Tensor, "bsz response_seq_len vocab_size"],
        target_response_toks: Int[Tensor, "response_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        target_response_toks = target_response_toks.unsqueeze(0).expand(
            prefill_response_logits.shape[0], -1
        )  # (bsz, response_seq_len)
        prefill_response_logits = prefill_response_logits / self.temperature
        assert prefill_response_logits.shape[:-1] == target_response_toks.shape, "Shape mismatch"

        # 1. Create mask
        mask = target_response_toks != IGNORE_INDEX
        # replace ignore index with 0 to avoid index error (will be masked later anyway)
        target_response_toks = target_response_toks.masked_fill(~mask, 0)

        # 2. Gather the logits corresponding to the target IDs
        target_logits = prefill_response_logits.gather(-1, target_response_toks.unsqueeze(-1)).squeeze(-1)
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
class PrefillCWLoss(PrefillBasedLoss):
    """
    Encourages (=maximize likelihood) the model to produce the target output (mostly an affirmative response).
    CW-inspired hinge loss on the difference between the largest and the target logits.
    https://arxiv.org/abs/2402.09674
    """

    cw_margin: float = 1e-3
    first_token_weight: float = 1.0

    def __call__(
        self,
        prefill_response_logits: Float[Tensor, "bsz response_seq_len vocab_size"],
        target_response_toks: Int[Tensor, "response_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        target_response_toks = target_response_toks.unsqueeze(0).expand(
            prefill_response_logits.shape[0], -1
        )  # (bsz, response_seq_len)
        assert prefill_response_logits.shape[:2] == target_response_toks.shape, (prefill_response_logits.shape, target_response_toks.shape)
        vocab_dim: int = -1  # dimension of vocab size

        # Create mask and safe indices
        mask = target_response_toks != IGNORE_INDEX
        target_response_toks = target_response_toks.masked_fill(~mask, 0)  # replace ignore index with 0 to avoid index error (will be masked later anyway)

        # extract the target's logits (using the target ids as indices)
        tgt_logits = prefill_response_logits.gather(vocab_dim, target_response_toks.unsqueeze(-1)).squeeze(-1)

        # Set logits of target tok to -inf so it cannot be the largest
        tmp_logits = prefill_response_logits.clone()
        tmp_logits.scatter_(vocab_dim, target_response_toks.unsqueeze(-1), -torch.inf)

        # pick the largest logit among the non-target tokens
        largest_non_tgt_logits = tmp_logits.max(vocab_dim).values

        # calculate the CW loss:
        loss = largest_non_tgt_logits - tgt_logits
        loss = loss.clamp_min(-self.cw_margin)

        # Apply first-token weighting (e.g., to emphasize the affirmative "Sure" token)
        if self.first_token_weight != 1.0:
            weights = torch.ones_like(loss)
            weights[:, 0] = self.first_token_weight
            loss = loss * weights

        # Zero out loss for padding tokens
        loss = loss * mask.float()

        return masked_mean(loss, mask.float())


############################
@dataclass
class TriggerLogitBasedLoss(BaseLoss):
    """
    Loss computed on full-sequence logits (`full_logits`) sliced to trigger positions.
    Useful for optimizing properties of the triggers directly.
    """

    @abstractmethod
    def __call__(
        self,
        full_logits: Float[Tensor, "bsz seq_len vocab_size"],
        input_trigger_ids: Int[Tensor, "trigger_seq_len"],
        input_slices: dict[SliceKey, slice],
    ) -> Float[Tensor, "bsz"]:
        pass


@dataclass
class TriggerPerplexityLoss(TriggerLogitBasedLoss):
    """
    Calculates perplexity wrt to the target model logits themselves.
    Useful for penalizing non-fluent triggers.
    """

    temperature: float = 1.0
    slc_name: SliceKey = SliceKey.TRIGGER  # Which slice contains the trigger tokens

    def __call__(
        self,
        full_logits: Float[Tensor, "bsz seq_len vocab_size"],
        input_trigger_ids: Int[Tensor, "bsz trigger_seq_len"],
        input_slices: dict[SliceKey, slice],
    ) -> Float[Tensor, "bsz"]:
        """
        Compute perplexity loss on the trigger tokens.
        Perplexity is computed wrt to the trigger logits in `full_logits` and the target ids in `input_trigger_ids`.
        """

        # Extract trigger logits
        slc = input_slices[self.slc_name]
        assert slc.start > 0, (
            f"TriggerPerplexityLoss requires the trigger slice to start after position 0 "
            f"(got slc.start={slc.start}). This could happen since loss is incompatible with `use_prefix_cache=True` -- try setting it to `False`."
        )
        trigger_logits = full_logits[:, slc.start - 1 : slc.stop - 1, :]  # (bsz, trigger_seq_len, vocab_size)
        trigger_logits = trigger_logits / self.temperature

        assert (
            trigger_logits.ndim == 3 and trigger_logits.shape[:2] == input_trigger_ids.shape[:2]
        ), f"Shape mismatch: trigger_logits {trigger_logits.shape}, input_trigger_ids {input_trigger_ids.shape}"

        loss = torch.nn.functional.cross_entropy(
            trigger_logits.transpose(-1, -2),  # (bsz, vocab_size, trigger_seq_len)
            input_trigger_ids,  # (bsz, trigger_seq_len)
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )  # (bsz, trigger_seq_len)

        ce_loss = masked_mean(loss, (input_trigger_ids != IGNORE_INDEX).float())

        return ce_loss

#############################
@dataclass
class AttentionBasedLoss(BaseLoss):
    """Loss computed on model attention weights (`full_attentions`)."""

    require_attentions: ClassVar[bool] = True

    @abstractmethod
    def __call__(
        self,
        full_attentions: Float[Tensor, "bsz n_layers n_heads seq_len[dst] seq_len[src]"],
        input_slices: dict[SliceKey, slice],
    ) -> Float[Tensor, "bsz"]:
        pass


@dataclass
class AttentionEnhLoss(AttentionBasedLoss):
    """
    Encourages attention from the trigger tokens to the chat template after the adversarial trigger.
    *Note*: the sign of the loss is set such that minimizing the loss maximizes the attention.

    Enable to instantiate the (different) losses from:
    https://arxiv.org/abs/2506.12880, https://arxiv.org/abs/2410.09040
    """

    targeted_layers: slice = slice(None)
    src_slc_name: SliceKey = SliceKey.TRIGGER
    dst_slc_name: SliceKey = SliceKey.INPUT_AFTER

    def __call__(
        self,
        full_attentions: Float[Tensor, "bsz n_layers n_heads seq_len[dst] seq_len[src]"],
        input_slices: dict[SliceKey, slice],
    ) -> Float[Tensor, "bsz"]:
        if SliceKey.INPUT_AFTER in (self.src_slc_name, self.dst_slc_name):
            logger.debug("Note: `chat_template_after` is currently only correct for LMs and on suffix attacks. If the usage is different, somethings may break, or worse -- be wrong.")
        slc_src = input_slices.get(self.src_slc_name, slice(None))
        slc_dst = input_slices.get(self.dst_slc_name, slice(None))

        # Extract attention weights for target slices and layers
        attn_subset = full_attentions[
            :, self.targeted_layers, :, slc_dst, slc_src
        ]  # (bsz, n_targeted_layers, n_heads, dst_len, src_len)

        # Mean over all dimensions except batch: layers, heads, dst, src
        loss = attn_subset.mean(dim=tuple(range(1, attn_subset.ndim)))  # (bsz,)
        loss *= -1  # maximize attention

        return loss


############################
@dataclass
class EmbeddingBasedLoss(BaseLoss):
    """Loss is computed based on model embeddings, compared to given target vectors.

    Requires the target vectors (shape: (n_templates, d_model)) to be provided in the targets dict.
    """

    @abstractmethod
    def __call__(
        self,
        output_embeddings: Float[Tensor, "bsz d_model"],
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        pass

@dataclass
class SimilarityLoss(EmbeddingBasedLoss):
    """
    Encourages given representation(s) to align (cos-sim) with the given target vectors.
    """

    def __call__(
        self,
        output_embeddings: Float[Tensor, "bsz d_model"],
        target_vectors: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "bsz"]:
        target_vectors = target_vectors.unsqueeze(0).expand(output_embeddings.shape[0], -1)  # (bsz, d_model)
        assert output_embeddings.ndim == target_vectors.ndim == 2, "Shape mismatch"
        target_vectors = target_vectors.to(output_embeddings.device)

        # normalize:
        output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)
        target_vectors = target_vectors / target_vectors.norm(dim=-1, keepdim=True)

        # cosine similarity via normalized dot product:
        cos_sim = (output_embeddings * target_vectors).sum(dim=-1, keepdim=True)
        loss = -1 * cos_sim  # maximize cos-sim <=> minimize (-1 * cos-sim)

        return loss.squeeze(-1)


############################
@dataclass
class HiddenStateBasedLoss(BaseLoss):
    """Loss computed on model hidden states (`full_hidden_states`)."""

    require_hidden_states: ClassVar[bool] = True

    @abstractmethod
    def __call__(
        self,
        full_hidden_states: Float[Tensor, "bsz n_layers seq_len d_model"],
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        pass


@dataclass
class SteeringActivationLoss(HiddenStateBasedLoss):
    """
    Encourages hidden activations at specific layers/positions to align with a target direction.
    - Each message has a target direction vector (optionally its own unique one).
        - target_directions: (n_templates, d_model)
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
        apply_square: Whether to square the similarity scores (default: False)
    """

    targeted_layers: slice = slice(None)
    steer_away: bool = False
    slc_name: str = SliceKey.INPUT_LAST_TOKEN
    do_cosine_sim: bool = False
    apply_square: bool = False
    apply_abs: bool = False

    def __call__(
        self,
        full_hidden_states: Float[Tensor, "bsz n_layers seq_len d_model"],
        target_directions: Float[Tensor, "d_model"],
        input_slices: Optional[dict[str, slice]] = None,
    ) -> Float[Tensor, "bsz"]:
        """
        Compute steering loss by measuring cosine similarity between hidden states and target directions.

        Args:
            output_hidden_states: Model hidden states from all layers and positions (bsz, n_layers, seq_len, d_model)
            target_directions: Direction vectors to align with (, d_model)
            input_slices: Position slices reflecting the input tokens (dict mapping slice names to slices)

        Returns:
            Loss tensor of shape (bsz,).
        """
        target_directions = target_directions.to(full_hidden_states.device)
        target_directions = target_directions.unsqueeze(0).expand(
            full_hidden_states.shape[0], -1
        )  # (bsz, d_model)

        # Normalize target directions
        target_directions = target_directions / target_directions.norm(dim=-1, keepdim=True)

        # Extract slices for the tokens we want to steer
        if input_slices is not None:
            slc = input_slices.get(self.slc_name, slice(None))
        else:
            slc = slice(None)  # Use all positions if no slices provided

        # (bsz, n_targeted_layers, slc_seq_len, d_model)
        h = full_hidden_states[:, self.targeted_layers, slc, :]

        if self.do_cosine_sim:
            h = h / h.norm(dim=-1, keepdim=True)

        # (bsz, 1, 1, d_model) -> broadcast dot product -> (bsz, n_targeted_layers, slc_seq_len)
        res = (h * target_directions[:, None, None, :]).sum(dim=-1)

        # Optionally {abs, square, ..} the similarity scores
        if self.apply_abs:
            res = res.abs()
        if self.apply_square:
            res = res.pow(2)

        # Average over layers and positions -> (bsz,)
        loss = res.mean(dim=(-1, -2))

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
class ClassificationBasedLoss(BaseLoss):
    """Loss computed on classifier logits (`output_class_logits`)."""

    @abstractmethod
    def __call__(
        self,
        output_class_logits: Float[Tensor, "bsz n_classes"],
        **kwargs,
    ) -> Float[Tensor, "bsz"]:
        pass


@dataclass
class MisclassCELoss(ClassificationBasedLoss):
    """Encourages misclassification via cross-entropy on classifier logits.

    Two modes:
    - Untargeted (targeted=False): minimizes probability of `true_class_idx`.
    - Targeted (targeted=True): maximizes probability of `target_class_idx`.
    """

    true_class_idx: Optional[int] = None
    target_class_idx: Optional[int] = None
    targeted: bool = False

    def __post_init__(self):
        if self.targeted:
            assert self.target_class_idx is not None, (
                "target_class_idx must be provided for targeted misclassification."
            )
        else:  # untargeted mode
            assert self.true_class_idx is not None, (
                "true_class_idx must be provided for untargeted misclassification."
            )

    def __call__(
        self,
        output_class_logits: Float[Tensor, "bsz n_classes"],
    ) -> Float[Tensor, "bsz"]:
        bsz = output_class_logits.shape[0]
        device = output_class_logits.device

        if self.targeted:
            # Minimize CE w.r.t. target class => maximize target class probability
            target = torch.full(
                (bsz,), self.target_class_idx, dtype=torch.long, device=device
            )
            return torch.nn.functional.cross_entropy(
                output_class_logits, target, reduction="none"
            )
        else:  # untargeted mode
            # Negate CE w.r.t. true class => minimize true class probability
            target = torch.full(
                (bsz,), self.true_class_idx, dtype=torch.long, device=device
            )
            return -torch.nn.functional.cross_entropy(
                output_class_logits, target, reduction="none"
            )


import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float
from torch import Tensor

from tropt.common import (
    MessageTargets,
    ModelInput,
    ModelOutput,
    Targets,
    TextTemplates,
    TokenTriggerCandidates,
)
from tropt.loss import BaseLoss
from tropt.loss.resolution import resolve_and_compute_loss

from .inputs_manager import (
    TextInputManager,
    TokenInputManager,
)
from .model_base import BaseTokenizer

logger = logging.getLogger(__name__)

# ======================================================================
# Token Access Flow
# ======================================================================

class TokenAccessMixin(ABC):
    """Mixin for models that have a tokenizer and can prepare token-level inputs.

    This is the base mixin for any model with token-level access (tokenizer,
    set/reset inputs). Note that such models may not have access to _compute_ the loss from tokens (see LossTokenAccessMixin), but they must be able to at least prepare the token inputs (e.g., OpenAI models).
    """

    _token_input_manager: Optional[TokenInputManager] = None

    @abstractmethod
    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepare and store the inputs manager as self._token_input_manager.

        Args:
            templates: List of text templates containing the trigger placeholder.
            targets: Optional targets for the loss function.
        """
        raise NotImplementedError

    def reset_inputs_from_tokens(self) -> None:
        """Clear self._token_input_manager."""
        self._token_input_manager = None

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self) -> BaseTokenizer:
        """
        Force the class using this mixin to implement a tokenizer.
        This tokenizer implement API defined by BaseTokenizer, which matches the main functionality of  HuggingFace tokenizer.
        """
        raise NotImplementedError


class InvokeTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can perform a forward pass from token-level inputs.

    Adds the abstract invoke_from_tokens method. All compute-* mixins
    (LossTokenAccessMixin, GradientTokenAccessMixin, etc.) inherit from this.
    """
    @abstractmethod
    def invoke_from_tokens(
        self,
        input_ids: Optional[Float[Tensor, "bsz seq_len"]] = None,

        message_targets: Optional[MessageTargets] = None,
        require_target_prefill: bool = False,
        require_generation: bool = False,
        **kwargs
    ) -> ModelOutput:
        """Perform a forward pass from token-level (embedding) inputs.

        Args:
            input_ids: Token IDs of the full input sequence (incl. trigger), plus optionally target tokens. Shape: (batch_size, seq_len).

            message_targets: Optional MessageTargets object containing the targets for the messages.
            require_target_prefill: Whether to prefill the target response from `message_targets`, and return the corresponding logits (e.g., for LMs).
            require_generation: Whether to perform autoregressive generation after the forward pass (for LMs).

        Returns:
            ModelOutput with the fields this model can provide.
        """
        raise NotImplementedError


## "Grey-box" Model Mixins:
class LossTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute losses based on token-level inputs."""

    @abstractmethod
    def compute_loss_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "n_templates n_candidates"]:
        """Compute the loss on the stored token inputs with the given trigger merged in."""
        raise NotImplementedError


class LogitsTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute logits based on token-level inputs."""

    @abstractmethod
    def compute_logits_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute logits w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError


## "White-box" Model Mixins:
class GradientTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute gradients w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError

## "White-box" Model Mixins w/ embed access:
class GradientEmbedAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute gradients from the *input embeddings* based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_embeds(
        self,
        loss_func: BaseLoss,
        candidate_trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embed_dim"],
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len embed_dim"]:
        """Compute gradients w.r.t. `trigger` embeddings using stored token inputs."""
        raise NotImplementedError


# ======================================================================
# Text Access Flow
# ======================================================================

class TextAccessMixin(ABC):
    _text_input_manager: Optional[TextInputManager] = None

    def set_inputs_from_texts(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepare and store the text-based inputs manager."""
        self._text_input_manager = TextInputManager(
            templates=templates,
            targets=targets,
        )

    def reset_inputs_from_texts(self) -> None:
        """Clear the stored text input manager."""
        self._text_input_manager = None


class LossTextAccessMixin(TextAccessMixin):
    """Mixin for models that compute losses based on text-level inputs (black-box access)."""

    @torch.no_grad()
    def compute_loss_from_texts(
        self,
        candidate_trigger_strs: List[str],
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"]:
        """
        Computes the loss on all candidate string texts using the stored text inputs manager.
        This computation is based on the invoke_from_texts() method of the model, and the information it provides.
        """
        assert self._text_input_manager is not None, "Text input manager is not initialized. Please call set_inputs_from_texts() first."

        input_manager = self._text_input_manager
        n_templates: int = input_manager.n_templates
        n_candidates: int = len(candidate_trigger_strs)  # noqa

        # Main Loop: for each template, we compute the loss for all candidates.
        # Candidates are chunked up to `self._forward_pass_batch_size` (defined
        # on BaseModel, overridable per subclass); on CUDA OOM we halve the
        # batch size and retry (via accelerate's find_executable_batch_size).
        losses = []
        for template_idx in range(n_templates):

            @find_executable_batch_size(starting_batch_size=self._forward_pass_batch_size)
            def _compute_loss_batched(batch_size, template_idx=template_idx):
                # --- Update forward batch size ---
                # Automatically lower the default for future calls if this run required a downgrade
                if batch_size < self._forward_pass_batch_size:
                    logger.info(f"OOM detected. Reducing _forward_pass_batch_size from {self._forward_pass_batch_size} to {batch_size}")
                    self._forward_pass_batch_size = batch_size
                # --------------------

                chunk_losses = []
                for start in range(0, n_candidates, batch_size):
                    end = min(start + batch_size, n_candidates)
                    chunk_strs = candidate_trigger_strs[start:end]

                    model_input: ModelInput = input_manager.get_triggered_inputs(
                        chosen_template_idx=template_idx, trigger_strs=chunk_strs,
                    )

                    # Only enable gradient is it's required by the loss (e.g. for gradient matching losses); mostly false.
                    with torch.set_grad_enabled(loss_func.require_gradients):
                        # Forward pass for this candidate chunk
                        model_output: ModelOutput = self.invoke_from_texts(
                            input_texts=model_input.input_texts,
                            message_targets=model_input.message_targets,
                            require_target_prefill=loss_func.require_target_prefill,
                            require_generation=loss_func.require_generation,
                            require_first_token_logprobs=loss_func.require_first_token_logprobs,
                        )

                        # Use unified loss resolution
                        chunk_loss = resolve_and_compute_loss(
                            model_output, model_input, loss_func
                        )  # shape: (chunk_size,)
                        chunk_losses.append(chunk_loss)

                return torch.cat(chunk_losses, dim=0)  # shape: (n_candidates,)

            losses.append(_compute_loss_batched())

        losses = torch.stack(losses, dim=0)  # shape: (n_templates, n_candidates)

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # shape: (n_candidates,)

        return losses

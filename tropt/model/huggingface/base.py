import functools
import inspect
import itertools
import logging
from abc import abstractmethod
from functools import cached_property
from typing import Annotated, Any, Dict, List, Optional, Tuple

import torch
import transformers
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float, Int
from torch import Tensor
from transformers.cache_utils import DynamicCache

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelInput,
    ModelOutput,
    SliceKey,
    Targets,
    is_debug_mode,
)
from tropt.loss import BaseLoss
from tropt.loss.resolution import resolve_and_compute_loss
from tropt.model import (
    TokenInputManager,
)
from tropt.model.model_base import HFTokenizerWrapper

logger = logging.getLogger(__name__)

# ======================= Input/Output Handlers logic =======================


class HuggingFaceTokenInputManager(TokenInputManager):
    """
    HuggingFace implementation of the TokenInputManager.

    Implementation Notes:
        - The inputs templates is split into *before* and *after* the trigger parts, and then maintained separately by the input manager. Then, given trigger candidates, we craft multiple inputs from the templates.
        - HuggingFaceBackend ensures that the `OPTIMIZED_TRIGGER_PLACEHOLDER` token is registered
        in the tokenizer (as a _single_ token), so we can reliably split the templates into
        before/after trigger parts by this input manager.
        - Each input template is handled separately, as it they may vary in length, have their own targets, etc.
    """

    before_ids: Annotated[List[Float[Tensor, "bef_len"]], "n_templates"]
    after_ids: Annotated[List[Float[Tensor, "aft_len"]], "n_templates"]
    embed_func: torch.nn.Module
    targets: Targets
    padding_side: str
    pad_token_id: int
    tokenizer: transformers.PreTrainedTokenizerBase

    # Optional prefix cache (for models that support it)
    prefix_cache: Optional[List[tuple]] = None

    @torch.no_grad()
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: torch.device,
        templates_ids: Annotated[List[List[int]], "n_templates seq_len"],
        embed_func: torch.nn.Module,
        targets: Targets,
        use_prefix_cache: Optional[bool] = False,
        model: Optional[transformers.PreTrainedModel] = None,
    ):
        """
        Initializes the HuggingFace input manager for a given tokenzier and input templates.

        Args:
            tokenizer: The HuggingFace tokenizer to use.
            device: The device to use for any tensors created by the input manager.
            templates_ids: the user-provided input templates tokenized into token ids, as a list of lists of
            ints (n_templates, seq_len). These are expected to include all special tokens, including the
            chat template in the case of instruction-tuned LMs.
            embed_func: The embedding function (torch module) to use for embedding token ids into vectors.

            use_prefix_cache: Whether to compute and use the prefix cache for the part before the trigger (as
            it is static throughout optimization).
            model: The HuggingFace model, required if `use_prefix_cache` is True, to compute the prefix cache.
            targets: The Targets object containing the optimization targets for the input templates.

        """
        self.padding_side = tokenizer.padding_side
        self.pad_token_id = tokenizer.pad_token_id

        ## Split texts into before/after optimized trigger parts
        placeholder_id = tokenizer.convert_tokens_to_ids(
            OPTIMIZED_TRIGGER_PLACEHOLDER  # expected to be a single token
        )
        before_ids, after_ids = [], []
        before_texts, after_texts = [], []
        for ids in templates_ids:  # iterate on each template
            # extract trigger position:
            ids = torch.tensor(ids, device=device, dtype=torch.int64)
            trig_positions = (ids == placeholder_id).nonzero(as_tuple=True)[0]
            assert len(trig_positions) == 1, (
                f"Expected exactly 1 '{OPTIMIZED_TRIGGER_PLACEHOLDER}' token in the template "
                f"token sequence, found {len(trig_positions)}."
            )
            trig_pos = trig_positions.item()
            # split into before/after trigger parts:
            before_ids.append(ids[:trig_pos])
            after_ids.append(ids[trig_pos + 1:])
            before_texts.append(tokenizer.decode(ids[:trig_pos].tolist()))
            after_texts.append(tokenizer.decode(ids[trig_pos + 1:].tolist()))

        self.before_ids = before_ids
        self.before_texts = before_texts
        self.after_ids = after_ids
        self.after_texts = after_texts
        self.embed_func = embed_func
        self.tokenizer = tokenizer

        # Prepare targets
        self.targets: Targets = targets.to_device(device)

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
            assert model is not None, "Model must be provided to compute prefix cache."
            assert model.device == device, f"Model device {model.device} does not match input manager device {device}."  # sanity check
            for i in range(self.n_templates):
                # (seq, emb) -> (1, seq, emb)
                curr_embeds = self.before_embeds[i].unsqueeze(0)
                curr_attn_mask = torch.ones(
                    curr_embeds.shape[:2], device=device, dtype=torch.int64
                )
                output = model(
                    inputs_embeds=curr_embeds,
                    attention_mask=curr_attn_mask,
                    use_cache=True,
                )
                curr_prefix = tuple((item[0], item[1]) for item in output.past_key_values)
                # tuple(layers) of tuple(k, v) where k,v are (1, n_head, seq_len, head_dim)
                prefix_cache.append(curr_prefix)

        self.prefix_cache = prefix_cache if use_prefix_cache else None
        # Memory for formatted prefix cache kwargs, keyed by (batch_size, template_idx)
        self._prefix_cache_kwargs_mem: dict = {}

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
    def before_embeds(self) -> List[Float[Tensor, "bef_len embd_dim"]]:
        return [self.embed_func(ids) for ids in self.before_ids]

    @cached_property
    def after_embeds(self) -> List[Float[Tensor, "aft_len embd_dim"]]:
        return [self.embed_func(ids) for ids in self.after_ids]

    @cached_property
    def pad_token_embeds(self) -> Float[Tensor, "1 embd_dim"]:
        return self.embed_func(torch.tensor([self.pad_token_id], device=self.device))

    def get_triggered_inputs(
        self,
        # trigger options:
        trigger_ids: Optional[Float[Tensor, "n_candidates trigger_seq_len"]] = None,
        trigger_embeds: Optional[Float[Tensor, "n_candidates trigger_seq_len embd_dim"]] = None,

        append_embeds: Optional[List[Float[Tensor, "n_app_ids embd_dim"]]] = None,  # of length n_templates
        do_append_embeds: bool = False,
        chosen_template_idx: Optional[int] = None,
    ) -> ModelInput:
        """
        Returns the input embeddings with the given trigger merged in for a specific message.

        Notes:
        - We do not support varying trigger lengths in the same candidate batch
            (they must share `trigger_seq_len`).
        - for *specific* use cases, the following method is suboptimal; however,
            currently generality and support for different input types/shapes are prioritized.
        - Allows gradient flow through `trigger_embeds`, which can be useful for combining backporable trigger
            candidates (e.g., for `compute_grad_from_*()` methods).


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
            do_append_embeds: If True, the provided `append_embeds` will be used and appended at the end of the input.
            chosen_template_idx: int (required)
                the index of the message to process. Must be provided; multi-message is not supported by this method.

        Returns: A ModelInput object containing:
                - input_trigger_ids: Tensor, shape = (n_candidates, trigger_seq_len)
                    the token ids of the trigger(s) inserted (detached from grad graph; for reference)
                - inputs_embeds: Tensor, shape = (n_candidates, seq_len, embd_dim)
                    the input embeddings with the trigger merged in;
                    if the provided input_embds required grad, then this tensor will also require grad.
                - attention_mask: Tensor, shape = (n_candidates, seq_len)
                    the attention mask matching the input embeddings
                - input_prefix_cache_kwargs: Dict, optional
                    the kwargs to pass to the model forward pass for using the prefix cache on `chosen_template_idx`, if applicable.
                - message_targets: MessageTargets
                    the targets dict for the chosen message, expanded to match n_candidates dimension
        """
        # assert trigger_ids is not None, "`trigger_ids` must be provided to `get_triggered_inputs()`."
        assert chosen_template_idx is not None, "`chosen_template_idx` must be provided to `get_triggered_inputs()`. Multi-message calls should loop over messages."

        if trigger_embeds is None:
            # embed the trigger-ids, if trigger embeddings are not provided
            trigger_embeds = self.embed_func(trigger_ids)
        if do_append_embeds:
            assert append_embeds is not None, "`append_embeds` must be provided if `do_append_embeds` is True."

        n_candidates = trigger_embeds.shape[0]
        template_idx = chosen_template_idx

        ## Construct the parts of the input for this message:
        curr_before = self.before_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
        curr_trigger = trigger_embeds
        curr_after = self.after_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)
        curr_append = None
        if do_append_embeds:
            assert append_embeds is not None
            curr_append = append_embeds[template_idx].unsqueeze(0).repeat(n_candidates, 1, 1)

        # Build embeddings and attention mask
        embeds_parts = []
        attn_parts = []

        # add before part
        if not self.use_prefix_cache:
            # add 'before' part only if not using prefix cache
            embeds_parts.append(curr_before)
        # always add 'before' part attention (even with prefix cache)
        attn_parts.append(torch.ones((n_candidates, curr_before.shape[-2])))

        # add trigger and after trigger parts
        embeds_parts.extend([curr_trigger, curr_after])
        attn_parts.extend([
            torch.ones((n_candidates, curr_trigger.shape[-2])),
            torch.ones((n_candidates, curr_after.shape[-2])),
        ])

        # optionally add the appended part (e.g., target prefiling in LMs)
        if curr_append is not None:
            embeds_parts.append(curr_append)
            attn_parts.append(torch.ones((n_candidates, curr_append.shape[-2])))

        # concatenate parts
        inputs_embeds = torch.cat(embeds_parts, dim=-2)  # (n_candidates, seq_len, embd_dim)
        attention_mask = torch.cat(attn_parts, dim=-1)  # (n_candidates, seq_len)
        attention_mask = attention_mask.to(self.device, torch.int64)

        # Calculate slices for different regions
        # Note: since prefix-caching removes the 'before' part from the input, we need to adjust the slices accordingly
        # (this "removal" will be reflected in HF model outputs, which is where we use the slicing info)
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
        message_targets: MessageTargets = self.targets.select_message(chosen_template_idx)

        ## Prepare prefix cache kwargs (only if both message and batching are provided)
        prefix_cache_kwargs: dict[str, Any] = {}
        if self.use_prefix_cache:
            prefix_cache_kwargs = self._get_prefix_cache_kwargs(
                batch_size=n_candidates,
                template_idx=chosen_template_idx,
            )

        return ModelInput(
            input_trigger_ids=trigger_ids,  # detached triggers for reference
            input_embeds=inputs_embeds.to(self.device, self.float_dtype),
            input_attention_mask=attention_mask.to(self.device, torch.int64),
            input_slices=input_slices,
            message_targets=message_targets,
            input_prefix_cache_kwargs=prefix_cache_kwargs,

            input_trigger_strs=self.tokenizer.batch_decode(trigger_ids) if trigger_ids is not None else None,
        )

    def _get_prefix_cache_kwargs(
        self, batch_size: int = 1, template_idx: Optional[int] = None
    ) -> Dict[str, Any]:
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
            past_key_values = DynamicCache(ddp_cache_data=saved_kv)
            return dict(
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Compute if not in memory
        assert self.prefix_cache is not None
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
        past_key_values = DynamicCache(ddp_cache_data=past_key_values)
        return dict(
            past_key_values=past_key_values,
            use_cache=True,
        )


# ======================= Model logic =======================


class HuggingFaceBackendModel:
    """Implementation of common methods for HuggingFace models.

    Should be used as a mixin for specific HuggingFace model wrappers (e.g., ClassifierHFModel, CLIPEncoderHFModel).

    Implementation Notes:
    - This class wraps the __init__ method of subclasses to handle common bookkeeping tasks, done before and after initialization.
    These include storing common fields (e.g., batch sizes), optionally setting eval mode and freezing weights, registering special tokens, etc.
    - Subclasses are expected to initialize the model object in `self._model` as the runnable nn.Module, the tokenizer in `self._tokenizer`, and the _embedding_layer property (used for token embedding) in their __init__ method.
    - ``self._hf_model`` is the PreTrainedModel object of the model and can be overridden by subclasses that wrap
      the inner HF model. While it
      defaults to ``self._model`` (in most models these are PreTrainedModel objects), some models (e.g., EncoderHFModel) use other backends that wrap it (e.g., SententeTransformer), so for them it's neceessary to separate between the two fields.
    - We currently only support models that allow using inputs_embeds in their forward pass (instead
    of input_ids), which is the most common case. This is conveniant for grad computation and some optimziers
    (e.g., SoftPrompt, GBDA) require such access.
    - In the rare models where there is not such access we currenly need to be a bit hacky; see
    CLIPEncoderHFModel. In the future we might consider to a separate backend for these; currently,
    the demand it is not high enough to justify the engineering effort.
    """

    _model: torch.nn.Module
    _embedding_layer: torch.nn.Module

    @property
    def _hf_model(self) -> transformers.PreTrainedModel:
        """The inner HF ``PreTrainedModel`` used for config/dtype introspection,
        the ``inputs_embeds`` probe, and FLOP counting.

        Expected to be the core module of the model that inherits from ``PreTrainedModel``.
        In some cases it might not contain the full model (e.g., in EncoderHFModel).

        Defaults to ``self._model``. Subclasses whose ``_model`` is a wrapper
        around an HF model (e.g., :class:`EncoderHFModel` holding a
        ``SentenceTransformer``) should override this to return the inner
        ``PreTrainedModel``.
        """
        assert isinstance(self._model, transformers.PreTrainedModel), (
            f"Default `_hf_model` expects `self._model` to be a "
            f"transformers.PreTrainedModel, got {type(self._model).__name__}. "
            f"Override `_hf_model` in the subclass to expose the inner HF model."
        )
        return self._model

    # -- fields extracted from subclass __init__ signatures --
    _PRE_INIT_FIELDS = ("model_name", "forward_pass_batch_size", "backward_pass_batch_size")

    def __init_subclass__(cls, **kwargs):
        """
         Subclass ``__init__`` methods are automatically wrapped (via
    ``__init_subclass__``) to handle common bookkeeping:

        **Before** the subclass ``__init__`` body runs:
        - ``_model_name``, ``_forward_pass_batch_size``, and
            ``_backward_pass_batch_size`` are stored from the matching
            ``__init__`` parameters (looked up by name).

        **After** the subclass ``__init__`` body runs (expects ``_model`` and
        ``_tokenizer`` to be set by then):
        - Model set to eval mode and parameters frozen, unless ``set_model_to_train``.
        - ``OPTIMIZED_TRIGGER_PLACEHOLDER`` registered as a special token.
        - Precision warning emitted when model dtype is float32/float64.
        """
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            return
        original_init = cls.__dict__["__init__"]
        _sig = inspect.signature(original_init)

        @functools.wraps(original_init)
        def _wrapped_init(self, *args, **kwargs):
            ba = _sig.bind(self, *args, **kwargs)
            ba.apply_defaults()

            # --- pre-init: store common fields ---
            for field in HuggingFaceBackendModel._PRE_INIT_FIELDS:
                if field in ba.arguments:
                    setattr(self, f"_{field}", ba.arguments[field])

            # --- run the subclass __init__ ---
            original_init(self, *args, **kwargs)

            # --- post-init: eval, freeze, placeholder, warnings ---
            # Default (set_model_to_train=False): eval mode + frozen weights, the
            # common attack setting. Opt into train mode (trainable weights, no
            # eval) only when an objective differentiates the weights themselves.
            set_model_to_train = ba.arguments.get("set_model_to_train", False)

            if not set_model_to_train:
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False
            else:
                for param in self._model.parameters():
                    param.requires_grad = True

            if OPTIMIZED_TRIGGER_PLACEHOLDER not in self._tokenizer.get_vocab():
                self._tokenizer.add_special_tokens(
                    {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
                )

            if self.dtype in (torch.float32, torch.float64):
                logger.warning(
                    f"Model is in {self.dtype}. Use a lower precision data type, "
                    "if possible, for much faster optimization."
                )

            if self.device == torch.device("cpu"):
                logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

            ## additional check that the model have input_embeds for forward pass
            @torch.no_grad()
            def _requires_input_embeds() -> bool:
                """Returns True if the model requires input_ids even when inputs_embeds are provided."""
                dummy_len = 4
                dummy_embeds = torch.zeros(
                    1, dummy_len, self.embedding_matrix.shape[1],
                    device=self.device, dtype=self.dtype,
                )
                dummy_mask = torch.ones(1, dummy_len, device=self.device, dtype=torch.int64)
                try:
                    self._hf_model(inputs_embeds=dummy_embeds, attention_mask=dummy_mask)
                    return False
                except Exception:
                    return True

            if is_debug_mode() and not getattr(self, "_handles_input_embeds_manually", False) and _requires_input_embeds():
                logger.warning(
                    f"Model `{self._model_name}` seems to not support `inputs_embeds` as forward pass input."
                    "Gradient-based optimization (GradientTokenAccessMixin / invoke_from_tokens) will not work with this model. Only text-level access (invoke_from_texts) is supported."
                )

            logger.info(
                f"Loaded model `{self._model_name}` (device={self.device}, dtype={self.dtype})."
            )

        # Replace the subclass __init__ with the wrapped version
        setattr(cls, "__init__", _wrapped_init)

    @property
    def n_layers(self) -> int:
        """Number of hidden layers in the model.

        ``get_text_config()`` returns the config itself for text-only models and
        the nested text config for multimodal ones (e.g. Gemma-3).
        """
        config = self._hf_model.config
        n = getattr(config.get_text_config(), "num_hidden_layers", None)
        if n is None:
            raise ValueError(
                f"Could not extract `num_hidden_layers` from model config of `{self.get_model_name()}`. "
                f"This model might need special care. Please report this issue."
            )
        return n

    @property
    def dtype(self):
        return next(self._model.parameters()).dtype

    @cached_property
    def tokenizer(self) -> HFTokenizerWrapper:
        return HFTokenizerWrapper(self._tokenizer)

    @property
    def device(self):
        return next(self._model.parameters()).device

    @property
    def embedding_layer(self) -> torch.nn.Module:
        """The input embedding layer of the model (torch module), used for embedding token ids into vectors."""
        return self._embedding_layer

    @property
    def vocab_size(self) -> int:
        # The effective vocab size, as determined by the emb matrix
        vocab_size = self._embedding_layer.num_embeddings
        assert isinstance(vocab_size, int)
        return vocab_size

    @cached_property
    def embedding_matrix(self) -> Float[Tensor, "vocab_size embd_dim"]:
        """
        Compuates the effective embedding matrix used by the model.

        Explanation:
            Sometimes the input embedding function is not a simple one-hot-matmul embedding layer, but rather it's
            enriched with some additional logic (e.g. scaling the embeddings).
            See Gemma3 for example: https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/gemma3/modular_gemma3.py#L348
            Since our (gradient) calculation operates on the matrix *directly*, we need to take this into account.
            In this not-exactly-matmul case, merely multiplying the one-hot encoding with the matrix may provide
            an incorrect embedding. One possible fix is to require the "effective" embedding matrix, and
            operate on it instead, as we do here.
            Naturally, this assumes that the embedding function works position-wise, which is usually the case.

        Returns:
            Tensor of shape (vocab_size, embd_dim)
        """
        assert isinstance(self._embedding_layer, torch.nn.Embedding), "We currently only support models with a standard, yet potentially subclassed, embedding layer."
        all_token_ids = torch.arange(self._embedding_layer.num_embeddings, device=self.device)
        effective_embedding_matrix = self._embedding_layer(all_token_ids)  # (vocab_size, dim)
        return effective_embedding_matrix  # shape: (vocab_size, embd_dim)

    def compute_grad_from_tokens(
        self,
        loss_func: BaseLoss,

        # Hard trigger input:
        candidate_trigger_ids: Optional[Int[Tensor, "n_candidates trigger_seq_len"]] = None,

        # Semi-Soft trigger input:
        candidate_trigger_probs: Optional[Float[Tensor, "n_candidates trigger_seq_len vocab_size"]] = None,
        do_gumbel_softmax: bool = False,
        gumbel_softmax_temp: Optional[float] = None,

        # Additional config:
        normalize_grads: bool = False,
        keep_message_dim: bool = False,
        return_loss: bool = False,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len vocab_size"] | Tuple[Tensor, Tensor]:
        """Compute gradients of loss w.r.t. one-hot token representations for gradient-based optimization.

        This method is the core of white-box, gradient-based trigger optimization (e.g., GCG, GASLITE).
        It computes the gradient of the loss with respect to the one-hot token matrix for each candidate
        trigger, enabling gradient-guided token selection.

        This method supports two possible flows:
            (i) the "hard" trigger flow, starting from discrete token ids (attacks like GCG),
            (ii) the ~"continuous" trigger flow, starting from probability distributions over the vocabulary (attacks like GBDA).

        Args:
            loss_func: Loss function to optimize. Must be compatible with model outputs
                (e.g., PrefillCELoss for LMs, SimilarityLoss for encoders).

            candidate_trigger_ids: Discrete token IDs for hard triggers. Can accepts multiple candidates.
                Shape: (n_candidates, trigger_seq_len)
                Mutually exclusive with `candidate_trigger_probs`.

            candidate_trigger_probs: Probability distributions over vocabulary for continuous triggers.
                Shape: (n_candidates, trigger_seq_len, vocab_size)
                - Mutually exclusive with `candidate_trigger_ids`.
                - Used for continuous optimization methods.

            do_gumbel_softmax: If True, apply Gumbel-softmax to `candidate_trigger_probs`
                before embedding. That is, the provided `candidate_trigger_probs` are treated as logits, and Gumbel-softmax is applied to *draw* a `n_candidades` samples, each respective to its logits.
                - A common pattern here is for `candidate_trigger_probs` to be a repeated tensor of the
                  *same* logits, so we can draw here multiple samples from the same distribution--which is often the one being optimized.
                - Requires `gumbel_softmax_temp` to be set.

            gumbel_softmax_temp: Temperature for Gumbel-softmax sampling.
                Lower values -> more discrete (sharper); higher values -> more uniform.
                Only used when `do_gumbel_softmax=True`.

            normalize_grads: If True, L2-normalize the gradients along the vocab dimension.

            keep_message_dim: If True, return per-template gradients/losses without averaging
                across templates (shape gains a leading n_templates dim). Defaults to False
                (mean-reduced over templates).

            return_loss: If True, also return the computed detached loss values for each candidate. Useful for debugging.

        Returns:
            Gradients w.r.t. one-hot token matrix.
            Shape: (n_candidates, trigger_seq_len, vocab_size), or
            (n_templates, n_candidates, trigger_seq_len, vocab_size) if keep_message_dim=True.

            Gradients are L2-normalized along the vocab dimension (dim=-1) to enable
            fair comparison across different token positions.

        Raises:
            ValueError: If loss_func.is_differentiable is False (e.g. ExternalTriggerPerplexityLoss,
                TextBasedLoss). Use a black-box optimizer instead.
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
        if not loss_func.is_differentiable:
            raise ValueError(
                f"{type(loss_func).__name__}.is_differentiable=False: this loss has no gradient path "
                f"through the model's input embeddings. Grad computation is not possible."
            )
        assert (candidate_trigger_ids is not None) ^ (candidate_trigger_probs is not None), \
            "Exactly one of `candidate_trigger_ids` or `candidate_trigger_probs` must be provided."

        embedding_layer = self._embedding_layer
        assert isinstance(embedding_layer, torch.nn.Embedding), (
            "Expected a standard nn.Embedding layer for one-hot gradient computation."
        )
        vocab_size = embedding_layer.num_embeddings

        # The differentiated leaf is the one-hot / probability matrix; the trigger
        # embeddings are derived from it via the effective embedding matrix.
        if candidate_trigger_probs is None:
            leaves = torch.nn.functional.one_hot(
                candidate_trigger_ids, num_classes=vocab_size
            ).to(self.device, self.dtype)
        else:
            leaves = candidate_trigger_probs.to(self.device, self.dtype)

        def _leaf_to_embeds(leaf_batch, batch_slice):
            if do_gumbel_softmax:
                assert gumbel_softmax_temp is not None, "gumbel_softmax_temp must be provided if do_gumbel_softmax is True."
                leaf_batch = torch.nn.functional.gumbel_softmax(
                    logits=leaf_batch, tau=gumbel_softmax_temp, hard=False, dim=-1,
                )
            # (bsz, trigger_seq_len, vocab_size) @ (vocab_size, embed_dim)
            candidate_embeds = leaf_batch @ self.embedding_matrix

            # Only check when using discrete tokens (not soft probabilities)
            if is_debug_mode() and candidate_trigger_ids is not None:
                assert torch.allclose(
                    candidate_embeds,
                    self._token_input_manager.embed_func(candidate_trigger_ids[batch_slice])
                ), ("Mismatch between effective embedding matrix and embed-func. It could be that you use " \
                "a model with non-standard embedding logic. Please report this issue!")

            # Trigger ids for reference; for soft triggers take the argmax
            # of the *pre*-Gumbel distribution.
            ref_trigger_ids = (
                candidate_trigger_ids[batch_slice] if candidate_trigger_ids is not None
                else leaves[batch_slice].argmax(dim=-1)
            )
            return candidate_embeds, ref_trigger_ids

        return self._grad_wrt_leaves(
            loss_func, leaves, _leaf_to_embeds,
            normalize_grads=normalize_grads,
            keep_message_dim=keep_message_dim,
            return_loss=return_loss,
        )

    def _grad_wrt_leaves(
        self,
        loss_func: BaseLoss,
        leaves: Float[Tensor, "n_candidates trigger_seq_len leaf_dim"],
        leaf_to_embeds,
        normalize_grads: bool,
        keep_message_dim: bool,
        return_loss: bool,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len leaf_dim"] | Tuple[Tensor, Tensor]:
        """Shared backward pass for :meth:`compute_grad_from_tokens` / :meth:`compute_grad_from_embeds`.

        Differentiates ``loss_func`` w.r.t. ``leaves``, batching candidates (with
        OOM backoff) and back-propagating each template separately so only one
        autograd graph is alive at a time.

        Args:
            leaves: The tensor to differentiate w.r.t. — one-hot/probabilities for
                the token flow, raw trigger embeddings for the embedding flow.
            leaf_to_embeds: ``(leaf_batch, batch_slice) -> (trigger_embeds, ref_trigger_ids)``.
        """
        assert self._token_input_manager is not None, "Token input manager is not initialized. Please call set_inputs_from_tokens() first."

        device, dtype = self.device, self.dtype
        input_manager = self._token_input_manager
        n_templates = input_manager.n_templates
        n_candidates, trigger_seq_len, leaf_dim = leaves.shape

        @find_executable_batch_size(starting_batch_size=self._backward_pass_batch_size)
        def _compute_grad__batched(
            batch_size: int,
        ) -> Tuple[Tensor, Tensor]:

            # --- Update backward batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self._backward_pass_batch_size:
                logger.info(f"OOM detected. Reducing _backward_pass_batch_size from {self._backward_pass_batch_size} to {batch_size}")
                self._backward_pass_batch_size = batch_size
            # --------------------

            all_grads = []  # of len `n_candidate // batch_size`
            all_losses = []  # per-batch mean losses (detached)

            for cand_idx_start in range(0, n_candidates, batch_size):
                cand_idx_end = min(cand_idx_start + batch_size, n_candidates)
                cand_bsz = cand_idx_end - cand_idx_start
                batch_slice = slice(cand_idx_start, cand_idx_end)

                # Backward each template immediately to avoid keeping n_templates graphs at once
                accum_grad = torch.zeros(
                    (n_templates, cand_bsz, trigger_seq_len, leaf_dim),
                    device=device, dtype=dtype,
                )
                accum_loss = torch.zeros((n_templates, cand_bsz), device=device, dtype=dtype)

                for template_idx in range(0, n_templates):
                    # 1. Enable gradients on the leaf input
                    leaf = leaves[batch_slice].clone()
                    leaf.requires_grad_()

                    # 2. Map the leaf to trigger embeddings
                    candidate_embeds, ref_trigger_ids = leaf_to_embeds(leaf, batch_slice)

                    # 3. Get batched inputs & compute loss:
                    logger.debug(f"from grad [msg={template_idx}]: {candidate_embeds.shape}")

                    model_input = input_manager.get_triggered_inputs(
                        chosen_template_idx=template_idx,
                        trigger_embeds=candidate_embeds,
                        trigger_ids=ref_trigger_ids,  # Also pass trigger ids as a reference

                        # loss-conditional flags:
                        do_append_embeds=loss_func.require_target_prefill,
                    )
                    model_output = self.invoke_from_tokens(
                        **model_input.to_dict(),

                        # loss-conditional flags:
                        require_target_prefill=loss_func.require_target_prefill,
                        require_generation=loss_func.require_generation,
                        require_hidden_states=loss_func.require_hidden_states,
                        require_attentions=loss_func.require_attentions,
                        count_backward=True,
                    )
                    loss = resolve_and_compute_loss(model_output, model_input, loss_func)

                    # 4. Backward and store per-template
                    template_grad = torch.autograd.grad(
                        outputs=loss,
                        inputs=[leaf],
                        grad_outputs=torch.ones_like(loss, device=device),
                    )[0]  # (bsz_triggers, trigger_seq_len, leaf_dim)
                    accum_grad[template_idx] = template_grad
                    accum_loss[template_idx] = loss.detach()

                all_grads.append(accum_grad)
                all_losses.append(accum_loss)

            return (
                torch.cat(all_grads, dim=1),  # (n_templates, n_candidates, trigger_seq_len, leaf_dim)
                torch.cat(all_losses, dim=1),  # (n_templates, n_candidates)
            )

        # Per-template grads/losses of the candidates
        all_grads, all_losses = _compute_grad__batched()

        # Reduce message dim unless caller requested per-template outputs
        if not keep_message_dim:
            all_grads = all_grads.mean(dim=0)
            all_losses = all_losses.mean(dim=0)

        # normalize each token's gradient vector (over the last dim)
        if normalize_grads:
            all_grads = all_grads / (all_grads.norm(dim=-1, keepdim=True) + 1e-10)

        if return_loss:
            return all_grads, all_losses

        return all_grads

    def compute_grad_from_embeds(
        self,
        loss_func: BaseLoss,
        candidate_trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embed_dim"],
        normalize_grads: bool = False,
        keep_message_dim: bool = False,
        return_loss: bool = False,
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len embed_dim"] | Tuple[Tensor, Tensor]:
        """Compute gradients of loss w.r.t. trigger embeddings.

        This variant optimizes directly in the continuous embedding space, with no constraints.

        Args:
            keep_message_dim: If True, return per-template gradients/losses without averaging
                across templates (shape gains a leading n_templates dim). Defaults to False
                (mean-reduced over templates).

        Returns:
            If return_loss is False: gradients tensor (n_candidates, trigger_seq_len, embed_dim),
            or (n_templates, n_candidates, trigger_seq_len, embed_dim) if keep_message_dim=True.
            If return_loss is True: tuple of (gradients tensor, per-candidate loss tensor (n_candidates,)),
            or per-template losses (n_templates, n_candidates) if keep_message_dim=True.
        """
        # Here the differentiated leaf *is* the trigger embedding, so no mapping
        # is needed and there are no reference trigger ids to report.
        return self._grad_wrt_leaves(
            loss_func,
            candidate_trigger_embeds.detach(),
            lambda leaf, batch_slice: (leaf, None),
            normalize_grads=normalize_grads,
            keep_message_dim=keep_message_dim,
            return_loss=return_loss,
        )

    @torch.no_grad()
    def compute_loss_from_tokens(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"] | Float[Tensor, "n_templates n_candidates"]:
        """Computes the loss on all candidate token id sequences. Runs under
        ``torch.no_grad`` unless the loss sets ``require_gradients`` (e.g. gradient
        matching, whose value is itself a weight-gradient).

        Args:
            candidate_trigger_ids : Tensor, shape = (n_candidates, trigger_seq_len)
                candidate trigger token ids to evaluate
            loss_func : BaseLoss
                the loss to compute for each candidate
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
                    chosen_template_idx=template_idx,
                    trigger_ids=batch_candidate_trigger_ids,

                    # loss-conditional flags:
                    do_append_embeds=loss_func.require_target_prefill,
                )

                # Only enable gradient is it's required by the loss (e.g. for gradient matching losses); mostly false.
                with torch.set_grad_enabled(loss_func.require_gradients):
                    model_output = self.invoke_from_tokens(
                        **model_input.to_dict(),

                        # loss-conditional flags:
                        require_target_prefill=loss_func.require_target_prefill,
                        require_generation=loss_func.require_generation,
                        require_hidden_states=loss_func.require_hidden_states,
                        require_attentions=loss_func.require_attentions,
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
        input_attention_mask: Optional[Int[Tensor, "bsz seq_len"]] = None,

        require_target_prefill: bool = False,
        require_generation: bool = False,
        require_hidden_states: bool = False,
        require_attentions: bool = False,
        count_backward: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Performs a forward pass with the given token-based model input. Forward pass is expected to be done on `input_embeds`.

        Args:
            input_embeds: Float[Tensor, "bsz seq_len d_model"]
                the input embeddings with the trigger merged in; if provided, used instead of any other potential input.
            input_attention_mask: Optional[Int[Tensor, "bsz seq_len"]] = None
                the attention mask matching the input embeddings
            require_target_prefill: bool
                whether to prefill the target response, and return the corresponding logits (e.g., for LMs).
            require_generation: bool
                whether to perform autoregressive generation after the forward pass (for LMs).
            require_hidden_states: bool
                whether to return the hidden states from the model output.
            require_attentions: bool
                whether to return the attention weights from the model output.
            count_backward: bool
                whether this forward pass will be back-propagated through (set by gradient methods). Could be used by FLOP counters.

        Returns:
            ModelOutput
                the model output containing logits, embeddings, attentions, etc.
        """
        pass

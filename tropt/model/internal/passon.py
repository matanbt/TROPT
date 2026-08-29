"""Model class that acts as a pass-through "model" -- no model behind it. Useful when optimizing against self-contained trigger losses that expect no model to be queried.


"""

from typing import List, Optional, Union

from transformers import AutoTokenizer

from tropt.common import ModelOutput, Targets, TextTemplates
from tropt.model.inputs_manager import DefaultTokenInputManager
from tropt.model.model_base import BaseModel, BaseTokenizer, HFTokenizerWrapper
from tropt.model.model_mixins import LossTextAccessMixin, TokenAccessMixin


class PassOnModel(
    BaseModel,
    LossTextAccessMixin,
    TokenAccessMixin,  # tokenizer access, but not loss access on it
):
    """Passes candidate triggers straight to the loss; holds no model API/weights.

    This model class have no undelying model, and implement model invocation methods as no-ops. It also exposes an auxiliary tokenizer.

    - Any call for loss compuation acts as a no-op, and simply forwards the existing model inputs to the loss.
    - The model exposes an auxiliary tokenizer to support token-level optimziers, though it does not provide any token-level loss access (similarly to `EncoderOpenAIModel`).
    - Pair with a loss reading model input (e.g., ``input_trigger_strs`` / ``input_texts``), and any optimizer requiring ``LossTextAccessMixin`` (e.g. ``RandomSearchOptimizer``).


    *Motivation:*
        Some losses are self-contained oracles: they score the trigger text alone
        (e.g. ``ExternalTriggerPerplexityLoss``, or losses against compliated APIs such as coding agents).
        These losses don't expect any model to be queried or deliver arguments to them; it would therefore be wasteful to have a model component that queries a model and then discards the result. 
        This is precisely what this `PassOnModel` is for: it has no underlying model, and simply forwards candidate triggers to the loss.

    *Use for:*
        `PassOnModel` is useful for cases where we have a loss that queries an external API using the triggers, or a loss that computes some complicated self-contained metric on top of the triggers. In such cases, we can still use TROPT's optimziers (despite querying not actual model), by using this class, and climb-hill the given metric.
    """

    def __init__(self, tokenizer: Union[str, BaseTokenizer] = "google/gemma-3-270m-it"):
        """
        Args:
            tokenizer: Auxiliary tokenizer defining the optimizer's search space,
                as a HuggingFace name or a ``BaseTokenizer`` (e.g. ``OpenAITokenizer``
                when targeting OpenAI models). Only the tokenizer is downloaded.
        """
        self._tokenizer = (
            HFTokenizerWrapper(AutoTokenizer.from_pretrained(tokenizer))
            if isinstance(tokenizer, str) else tokenizer
        )
        self.model_name = self._tokenizer.name_or_path

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepares and stores the token-level inputs manager."""
        assert isinstance(templates, list), "templates must be a list of strings."

        tok_ids = self._tokenizer(
            templates, return_tensors="list", add_special_tokens=False
        )["input_ids"]

        self._token_input_manager = DefaultTokenInputManager(
            tokenizer=self._tokenizer,
            templates_ids=tok_ids,
            targets=targets,
        )

    def __call__(self, input_texts: List[str], **kwargs) -> List[str]:
        return input_texts

    def invoke_from_texts(self, input_texts: List[str], **kwargs) -> ModelOutput:
        """No-op model invocation. Returns an empty ModelOutput; still counts the tokens in the input
        texts for budget tracking.
        """
        n_tokens = sum(len(ids) for ids in self._tokenizer(input_texts)["input_ids"])
        self._update_invoke_stats(n_tokens=n_tokens, n_samples=len(input_texts))
        return ModelOutput()

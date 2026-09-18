from typing import Annotated, Any, List, Optional

from jaxtyping import Int
from torch import Tensor

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelInput,
    Targets,
    TextTemplates,
)

# ======================= Triggered Input Managers =======================
#
# An inputs manager wraps `n_templates` templates (each containing the substring
# OPTIMIZED_TRIGGER_PLACEHOLDER) plus their targets, and exposes
# `get_triggered_inputs(chosen_template_idx, <trigger candidates>) -> ModelInput`.
# The trigger-candidate argument differs per flow (strings vs ids vs embeds), so
# the text and token families deliberately share no base class.


## Text inputs manager ##
class TextInputManager:
    """
    Class for maintaining text-based trigger-combined inputs (fits black-box text-level query access).
    Instances of this class store `n_templates` templates and targets, and provide the method `get_triggered_inputs` to combine them with given trigger strings.
    """

    before_texts: Annotated[List[str], "n_templates"]
    after_texts: Annotated[List[str], "n_templates"]
    targets: Targets

    def __init__(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ):
        assert isinstance(templates, list), "templates must be a list of strings."
        if targets is None:
            targets = Targets()

        before_texts, after_texts = [], []
        for i, template in enumerate(templates):
            n_found = template.count(OPTIMIZED_TRIGGER_PLACEHOLDER)
            assert n_found == 1, (
                f"`templates[{i}]` must contain exactly one `{OPTIMIZED_TRIGGER_PLACEHOLDER}` "
                f"placeholder, found {n_found}. Note the placeholder is case-sensitive and takes "
                f"no inner spaces. Got: {template!r}"
            )
            bef, aft = template.split(OPTIMIZED_TRIGGER_PLACEHOLDER, 1)
            before_texts.append(bef)
            after_texts.append(aft)

        self.before_texts = before_texts
        self.after_texts = after_texts
        self.targets = targets

    @property
    def n_templates(self) -> int:
        return len(self.before_texts)

    def get_triggered_inputs(
        self,
        chosen_template_idx: int,
        trigger_strs: Annotated[List[str], "n_candidates"],
    ) -> ModelInput:
        """
        Returns a list of inputs with the given trigger strings merged in.
        The list is two-dimensional: outer list over templates, inner list over trigger variations; also, returns the corresponding targets.

        Given `chosen_template_idx`, returns only the inputs for that template (1D list), and the corresponding targets.
        """
        assert isinstance(trigger_strs, list) and all(
            isinstance(s, str) for s in trigger_strs
        ), "trigger_strs must be a list of strings."
        n_candidates: int = len(trigger_strs)  # noqa 

        input_texts: List[str] = []
        for trigger_str in trigger_strs:
            curr_text = (
                self.before_texts[chosen_template_idx]
                + trigger_str
                + self.after_texts[chosen_template_idx]
            )
            input_texts.append(curr_text)

        # select only the chosen template's targets
        message_targets = self.targets.select_message(chosen_template_idx)

        return ModelInput(
            input_texts=input_texts,
            input_trigger_strs=trigger_strs,
            message_targets=message_targets
        )

## Token inputs manager ##
class TokenInputManager:
    """
    Base class for token-level inputs managers (the shared type for
    ``TokenAccessMixin._token_input_manager``).

    Subclasses manage the combination of candidate triggers into tokenized
    templates.
    """

    tokenizer: Any
    targets: Targets
    n_templates: int


class DefaultTokenInputManager(TokenInputManager):
    """
    Default token-level inputs manager for models with token-level access.

    This implementation works with any tokenizer supporting the BaseTokenizer
    interface (or HuggingFace PreTrainedTokenizer). It decodes trigger token IDs
    to strings and reconstructs full texts — suitable for API-based models or
    any model where embedding-level manipulation is not needed.
    """

    def __init__(
        self,
        tokenizer: Any,
        templates_ids: List[List[int]],
        targets: Optional[Targets] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer

        if targets is None:
            targets = Targets()
        self.targets = targets

        # Decode the input tokens back to text and split by placeholder
        raw_texts = tokenizer.batch_decode(templates_ids)
        self.before_texts = []
        self.after_texts = []

        for text in raw_texts:
            assert text.count(OPTIMIZED_TRIGGER_PLACEHOLDER) == 1, (
                f"Text must contain exactly one placeholder '{OPTIMIZED_TRIGGER_PLACEHOLDER}'"
            )
            bef, aft = text.split(OPTIMIZED_TRIGGER_PLACEHOLDER, 1)
            self.before_texts.append(bef)
            self.after_texts.append(aft)

        self.n_templates = len(raw_texts)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_triggered_inputs(
        self,
        chosen_template_idx: int,
        trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        **kwargs
    ) -> ModelInput:
        """
        Constructs full text inputs by decoding candidate trigger tokens
        and inserting them into the templates.
        """
        trigger_strs = self.tokenizer.batch_decode(trigger_ids, skip_special_tokens=True)

        bef = self.before_texts[chosen_template_idx]
        aft = self.after_texts[chosen_template_idx]

        curr_message_candidates = [
            f"{bef}{trig}{aft}" for trig in trigger_strs
        ]

        targets = self.targets.select_message(chosen_template_idx)

        return ModelInput(
            input_trigger_ids=trigger_ids,
            input_trigger_strs=trigger_strs,
            input_texts=curr_message_candidates,
            message_targets=targets,
        )

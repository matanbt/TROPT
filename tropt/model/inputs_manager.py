from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelInput,
    SliceKey,
    Targets,
    TextTemplates,
)

# ======================= Triggered Input Managers =======================


class InputsManager(ABC):
    """
    Base class for maintaining the input template, corresponding targets, and the method for injecting triggers into the inputs.
    This class wraps `n_templates` templates and targets, and provides a unified interface for different types of inputs (e.g., text-based, token-based) used in adversarial trigger optimization.
    """

    optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER

    def __init__(
        self,
        templates: TextTemplates,
        targets: Targets,  # n_templates elements per target entry
    ):
        raise NotImplementedError

    @abstractmethod
    def get_triggered_inputs(self, *args, **kwargs) -> ModelInput:
        raise NotImplementedError

## Text inputs manager ##
class TextInputManager(InputsManager):
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
        targets: Targets = None,
        optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
    ):
        assert isinstance(templates, list), "templates must be a list of strings."
        if targets is None:
            targets = Targets()

        targets = targets.to_device("cuda" if torch.cuda.is_available() else "cpu")

        before_texts, after_texts = [], []
        for template in templates:
            bef, aft = template.split(optimized_trigger_placeholder)
            before_texts.append(bef)
            after_texts.append(aft)

        self.before_texts = before_texts
        self.after_texts = after_texts
        self.targets = targets

    @property
    def n_templates(self):
        return len(self.before_texts)

    def get_triggered_inputs(
        self,
        trigger_strs: Annotated[List[str], "n_candidates"],
        chosen_template_idx: Optional[int],
    ) -> ModelInput:
        """
        Returns a list of inputs with the given trigger strings merged in.
        The list is two-dimensional: outer list over templates, inner list over trigger variations; also, returns the corresponding targets.

        Given `chosen_template_idx`, returns only the inputs for that template (1D list), and the corresponding targets.
        """
        assert isinstance(trigger_strs, list) and all(
            isinstance(s, str) for s in trigger_strs
        ), "trigger_strs must be a list of strings."
        n_candidates = len(trigger_strs)

        input_texts: List[str] = []
        for trigger_str in trigger_strs:
            curr_text = (
                self.before_texts[chosen_template_idx]
                + trigger_str
                + self.after_texts[chosen_template_idx]
            )
            input_texts.append(curr_text)

        # select only the chosen template's targets
        targets = self.targets.select_message(chosen_template_idx)

        return ModelInput(
            input_texts=input_texts,
            input_trigger_strs=trigger_strs,
            targets=targets
        )

## Token inputs manager ##
class TokenInputManager(InputsManager):
    """
    Base class for maintaining token-level trigger-combined inputs (fits models with token-level access).
    """

    before_ids: Annotated[List[Float[Tensor, "bef_len"]], "n_templates"]
    after_ids: Annotated[List[Float[Tensor, "aft_len"]], "n_templates"]
    targets: Targets
    tokenizer: Any

    # Properties:
    vocab_size: int
    n_templates: int

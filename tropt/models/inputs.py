from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelInput,
    SliceKey,
    Targets,
)

# ======================= Triggered Input Managers =======================


class InputsManager(ABC):
    """
    Base class for maintaining the input template, corresponding targets, and the method for injecting triggers into the inputs.
    This class wraps `n_messages` texts and targets, and provides a unified interface for different types of inputs (e.g., text-based, token-based) used in adversarial trigger optimization.
    """

    optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER

    def __init__(
        self,
        text_templates: List[str],  # n_messages texts
        targets: Targets,  # n_messages elements per target entry
    ):
        raise NotImplementedError

    @abstractmethod
    def get_triggered_inputs(self, *args, **kwargs) -> ModelInput:
        raise NotImplementedError

## Text inputs manager ##
class TextInputsManager(InputsManager):
    """
    Class for maintaining text-based trigger-combined inputs (fits black-box text-level query access).
    """

    before_texts: Annotated[List[str], "n_messages"]
    after_texts: Annotated[List[str], "n_messages"]
    targets: Targets

    def __init__(
        self,
        texts: Annotated[List[str], "n_messages"],
        targets: Targets = None,
        optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
    ):
        assert isinstance(texts, list), "texts must be a string or a list of strings."
        if targets is None:
            targets = Targets()

        targets = targets.to_device("cuda" if torch.cuda.is_available() else "cpu")

        before_texts, after_texts = [], []
        for text in texts:
            bef, aft = text.split(optimized_trigger_placeholder)
            before_texts.append(bef)
            after_texts.append(aft)

        self.before_texts = before_texts
        self.after_texts = after_texts
        self.targets = targets

    @property
    def n_messages(self):
        return len(self.before_texts)

    def get_triggered_inputs(
        self,
        trigger_strs: Annotated[List[str], "n_candidates"],
        chosen_message_idx: Optional[int],
    ) -> ModelInput:
        """
        Returns a list of inputs with the given trigger strings merged in.
        The list is two-dimensional: outer list over messages, inner list over trigger variations; also, returns the corresponding targets.

        Given `chosen_message_idx`, returns only the inputs for that message (1D list), and the corresponding targets.
        """
        assert isinstance(trigger_strs, list) and all(
            isinstance(s, str) for s in trigger_strs
        ), "trigger_strs must be a list of strings."
        n_candidates = len(trigger_strs)

        input_texts: List[str] = []
        for trigger_str in trigger_strs:
            curr_text = (
                self.before_texts[chosen_message_idx]
                + trigger_str
                + self.after_texts[chosen_message_idx]
            )
            input_texts.append(curr_text)

        # select only the chosen message's targets
        targets = self.targets.select_message(chosen_message_idx)

        return ModelInput(
            input_texts=input_texts,
            input_trigger_strs=trigger_strs,
            targets=targets
        )

## Token inputs manager ##
class TokenInputsManager(InputsManager):
    """
    Base class for maintaining token-level trigger-combined inputs (fits models with token-level access).
    """

    before_ids: List[Float[Tensor, "bef_len"]]
    after_ids: List[Float[Tensor, "aft_len"]]  # of length n_messages
    targets: Targets
    tokenizer: Any

    # Properties:
    vocab_size: int
    n_messages: int

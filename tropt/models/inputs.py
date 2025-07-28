from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER

# ======================= Common input types =======================
TokenTrigger = Float[Tensor, "1 trigger_seq_len"]
TokenTriggerCandidates = Float[Tensor, "n_candidates trigger_seq_len"]

# ======================= Target Types and Utils =======================

## A dict of each target; where each message is mapped to its target tensor/string/etc
TargetsDict = Dict[str,
    List[str]   # list of length n_messages
      | Float[Tensor, "n_messages target_seq_len"]
      | Float[Tensor, "n_messages d_model"],
]  # each entry has n_messages elements, each elenents has the target data.

## A dict for each target; where each message is mapped
# to a repeated _batch_ of its target tensor/string/etc
BatchedTargetsDict = Dict[str,
    List[List[str]]   # list of n_messages batches of strings
      | Float[Tensor, "n_messages bsz target_seq_len"]
      | List[Float[Tensor, "bsz target_seq_len"]]
      | Float[Tensor, "n_messages bsz d_model"]
]  # each entry has n_messages, each a batch of identical targets

## A dict for each target; where _a single pre-selected message_ is mapped
# to a batch of its target tensor/string/etc
MessageBatchedTargetsDict = Dict[str,
    List[str]   # of length bsz
      | Float[Tensor, "bsz target_seq_len"]
      | List[Float[Tensor, "target_seq_len"]]  # of length bsz
      | Float[Tensor, "bsz d_model"]
]  # each entry has a batch of targets (for the selected message)


class TargetsDictPlus(dict):
    """
    Class for extending the TargetsDict with useful utilities; this dict maps target keys to
    their corresponding target values (used for different losses) for each input message.
    While for most common logic it's sufficient to use a plain dict (TargetsDict), this class
    provides some useful utils and validations, making it a good practice to use it as the
    targets container.
    """

    def __init__(self, targets: TargetsDict, n_messages: int = None):
        """
        Initializes the TargetsManager with the given targets dictionary.
        Optionally provide `n_messages` to validate the targets.

        Each entry can be a tensor (shape: (n_messages, *)) or a list (e.g., of string, of tensors of varying lengths) of size n_messages.
        """
        if targets is None:
            targets = {}
        super().__init__(targets)

        if n_messages is None:
            # if not provided, infer from the an entry
            n_messages = len(next(iter(self.values())))
        self.n_messages = n_messages

        assert isinstance(self, dict)
        assert all(isinstance(k, str) for k in self.keys())
        assert all(len(val) == n_messages for val in self.values())

    def to_device(self, device: torch.device) -> "TargetsDictPlus":
        """
        Moves all the tensor targets to the specified device; inplace.
        """
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(device)
            elif isinstance(self[k], list) and isinstance(self[k][0], torch.Tensor):
                self[k] = [t.to(device) for t in self[k]]
        return self

    def __setitem__(self, key, value):
        assert len(value) == self.n_messages, f"Length of target entry for key {key} must be {self.n_messages}, but got {len(value)}."
        return super().__setitem__(key, value)

    #----------------------------------------------------------------------------#
    ## Utils for obtaining and manipulating different views of the TargetsDict: ##
    # TODO consider having another class BatchedTargetsDictPlus inheriting from `dict`
    @staticmethod
    def get_expanded_with_candidates(targets: TargetsDict | "TargetsDictPlus", n_candidates: int) -> BatchedTargetsDict:
        """
        Repeats each target entry `n_repeats` times along the message dimension.
        Useful when expanding targets to match multiple candidate triggers per message.
        Returns result in a new dict (BatchedTargetsDict).
        """
        targets = targets.copy()

        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                # (n_messages, ...) -> (n_messages, n_candidates, ...)
                targets[k] = v.unsqueeze(1).expand(v.shape[0], n_candidates, *v.shape[1:])
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                # for each element: (...,) -> (n_candidates, ...)
                targets[k] = [t.unsqueeze(0).expand(n_candidates, *t.shape) for t in v]
            elif isinstance(v, list):
                targets[k] = [[elem] * n_candidates for elem in v]
            else:
                raise ValueError(f"Unsupported target type for key {k}: {type(targets[k])}")

        return targets

    @staticmethod
    def get_candidate_batch_from_batched_targets(
        targets: BatchedTargetsDict,
        cand_batch_slice: slice,
    ) -> BatchedTargetsDict:
        """
        Selects a batch slice from each target entry in the BatchedTargetsDict.
        Returns result in a new dict (BatchedTargetsDict).
        """
        targets = targets.copy()

        for k in targets.keys():
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k][:, cand_batch_slice]
            elif isinstance(targets[k], list) and isinstance(targets[k][0], torch.Tensor):
                targets[k] = [elem[cand_batch_slice] for elem in targets[k]]
            elif isinstance(targets[k], list):
                targets[k] = [elem[cand_batch_slice] for elem in targets[k]]
            else:
                raise ValueError(f"Unsupported target type for key {k}: {type(targets[k])}")

        return targets

    @staticmethod
    def get_message_from_batched_targets(
        targets: BatchedTargetsDict,
        chosen_message_idx: int,
    ) -> MessageBatchedTargetsDict:
        """
        Selects the targets for a specific message index from the BatchedTargetsDict.
        Returns result in a new dict (MessageBatchedTargetsDict).
        """
        targets = {k: v[chosen_message_idx] for k, v in targets.items()}
        return targets

    #----------------------------------------------------------------------------


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
        targets: TargetsDict | TargetsDictPlus,  # n_messages elements per target entry
    ):
        raise NotImplementedError

    @abstractmethod
    def get_triggered_inputs(self, *args, **kwargs):
        raise NotImplementedError


class TextInputsManager(InputsManager):
    """
    Class for maintaining text-based trigger-combined inputs (fits black-box text-level query access).
    """

    before_texts: List[str]
    after_texts: List[str]  # of length n_messages
    targets: TargetsDict | TargetsDictPlus

    def __init__(
        self,
        texts: List[str],  # n_messages texts
        targets: TargetsDict = {},  # n_messages elements per target entry
        optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
    ):
        assert isinstance(texts, list), "texts must be a string or a list of strings."
        n_messages = len(texts)
        targets = TargetsDictPlus(targets, n_messages=n_messages)
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
        trigger_strs: List[str],
        chosen_message_idx: Optional[int] = None,
    ) -> Dict[str, List[str] | BatchedTargetsDict | MessageBatchedTargetsDict]:
        """
        Returns a list of inputs with the given trigger strings merged in.
        The list is two-dimensional: outer list over messages, inner list over trigger variations; also, returns the corresponding targets.

        Given `chosen_message_idx`, returns only the inputs for that message (1D list), and the corresponding targets.
        """
        assert isinstance(trigger_strs, list) and all(
            isinstance(s, str) for s in trigger_strs
        ), "trigger_strs must be a list of strings."
        n_candidates = len(trigger_strs)
        inputs = []

        for message_idx in range(self.n_messages):
            inputs.append([])
            for trigger_str in trigger_strs:
                curr_text = (
                    self.before_texts[message_idx]
                    + trigger_str
                    + self.after_texts[message_idx]
                )
                inputs[-1].append(curr_text)

        # Expand the target entries accordingly
        targets: TargetsDictPlus = self.targets.copy()
        targets: BatchedTargetsDict = TargetsDictPlus.get_expanded_with_candidates(targets, n_candidates)

        # If specified, select only the chosen message's inputs and targets
        if chosen_message_idx is not None:
            inputs = inputs[chosen_message_idx]
            targets: MessageBatchedTargetsDict = TargetsDictPlus.get_message_from_batched_targets(
                targets, chosen_message_idx
            )

        return dict(
            inputs_texts=inputs, targets=targets
        )


class TokenInputsManager(InputsManager):
    """
    Base class for maintaining token-level trigger-combined inputs (fits models with token-level access).
    """

    before_ids: List[Float[Tensor, "bef_len"]]
    after_ids: List[Float[Tensor, "aft_len"]]  # of length n_messages
    targets: TargetsDict | TargetsDictPlus
    tokenizer: Any

    # Properties:
    vocab_size: int
    n_messages: int

    @abstractmethod
    def toks_to_strs(
        self,
        toks: Int[Tensor, "seq_len"],
        **kwargs,
    ) -> str:
        """Converts a 1D token ids tensor to a string using the tokenizer."""
        raise NotImplementedError

from .base import (
    BaseLoss,
    CombinedLoss,
)
from .resolution import (
    LossResolutionError,
    resolve_and_compute_loss,
)
from .losses import (
    AttentionBasedLoss,
    AttentionEnhLoss,
    ClassificationBasedLoss,
    EmbeddingBasedLoss,
    HiddenStateBasedLoss,
    # Concrete losses:
    MisclassCELoss,
    PrefillBasedLoss,
    PrefillCELoss,
    PrefillCWLoss,
    PrefillDistillationLoss,
    PrefillMellowMaxLoss,
    SimilarityLoss,
    SteeringActivationLoss,
    TriggerLogitBasedLoss,
    TriggerPerplexityLoss,
)
from .text_losses import (
    BinaryLMJudgeLoss,
    # Concrete text losses
    ExternalTriggerPerplexityLoss,
    FirstTokenNLLLoss,
    GeneratedResponseBasedLoss,
    InputFluencyLoss,
    ResponseHarmfulnessLoss,
    TextBasedLoss,
)

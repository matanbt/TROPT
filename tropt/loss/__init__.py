from .base import (
    BaseLoss,
    CombinedLoss,
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
    GeneratedResponseBasedLoss,
    InputFluencyLoss,
    ResponseHarmfulnessLoss,
    TextBasedLoss,
)

from .resolution import (
    resolve_and_compute_loss,
    LossResolutionError,
)

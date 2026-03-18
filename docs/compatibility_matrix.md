# Optimizer-Model-Loss Compatibility Matrix

> **Auto-generated** by `scripts/generate_compat_matrix.py` — do not edit manually.

Each cell lists the concrete loss functions supported for the given optimizer-model pair,
or shows **Unsupported** if the model does not satisfy the optimizer's requirements.
Optimizers with both token and text flows appear as two rows, one per flow.

| Optimizer | EncoderGeminiModel | EncoderHFModel | EncoderOpenAIModel | LMHFModel | LiteLLMModel |
|---|---|---|---|---|---|
| **BeamSearchOptimizer** | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss` | `InputReadabilityLoss` |
| **GASLITEOptimizer** | **Unsupported** | `SimilarityLoss` | **Unsupported** | `AttentionEnhLoss`, `PrefillCELoss`, `PrefillCWLoss`, `PrefillMellowMaxLoss`, `SteeringActivationLoss`, `TriggerPerplexityLoss` | **Unsupported** |
| **GASLITEPlusOptimizer** | **Unsupported** | `SimilarityLoss` | **Unsupported** | `AttentionEnhLoss`, `PrefillCELoss`, `PrefillCWLoss`, `PrefillMellowMaxLoss`, `SteeringActivationLoss`, `TriggerPerplexityLoss` | **Unsupported** |
| **GCGOptimizer** | **Unsupported** | `SimilarityLoss` | **Unsupported** | `AttentionEnhLoss`, `PrefillCELoss`, `PrefillCWLoss`, `PrefillMellowMaxLoss`, `SteeringActivationLoss`, `TriggerPerplexityLoss` | **Unsupported** |
| **RASLITEPlusOptimizer** | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss`, `SimilarityLoss` | `InputReadabilityLoss` | `InputReadabilityLoss` |

## Legend

### Access levels required by each optimizer

| Optimizer | Required Mixins | Flows | Access Level |
|---|---|---|---|
| BeamSearchOptimizer | `LossTextAccessMixin` | `text` | Black-box |
| GASLITEOptimizer | `LossTokenAccessMixin`, `GradientTokenAccessMixin` | `token` | White-box |
| GASLITEPlusOptimizer | `LossTokenAccessMixin`, `GradientTokenAccessMixin` | `token` | White-box |
| GCGOptimizer | `LossTokenAccessMixin`, `GradientTokenAccessMixin` | `token` | White-box |
| RASLITEPlusOptimizer | `LossTextAccessMixin` | `text` | Black-box |

### Concrete loss functions

| Loss | Base Type | Required Parameters |
|---|---|---|
| `AttentionEnhLoss` | `AttentionBasedLoss` | `output_attentions`, `input_slices` |
| `InputReadabilityLoss` | `BinaryLMJudgeLoss` | `input_texts` |
| `PrefillCELoss` | `LogitBasedLoss` | `response_logits`, `target_response_toks` |
| `PrefillCWLoss` | `LogitBasedLoss` | `response_logits`, `target_response_toks` |
| `PrefillMellowMaxLoss` | `LogitBasedLoss` | `response_logits`, `target_response_toks` |
| `SimilarityLoss` | `EmbeddingBasedLoss` | `output_embeddings`, `target_vectors` |
| `SteeringActivationLoss` | `HiddenStateBasedLoss` | `output_hidden_states`, `target_directions` |
| `TriggerPerplexityLoss` | `TriggerLogitBasedLoss` | `output_logits`, `input_trigger_ids`, `input_slices` |

### Discovered fields per model (via source AST)

| Model | Flow | ModelOutput fields | ModelInput fields |
|---|---|---|---|
| EncoderGeminiModel | token | *(none)* | *(none)* |
| EncoderGeminiModel | text | `output_embeddings` | `input_texts`, `targets` |
| EncoderHFModel | token | `output_embeddings` | `input_attention_mask`, `input_embeds`, `input_prefix_cache_kwargs`, `input_slices`, `input_trigger_ids`, `targets` |
| EncoderHFModel | text | `output_embeddings` | `input_texts`, `targets` |
| EncoderOpenAIModel | token | *(none)* | *(none)* |
| EncoderOpenAIModel | text | `output_embeddings` | `input_texts`, `targets` |
| LMHFModel | token | `output_attentions`, `output_hidden_states`, `output_logits`, `response_logits` | `input_attention_mask`, `input_embeds`, `input_prefix_cache_kwargs`, `input_slices`, `input_trigger_ids`, `targets` |
| LMHFModel | text | `full_template_ids`, `full_template_strs`, `generated_response_ids`, `generated_response_logits`, `generated_response_strs` | `input_texts`, `targets` |
| LiteLLMModel | token | *(none)* | *(none)* |
| LiteLLMModel | text | `generated_response_strs` | `input_texts`, `targets` |

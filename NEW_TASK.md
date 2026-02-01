# dataclass: ModelInputWrapper
input_texts
input_trigger_strs
input_trigger_ids
input_embeds
input_attention_mask
input_prefix_cache_kwargs
input_slices
targets

# dataclass: ModelOutputWrapper
output_embeddings
output_hidden_states
output_attentions
output_logits  # possibly with appended (=prefilled) target tokens
generated_response_logits
generated_response_ids
generated_response_strs


## Tasks:
1. All the models' __call__ should have a flag return_full_output=False; if this flag is passed as True, they should return a ModelOutput object, which they fill the entries available to them (and nothing else). Most of the models' __call__ methods currently return a dict anyway, with their available information, so your task then would simply be to return the object rather then a dict.

2. Now the next place where we need to use ModelOutput object is a bit non-trivial. We need to use them right after internal (non __call__) model calls. Start with huggingface/lm.py, there in _loss_hook() method i already put a commented-out initialization of the wrapper. You should fill the wrapper with the available entries from the huggingface's forward pass, and use the output when providing the losses (in the few lines after the wrapper init) with output information (e.g., outputs.logits should now be outputs.output_logits).
	- The general idea here is that we'd have a unified ModelOutput API over all models. In the future we will use it to resolve losses universally.
    -  After completing this, please fill in the correct typing & shape to each models output entry in the ModelOutputWrapper.

3. All get_triggered_inputs should return a ModelInputs object, instead of dict. Again, they might not be able to fill all entries, but instead of filling the dict, they should fill a ModelInput instantiation


4. Now after that you've learned the context of this tasks, please choose where it's best to position the Model{Input,Output}Wrapper dataclasses. Also, choose a better name for them.`.

5. Now the final stage in this is to make a large function that is incharge of loss resolution and computation. The function will receive all available entries from ModelInputWrapper and ModelOutputWrapper along with the (potentially expanded) TargetsDict, and will compute the given loss based on the available entries -- or raise an error failing to do so. Currently, we repeat this resolution logic in multiple places (e.g., in huggingface/lm.py _loss_hook, and in model_mixins.py _calc_loss_from_outputs). We should unify this logic into a single place. After that, the models will only be responsible for providing the input/output wrappers (ie all the information they have), and the loss computation will be handled elsewhere. This way, new model implementers don't need to take care of the lengthy list of losses (which is expected to grow), and will only need to provide the input/output information -- the loss resolution module will do the rest.


6. go over EACH file under tropt/models and verify that it is aligned with this most recent refactoring. It is a critical refactoring and we must veryfy the overall design, typing, shaping, comments and obciously logic is aligned


Idea:

> What instead of forming a gigantic loss function, we would make sure that each loss class clearly states the model data it requires (e.g., through a specific argument name), and a specific target entry (if at all). Our tests will always make sure that these are the arguments of the losses' __call__ function. Then, the loss resolution function would simply invoke the loss function with the right arguments taken from the given model data! this sounds quite clean. Wdyt? 
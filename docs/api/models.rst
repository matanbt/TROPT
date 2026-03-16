.. module:: tropt.models

Models
======


Data Classes
------------
.. autoclass:: tropt.common.ModelInput
   :members:
   :undoc-members:

.. autoclass:: tropt.common.ModelOutput
   :members:
   :undoc-members:

.. raw:: html

   <hr class="hr-major">

Models Interface
-----------
.. autoclass:: BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LMBaseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EncoderBaseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BaseTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html

   <hr class="hr-major">

Model Mixins
------------
Model Mixins provide specific functionalities or access levels to the models. They define methods that models must implement to support various text optimization processes. For example, `LossTokenAccessMixin` defines methods for computing loss based on token-level inputs. This modular approach allows for flexible composition of model capabilities.

Token Mixins
^^^^^^^^^^^^
Token mixins define model interactions at the token level.

.. autoclass:: TokenAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LossTokenAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LogitsTokenAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GradientTokenAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html
   
   <hr class="hr-minor">

Text Mixins
^^^^^^^^^^^
Text mixins define model interactions at the text level.

.. autoclass:: TextAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LossTextAccessMixin
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html

   <hr class="hr-major">

Input Managers
--------------
Input Managers are responsible for streamlining the repeated combination of new triggers into text templates. They depend on the input type and are strongly linked to the model's key methods. For instance, `LMHFTokenInputsManager` specializes in combining trigger tokens within user text templates and providing them as model input for loss computation.

.. automodule:: tropt.models.inputs
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html

   <hr class="hr-major">

Model Implementations
---------------------

.. automodule:: tropt.models
   :members:
   :exclude-members: BaseModel, LMBaseModel, EncoderBaseModel, BaseTokenizer, TokenAccessMixin, LossTokenAccessMixin, LogitsTokenAccessMixin, GradientTokenAccessMixin, TextAccessMixin, LossTextAccessMixin, BatchedTargetsDict, InputsManager, MessageBatchedTargetsDict, TargetsDict, TargetsDictPlus, TextInputsManager, TokenInputsManager, TokenTrigger, TokenTriggerCandidates
   :undoc-members:
   :show-inheritance:
   :imported-members:
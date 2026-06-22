.. module:: tropt.loss

Losses
======

Loss Resolution
---------------

The unified entry point for computing any loss — models call this instead of invoking loss functions directly. See :doc:`common` for ``ModelInput`` / ``ModelOutput``.

.. autofunction:: tropt.loss.resolution.resolve_and_compute_loss

.. autoclass:: tropt.loss.resolution.LossResolutionError
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html

   <hr class="hr-major">

Loss Classes Interfaces
-----------------------

.. autoclass:: BaseLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TriggerLogitBasedLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AttentionBasedLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EmbeddingBasedLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TextBasedLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SteeringActivationLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CombinedLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. raw:: html

   <hr style="margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #eee;">

Loss Implementations
--------------------

.. automodule:: tropt.loss
   :members:
   :exclude-members: BaseLoss, TriggerLogitBasedLoss, AttentionBasedLoss, EmbeddingBasedLoss, TextBasedLoss, SteeringActivationLoss, CombinedLoss
   :undoc-members:
   :show-inheritance:
   :imported-members:

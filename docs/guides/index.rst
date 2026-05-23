Guides
======

TROPT can be used at three levels of customization:

1. **Run an existing recipe** from the Recipe Hub: a single function call that runs an end-to-end attack (jailbreak, corpus poisoning, prompt recovery, etc.).
2. **Compose your own recipe**: wire a *Model* + *Loss* + *Optimizer* + *Input Setup* of your choice into a custom function, with optional trackers, token constraints, and FLOP budgets.
3. **Extend TROPT with new components** — write a new loss, optimizer, or model backend in your own script. Any new component you build can be simply dropped into an existing recipe.

For an end-to-end notebook walkthrough see [`quickstart.ipynb`](https://github.com/matanbt/TROPT/blob/main/quickstart.ipynb) notebook.[TODO fix this link!]
If you want to *contribute* a component back to the TROPT package itself, see `CONTRIBUTING.md <https://github.com/matanbt/TROPT/blob/main/CONTRIBUTING.md>`_.

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: Run a Recipe
      :link: running_a_recipe
      :link-type: doc

      Call a Recipe Hub entry with one function. Covers the available recipes and what arguments each expects.

   .. grid-item-card:: Compose a Recipe
      :link: adding_a_recipe
      :link-type: doc

      Compose a Model + Loss + Optimizer + Inputs/Targets into a custom recipe.

   .. grid-item-card:: Add a Loss
      :link: adding_a_loss
      :link-type: doc

      Define a new objective.

   .. grid-item-card:: Add an Optimizer
      :link: adding_an_optimizer
      :link-type: doc

      Implement a discrete search algorithm.

   .. grid-item-card:: Add a Model
      :link: adding_a_model
      :link-type: doc

      Plug in a new model backend (HF, OpenAI, …).

   .. grid-item-card:: Compatibility Matrix
      :link: compatibility_matrix
      :link-type: doc

      Auto-generated map of which Optimizer × Loss × Model combinations are valid.

.. toctree::
   :hidden:

   running_a_recipe
   adding_a_recipe
   adding_a_loss
   adding_an_optimizer
   adding_a_model
   compatibility_matrix

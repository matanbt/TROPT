Guides
======

TROPT can be used at three levels of customization:

1. **Run an existing recipe** from the Recipe Hub: a single function call that runs an end-to-end attack (jailbreak, corpus poisoning, prompt recovery, etc.).
2. **Compose your own recipe**: wire a *Model* + *Loss* + *Optimizer* + *Input Setup* of your choice into a custom function, with optional trackers, token constraints, and FLOP budgets.
3. **Extend TROPT with new components** — write a new loss, optimizer, or model backend in your own script. Anything new component you build can be simply dropped into an existing recipe.

The guides below cover all three levels. For an end-to-end notebook walkthrough with examples see ``quickstart.ipynb`` at the repository root. 
If you want to *contribute* a component back to the TROPT package itself, see ``CONTRIBUTING.md``.

.. toctree::
   :maxdepth: 2

   running_a_recipe
   adding_a_recipe
   adding_a_loss
   adding_an_optimizer
   adding_a_model
   compatibility_matrix

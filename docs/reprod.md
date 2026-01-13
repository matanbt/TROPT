# Reproduction Expreiments
In the following experiments, we show TROPT's implementation reproduces other implementation and paper results. We also demonstrate that its flexability allows us to experiment existing attacks in novel settings.

> Commit SHA: `6839a7542e0d643539dd048d6f20a960a6d019b7`
---

## GCG implementation: TROPT vs. NanoGCG

In this experiment we show that TROPT's GCG implementation successfully reproduces the results from the GCG implementation of NanoGCG [[GCG paper](https://arxiv.org/abs/2307.15043), [NanoGCG code](https://github.com/GraySwanAI/nanoGCG/)].

> Reproducible evaluation script is located at [this NanoGCG fork](https://github.com/matanbt/nanoGCG/blob/main/eval_gcg.py).

**Setup.** We take 15 samples (=harmful instructions) from AdvBench and StrongReject datasets, and target [Gemma2-2b-it](https://huggingface.co/google/gemma-2-2b-it). We run both implementations on multiple seeds, with the hyperparameters specified in the GCG paper. We then compare both the optimziation process (by plotting the loss over optimization steps) and the final attack results (average jailbreak rate across all AdvBench instructions).

**Results (Optimization).** We comare the optimization process of both implementations by plotting the loss over optimization time (minutes).

For a specific harmful instruction from AdvBench:
![alt text](imgs/gcg/tropt-vs-nanogcg-single.png){width=400px style="display: block; margin: 0 auto;"}
Repeating this for more messages:
![alt text](imgs/gcg/tropt-vs-nanogcg-grid.png){width=300px style="display: block; margin: 0 auto;"}

Overall, we see that both implementations show similar optimization trends.

**Results (Final Attack).** We compare the final jailbreak rates (= average StrongReject score on all AdvBench instructions; can be seen as the "universality" score) achieved by both implementations.
![alt text](imgs/gcg/tropt-vs-nanogcg-univ.png){width=500px style="display: block; margin: 0 auto;"}

Again, we see that both implementations achieve similar final attack results.

---

## GASLITE implementation: TROPT vs. GASLITE paper results
> currently in tuning_gaslite script, need to move here.

In this experiments we show that TROPT's GASLITE implementation successfully reproduces the results from the GASLITE paper [[GASLITE paper](https://arxiv.org/abs/2412.20953)] and code [[GASLITE code](https://github.com/matanbt/GASLITE)].

**Setup.** We reproduce the concept specific setup in which a single adversarial passage attack is crafted to target queries of specific concepts; this setup is shown at section ? in the paper. We particularly target (and evaluate) the [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) embedding model, and follow the same parameters used in the paper.

**Results.** The plot compares the attack success of the paper and TROPT's implementation across the different targeted concepts. The attack success is defined as the rate of held-out queries for which the advdersarial passage appears in the top-10 retrieved results.

![alt text](imgs/gaslite/tropt-vs-gaslite-paper.png){width=400px style="display: block; margin: 0 auto;"}
*Comparison of TROPT's GASLITE implementation vs. GASLITE paper results in the concept-specific setup.*

Overall we spot a slight improvement in TROPT's implementation over the paper results, possibly due to minor implementation differences.

---

## Enhancing GASLITE: GASLITE Variants Comparison

Utilizing TROPT's flexability, we form various GASLITE variants by combining additional tricks from prior discrete-optimization-based attacks (`GASLITEPlusOptimizer`), we also implement a black-box variant for GASLITE that simply replaces the gradient with a random tensor (`RASLITEPlusOptimizer`) -- calling it *RASLITE*. We compare these variants in the same concept-specific setup as above.

Specifically, we compare:
- **GASLITE**: The original GASLITE optimizer as in the paper.
- **GASLITE+**: GASLITE combined with additional tricks from prior discrete-optimization-based attacks.
- **QGASLITE+**: accelerated GASLITE+ variants, that prefer slight speed improvement over performance.
- **RASLITE+**: A black-box variant of GASLITE+ that replaces the gradient with a random tensor.


**Results.** Show that we can get some gains by combining additional tricks to GASLITE (`GASLITE+`), and that the black-box variant (`RASLITE+`) also performs reasonably well.

![alt text](imgs/gaslite/tropt-vs-gaslite-vars.png){width=600px style="display: block; margin: 0 auto;"}
*Comparison of various GASLITE variants in the concept-specific setup.*

---
## Enhancing GCG: Using GASLITE for jailbreak
Next, we further utilize the flexability of TROPT to attack LLMs *using GASLITE* (which was originally designed for embedding models). We compare the performance of GASLITE and GCG in the jailbreak setup above.

**Results.** We do not see improvement neither in convergence, nor in final attack performance by using GASLITE for jailbreak. This may suggest generally that the preferences of embedding models and LLMs are different, and that attacks should be designed (and adapted) accordingly (!).


![alt text](imgs/gcg/tropt-gcg-vs-gaslite-single.png){width=400px style="display: block; margin: 0 auto;"}
*An example optimization process of a harmful instruction from AdvBench; comparing GCG and GASLITE optimizers.*

![alt text](imgs/gcg/tropt-gcg-vs-gaslite-univ.png){width=400px style="display: block; margin: 0 auto;"}
*Overall, we see that both implementations achieve similar final attack results.*

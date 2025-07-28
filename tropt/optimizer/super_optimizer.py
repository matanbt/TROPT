## vision: should be an optimizer the include the best of all worlds, and allow configurability; this will be super ugly code that basically allows highly configurable attack (and to grid over all attacks).
# - beam search (both between steps and within step [eg GASLITE is a specific case of beam search within step = greedy search])
# - gradient sampling / averaging
# - allow random grad / logits
# - buffer [?] (optimize simoultaneously multiple triggers candidates, track the path of each of diversity)
# .. this could be defeated by beam search and re-running the attack multiple times with different seeds & inits

from .BEAST import run_beast
from .GASLITE import run_gaslite
from .GASLITEPlus import run_gaslite_plus_encoder, run_gaslite_plus_llm
from .GCG import run_gcg, run_gcg_perplexity
from .GCGEmb import run_gcg_embedding_variant
from .GCGHij import run_gcghij
from .GCGMult import run_gcg_mutl_instruction
from .IRIS import run_iris
from .PEZ import run_pez
from .RASLITEPlus import run_rasliteplus, run_rasliteplus_llm

ATTACK_RECIPES = {
    "beast": run_beast,
    "gaslite": run_gaslite,
    "gasliteplus_encoder": run_gaslite_plus_encoder,
    "gasliteplus_llm": run_gaslite_plus_llm,
    "gcg": run_gcg,
    "gcg_perplexity": run_gcg_perplexity,
    "gcg_emb": run_gcg_embedding_variant,
    "gcg_hij": run_gcghij,
    "gcg_mult": run_gcg_mutl_instruction,
    "iris": run_iris,
    "pez": run_pez,
    "rasliteplus": run_rasliteplus,
    "rasliteplus_llm": run_rasliteplus_llm,
}

def list_attacks():
    return list(ATTACK_RECIPES.keys())

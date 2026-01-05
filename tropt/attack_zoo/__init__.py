from .BEAST import run_beast
from .GASLITE import run_gaslite
from .GCG import run_gcg
from .GCGEmb import run_gcg_embedding_variant
from .GCGMult import run_gcg_mutl_instruction
from .RASLITE import run_raslite
from .RASLITEPlus import run_rasliteplus

ATTACK_RECIPES = {
    "beast": run_beast,
    "gaslite": run_gaslite,
    "gcg": run_gcg,
    "gcg_emb": run_gcg_embedding_variant,
    "gcg_mult": run_gcg_mutl_instruction,
    "raslite": run_raslite,
    "rasliteplus": run_rasliteplus,
}

def list_attacks():
    return list(ATTACK_RECIPES.keys())

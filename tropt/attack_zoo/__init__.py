from .BEAST import run_beast
from .GASLITE import run_gaslite
from .GCG import run_gcg
from .GCGEmb import run_gcg_embedding_variant
from .GCGMult import run_gcg_mutl_instruction
from .IRIS import run_iris
from .RASLITEPlus import run_rasliteplus

ATTACK_RECIPES = {
    "beast": run_beast,
    "gaslite": run_gaslite,
    "gcg": run_gcg,
    "gcg_emb": run_gcg_embedding_variant,
    "gcg_mult": run_gcg_mutl_instruction,
    "iris": run_iris,
    "rasliteplus": run_rasliteplus,
}

def list_attacks():
    return list(ATTACK_RECIPES.keys())

from .AdvDecoding import run_advdecoding_jailbreak, run_advdecoding_retrieval
from .BEAST import run_beast
from .ClassifierGCG import run_classifier_gcg
from .GASLITE import run_gaslite
from .GASLITEPlus import run_gaslite_plus_encoder, run_gaslite_plus_llm
from .GBDA import run_gbda
from .GCG import run_gcg, run_gcg_perplexity
from .GCGEmb import run_gcg_embedding_variant
from .GCGHij import run_gcghij
from .GCGMult import run_gcg_mutl_instruction
from .IRIS import run_iris, run_iris2
from .PEZ import run_pez
from .PGD import run_pgd
from .RASLITEPlus import run_rasliteplus, run_rasliteplus_llm
from .SoftGCG import run_soft_gcg
from .SoftPrompt import run_soft_prompt, run_soft_prompt_encoder_attack

ATTACK_RECIPES = {
    "advdecoding_jailbreak": run_advdecoding_jailbreak,
    "advdecoding_retrieval": run_advdecoding_retrieval,
    "beast": run_beast,
    "classifier_gcg": run_classifier_gcg,
    "gaslite": run_gaslite,
    "gasliteplus_encoder": run_gaslite_plus_encoder,
    "gasliteplus_llm": run_gaslite_plus_llm,
    "gbda": run_gbda,
    "gcg": run_gcg,
    "gcg_perplexity": run_gcg_perplexity,
    "gcg_emb": run_gcg_embedding_variant,
    "gcg_hij": run_gcghij,
    "gcg_mult": run_gcg_mutl_instruction,
    "iris": run_iris,
    "iris2": run_iris2,
    "pez": run_pez,
    "pgd": run_pgd,
    "rasliteplus": run_rasliteplus,
    "rasliteplus_llm": run_rasliteplus_llm,
    "soft_gcg": run_soft_gcg,
    "soft_prompt": run_soft_prompt,
    "soft_prompt_encoder": run_soft_prompt_encoder_attack,
}

def list_attacks():
    return list(ATTACK_RECIPES.keys())

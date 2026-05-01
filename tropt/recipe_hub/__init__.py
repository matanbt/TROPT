# Naming convention for recipe entry points (function names == dict keys):
#
#   {method}[_{variant}][_{task}][__{paperYYYY}]
#
# The trailing `__{paperYYYY}` ("reproduction tag") is reserved for recipes that
# precisely reproduce a published method's algorithm and hyperparameters as
# stated in the paper. Setting-ports (e.g. CLIP→LM) are allowed when the paper's
# algorithm transfers cleanly. Recipes that deviate from the paper's algorithm,
# or that introduce knobs not in the paper, MUST NOT carry the reproduction tag.
#
# See docs/guides/adding_a_recipe.md for details.

from .AdvDecoding__zhang2024 import (
    advdecoding_jailbreak__zhang2024,
    advdecoding_retrieval__zhang2024,
)
from .ARCA__jones2023 import arca__jones2023
from .ARCAToxicReverse import arca_toxic_reverse
from .AutoPrompt__shin2020 import autoprompt__shin2020
from .BEAST__sadasivan2024 import beast__sadasivan2024
from .ClassifierGCG import classifier_gcg
from .GASLITE__bentov2024 import gaslite__bentov2024
from .GASLITEPlus import gasliteplus_encoder, gasliteplus_llm
from .GBDA__guo2021 import gbda__guo2021
from .GCG__zou2023 import gcg__zou2023, gcg_perplexity
from .GCGEmb import gcg_emb
from .GCGHij import gcg_hij
from .GCGMult__zou2023 import gcg_mult__zou2023
from .FLRTDistill import flrt_distill
from .HotFlip__ebrahimi2018 import hotflip__ebrahimi2018
from .IRIS__huang2025 import iris__huang2025, iris2
from .MAC__wang2024 import mac__wang2024
from .PAL__sitawarin2024 import gcgp_pal__sitawarin2024, pal__sitawarin2024, ral__sitawarin2024
from .PEZ__wen2023 import pez__wen2023
from .PromptRecovery__williams2025 import (
    prompt_recovery__williams2025,
    evaluate_prompt_recovery,
    generate_image_from_prompt,
    get_image_embedding_for_clip_model,
)
from .PGD__geisler2024 import pgd__geisler2024
from .PRS import prs
from .RSEmb import rs_emb
from .QCG__hayase2024 import (
    gcgp_blackbox__hayase2024,
    gcgp_whitebox__hayase2024,
    qcg__hayase2024,
)
from .RASLITEPlus import rasliteplus, rasliteplus_llm
from .SoftGCG import soft_gcg
from .SoftPrompt import soft_prompt, soft_prompt_encoder
from .UAT import uat_classifier, uat_prompt_injection

RECIPES = {
    "advdecoding_jailbreak__zhang2024": advdecoding_jailbreak__zhang2024,
    "advdecoding_retrieval__zhang2024": advdecoding_retrieval__zhang2024,
    "arca__jones2023": arca__jones2023,
    "arca_toxic_reverse": arca_toxic_reverse,
    "autoprompt__shin2020": autoprompt__shin2020,
    "beast__sadasivan2024": beast__sadasivan2024,
    "classifier_gcg": classifier_gcg,
    "gaslite__bentov2024": gaslite__bentov2024,
    "gasliteplus_encoder": gasliteplus_encoder,
    "gasliteplus_llm": gasliteplus_llm,
    "gbda__guo2021": gbda__guo2021,
    "gcg__zou2023": gcg__zou2023,
    "gcg_perplexity": gcg_perplexity,
    "gcgp_pal__sitawarin2024": gcgp_pal__sitawarin2024,
    "gcg_emb": gcg_emb,
    "gcg_hij": gcg_hij,
    "hotflip__ebrahimi2018": hotflip__ebrahimi2018,
    "gcg_mult__zou2023": gcg_mult__zou2023,
    "flrt_distill": flrt_distill,
    "iris__huang2025": iris__huang2025,
    "iris2": iris2,
    "mac__wang2024": mac__wang2024,
    "pal__sitawarin2024": pal__sitawarin2024,
    "pez__wen2023": pez__wen2023,
    "pgd__geisler2024": pgd__geisler2024,
    "prs": prs,
    "rs_emb": rs_emb,
    "qcg__hayase2024": qcg__hayase2024,
    "gcgp_blackbox__hayase2024": gcgp_blackbox__hayase2024,
    "gcgp_whitebox__hayase2024": gcgp_whitebox__hayase2024,
    "ral__sitawarin2024": ral__sitawarin2024,
    "rasliteplus": rasliteplus,
    "rasliteplus_llm": rasliteplus_llm,
    "soft_gcg": soft_gcg,
    "soft_prompt": soft_prompt,
    "soft_prompt_encoder": soft_prompt_encoder,
    "prompt_recovery__williams2025": prompt_recovery__williams2025,
    "uat_classifier": uat_classifier,
    "uat_prompt_injection": uat_prompt_injection,
}


def list_recipes():
    return list(RECIPES.keys())

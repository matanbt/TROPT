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
from .ask_for_directions__zhang2025 import ask_for_directions__zhang2025
from .ARCAToxicReverse import arca_toxic_reverse
from .AutoPrompt__shin2020 import autoprompt__shin2020
from .BEAST__sadasivan2024 import beast__sadasivan2024
from .FLRT__thompson2024 import flrt_distill
from .GASLITE__bentov2024 import (
    gaslite__bentov2024,
    gasliteplus_encoder,
    gasliteplus_llm,
)
from .GBDA__guo2021 import gbda__guo2021
from .GCG__zou2023 import (
    classifier_gcg,
    gcg__zou2023,
    gcg_emb,
    gcg_perplexity,
)
from .GCGHij import attn_gcg__wang2024, gcg_hij__bentov2025
from .GCGMult__zou2023 import gcg_mult__zou2023
from .HotFlip__ebrahimi2018 import hotflip__ebrahimi2018
from .IRIS__huang2025 import iris__huang2025, iris2
from .MAC__zhang2024 import mac__zhang2024
from .McPAL import mcpal
from .PAL__sitawarin2024 import (
    gcgp_pal__sitawarin2024,
    pal__sitawarin2024,
    ral__sitawarin2024,
)
from .PEZ__wen2023 import pez__wen2023
from .PromptRecovery__wen2023 import (
    evaluate_prompt_recovery,
    generate_image_from_prompt,
    get_image_embedding_for_clip_model,
    prompt_recovery__wen2023,
)
from .PRS__andriushchenko2024 import (
    prs__andriushchenko2024,
    rs_emb,
    rs_oracle,
)
from .QCG__hayase2024 import (
    gcgp_blackbox__hayase2024,
    gcgp_whitebox__hayase2024,
    qcg__hayase2024,
)
from .RASLITEPlus import rasliteplus, rasliteplus_llm
from .SoftPrompt__schwinn2024 import soft_prompt__schwinn2024, soft_prompt_encoder
from .UAT import uat_classifier, uat_prompt_injection
from .utils import generate_from_model


# Naming: paper reproductions are written with first (with `__paperYYYY` tag), then variants/extensions/applications.
RECIPES = {
    # HotFlip (Ebrahimi 2018)
    "hotflip__ebrahimi2018": hotflip__ebrahimi2018,

    # UAT (Wallace 2019)
    "uat_classifier": uat_classifier,
    "uat_prompt_injection": uat_prompt_injection,

    # AutoPrompt (Shin 2020)
    "autoprompt__shin2020": autoprompt__shin2020,

    # GBDA (Guo 2021)
    "gbda__guo2021": gbda__guo2021,

    # PEZ (Wen 2023, Williams 2025)
    "pez__wen2023": pez__wen2023,
    "prompt_recovery__wen2023": prompt_recovery__wen2023,

    # ARCA (Jones 2023)
    "arca__jones2023": arca__jones2023,
    "arca_toxic_reverse": arca_toxic_reverse,

    # GCG (Zou 2023)
    "gcg__zou2023": gcg__zou2023,
    "gcg_mult__zou2023": gcg_mult__zou2023,
    "gcg_perplexity": gcg_perplexity,
    "gcg_emb": gcg_emb,
    "gcg_hij__bentov2025": gcg_hij__bentov2025,
    "attn_gcg__wang2024": attn_gcg__wang2024,
    "classifier_gcg": classifier_gcg,

    # Soft Prompt (Schwinn 2024) — embedding-space
    "soft_prompt__schwinn2024": soft_prompt__schwinn2024,
    "soft_prompt_encoder": soft_prompt_encoder,

    # PAL (Sitawarin 2024)
    "pal__sitawarin2024": pal__sitawarin2024,
    "ral__sitawarin2024": ral__sitawarin2024,
    "gcgp_pal__sitawarin2024": gcgp_pal__sitawarin2024,

    # QCG (Hayase 2024)
    "qcg__hayase2024": qcg__hayase2024,
    "gcgp_whitebox__hayase2024": gcgp_whitebox__hayase2024,
    "gcgp_blackbox__hayase2024": gcgp_blackbox__hayase2024,

    # BEAST (Sadasivan 2024)
    "beast__sadasivan2024": beast__sadasivan2024,

    # PRS (Andriushchenko 2024) + black-box random-search variants
    "prs__andriushchenko2024": prs__andriushchenko2024,
    "rs_emb": rs_emb,
    "rs_oracle": rs_oracle,

    # Ask-For-Directions on RS (Zhang 2025)
    "ask_for_directions__zhang2025": ask_for_directions__zhang2025,

    # MAC (Zhang & Wei 2024)
    "mac__zhang2024": mac__zhang2024,
    "mcpal": mcpal,

    # FLRT (Thompson & Sklar 2024)
    "flrt_distill": flrt_distill,

    # AdvDecoding (Zhang 2024)
    "advdecoding_jailbreak__zhang2024": advdecoding_jailbreak__zhang2024,
    "advdecoding_retrieval__zhang2024": advdecoding_retrieval__zhang2024,

    # GASLITE (Ben-Tov 2024) + extensions
    "gaslite__bentov2024": gaslite__bentov2024,
    "gasliteplus_encoder": gasliteplus_encoder,
    "gasliteplus_llm": gasliteplus_llm,
    "rasliteplus": rasliteplus,
    "rasliteplus_llm": rasliteplus_llm,

    # IRIS (Huang 2025)
    "iris__huang2025": iris__huang2025,
    "iris2": iris2,
}


def list_recipes():
    return list(RECIPES.keys())

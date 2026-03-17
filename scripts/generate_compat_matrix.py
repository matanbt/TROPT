"""Generate optimizer-model-loss compatibility matrix as a Markdown doc.

Lightweight script — no GPU, no model loading. Uses:
  - Auto-discovery of optimizers, models, and losses from __init__.py exports
  - Python introspection for optimizer requirements and loss signatures
  - AST parsing to discover which ModelOutput/ModelInput fields each model
    populates in its token-path vs text-path methods

The compatibility check mirrors what happens at runtime in resolve_and_compute_loss():
  1. Does the model satisfy the optimizer's mixin requirements?
  2. For the optimizer's access path, which ModelOutput/ModelInput fields does
     the model populate? (discovered via AST on the model's source + MRO)
  3. Combined with the user-provided target fields, can all required parameters
     of the loss function's __call__ be resolved?

Output: docs/compatibility_matrix.md
"""

import ast
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type

import tropt.loss as loss_pkg
import tropt.model as model_pkg
import tropt.optimizer as optimizer_pkg
from tropt.common import MessageTargets
from tropt.loss import BaseLoss, CombinedLoss
from tropt.model import (
    BaseModel,
    GradientEmbedAccessMixin,
    GradientTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.optimizer import BaseOptimizer

# ---------------------------------------------------------------------------
# Auto-discovery from package exports
# ---------------------------------------------------------------------------

def _discover_optimizers() -> List[Type[BaseOptimizer]]:
    """Find all concrete BaseOptimizer subclasses exported by tropt.optimizer."""
    result = [
        obj for name in dir(optimizer_pkg)
        if isinstance(obj := getattr(optimizer_pkg, name), type)
        and issubclass(obj, BaseOptimizer)
        and obj is not BaseOptimizer
        and not inspect.isabstract(obj)
    ]
    return sorted(result, key=lambda c: c.__name__)


def _discover_models() -> List[Tuple[type, str]]:
    """Find all concrete BaseModel subclasses exported by tropt.model."""
    result = []
    for name in dir(model_pkg):
        obj = getattr(model_pkg, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and not inspect.isabstract(obj)
            and "Base" not in name
            and "Mixin" not in name
        ):
            result.append((obj, name))
    return sorted(result, key=lambda t: t[1])


def _is_concrete_loss(cls: type) -> bool:
    return cls is not BaseLoss and cls is not CombinedLoss and not inspect.isabstract(cls)


def _discover_concrete_losses() -> List[Type[BaseLoss]]:
    """Find all concrete loss classes exported by tropt.loss."""
    result = []
    for name in dir(loss_pkg):
        obj = getattr(loss_pkg, name)
        if isinstance(obj, type) and issubclass(obj, BaseLoss) and _is_concrete_loss(obj):
            result.append(obj)
    return sorted(result, key=lambda c: c.__name__)


# ---------------------------------------------------------------------------
# AST-based field discovery
# ---------------------------------------------------------------------------
#
# Both ModelOutput and ModelInput are constructed in model source files.
# The same file may build them differently in token-path methods (e.g.
# invoke_from_tokens, get_triggered_inputs) vs text-path methods (invoke_from_texts,
# compute_loss_from_texts). We classify methods by name keywords and
# extract the keyword arguments from each constructor call.
#
# Methods matching neither set are conservatively assigned to both paths.

_TOKEN_PATH_KEYWORDS = {"token", "invoke_from_tokens", "loss_hook", "triggered"}
_TEXT_PATH_KEYWORDS = {"invoke_from_texts", "text"}


def _extract_constructor_fields(
    source_file: str, class_name: str,
) -> Tuple[Set[str], Set[str]]:
    """Find all `class_name(...)` calls in source_file, split by access path."""
    tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))

    token_fields: Set[str] = set()
    text_fields: Set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        method_name = node.name
        is_token = any(kw in method_name for kw in _TOKEN_PATH_KEYWORDS)
        is_text = any(kw in method_name for kw in _TEXT_PATH_KEYWORDS)

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            call_name = getattr(child.func, "id", None) or getattr(child.func, "attr", None)
            if call_name != class_name:
                continue

            kw_names = {kw.arg for kw in child.keywords if kw.arg is not None}
            if is_token and not is_text:
                token_fields |= kw_names
            elif is_text and not is_token:
                text_fields |= kw_names
            else:
                token_fields |= kw_names
                text_fields |= kw_names

    return token_fields, text_fields


def _discover_fields_from_mro(
    model_cls: type, class_name: str,
) -> Tuple[Set[str], Set[str]]:
    """Walk MRO and union constructor fields from all source files."""
    token_all: Set[str] = set()
    text_all: Set[str] = set()
    seen_files: Set[str] = set()

    for cls in model_cls.__mro__:
        try:
            src = inspect.getfile(cls)
        except TypeError:
            continue
        if src in seen_files:
            continue
        seen_files.add(src)
        tok, txt = _extract_constructor_fields(src, class_name)
        token_all |= tok
        text_all |= txt

    return token_all, text_all


# Target sub-fields — derived from the dataclass, always available from user input
_TARGET_FIELDS = set(MessageTargets.model_fields.keys())


# ---------------------------------------------------------------------------
# Core compatibility logic
# ---------------------------------------------------------------------------

def _get_loss_required_params(loss_cls: Type[BaseLoss]) -> List[str]:
    sig = inspect.signature(loss_cls.__call__)
    return [
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty
    ]


def _can_resolve_loss(loss_cls: Type[BaseLoss], available_fields: Set[str]) -> bool:
    return all(p in available_fields for p in _get_loss_required_params(loss_cls))


def _optimizer_supports_model(opt_cls: Type[BaseOptimizer], model_cls: type) -> bool:
    reqs = opt_cls.model_requirements
    return all(issubclass(model_cls, m) for m in reqs) if reqs else True


def _get_access_path(opt_cls: Type[BaseOptimizer]) -> str:
    reqs = opt_cls.model_requirements
    if reqs and any(issubclass(m, LossTextAccessMixin) for m in reqs):
        return "text"
    return "token"


def get_supported_losses(
    opt_cls: Type[BaseOptimizer],
    model_cls: type,
    field_cache: Dict[type, Dict[str, Set[str]]],
    concrete_losses: List[Type[BaseLoss]],
) -> List[str] | None:
    if not _optimizer_supports_model(opt_cls, model_cls):
        return None

    access = _get_access_path(opt_cls)
    available = (
        field_cache[model_cls][f"output_{access}"]
        | field_cache[model_cls][f"input_{access}"]
        | _TARGET_FIELDS
    )

    return [
        loss_cls.__name__
        for loss_cls in concrete_losses
        if _can_resolve_loss(loss_cls, available)
    ]


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown() -> str:
    optimizers = _discover_optimizers()
    models = _discover_models()
    concrete_losses = _discover_concrete_losses()

    print(f"Discovered {len(optimizers)} optimizers, {len(models)} models, {len(concrete_losses)} losses")

    # Pre-compute all fields per model via AST
    field_cache: Dict[type, Dict[str, Set[str]]] = {}
    for model_cls, name in models:
        out_tok, out_txt = _discover_fields_from_mro(model_cls, "ModelOutput")
        in_tok, in_txt = _discover_fields_from_mro(model_cls, "ModelInput")
        field_cache[model_cls] = {
            "output_token": out_tok,
            "output_text": out_txt,
            "input_token": in_tok,
            "input_text": in_txt,
        }

    lines: List[str] = []
    lines.append("# Optimizer-Model-Loss Compatibility Matrix")
    lines.append("")
    lines.append("> **Auto-generated** by `scripts/generate_compat_matrix.py` — do not edit manually.")
    lines.append("")
    lines.append("Each cell lists the concrete loss functions supported for the given optimizer-model pair,")
    lines.append("or shows **Unsupported** if the model does not satisfy the optimizer's requirements.")
    lines.append("")

    # Main table
    model_names = [name for _, name in models]
    lines.append("| Optimizer | " + " | ".join(model_names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(models)) + "|")

    for opt_cls in optimizers:
        cells = []
        for model_cls, _ in models:
            losses = get_supported_losses(opt_cls, model_cls, field_cache, concrete_losses)
            if losses is None:
                cells.append("**Unsupported**")
            elif not losses:
                cells.append("*No matching losses*")
            else:
                cells.append(", ".join(f"`{l}`" for l in losses))
        lines.append(f"| **{opt_cls.__name__}** | " + " | ".join(cells) + " |")

    # Legend: optimizer access levels
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("### Access levels required by each optimizer")
    lines.append("")
    lines.append("| Optimizer | Required Mixins | Access Level |")
    lines.append("|---|---|---|")
    for opt_cls in optimizers:
        reqs = opt_cls.model_requirements
        if not reqs:
            mixin_str, level = "*(none)*", "Any"
        else:
            mixin_str = ", ".join(f"`{m.__name__}`" for m in reqs)
            if any(issubclass(m, GradientTokenAccessMixin) for m in reqs):
                level = "White-box"
            elif any(issubclass(m, GradientEmbedAccessMixin) for m in reqs):
                level = "White-box (embedding)"
            elif any(issubclass(m, LossTokenAccessMixin) for m in reqs):
                level = "Grey-box"
            elif any(issubclass(m, LossTextAccessMixin) for m in reqs):
                level = "Black-box"
            else:
                level = "Custom"
        lines.append(f"| {opt_cls.__name__} | {mixin_str} | {level} |")

    # Legend: loss functions
    lines.append("")
    lines.append("### Concrete loss functions")
    lines.append("")
    lines.append("| Loss | Base Type | Required Parameters |")
    lines.append("|---|---|---|")
    for loss_cls in concrete_losses:
        for base in loss_cls.__mro__:
            if base is loss_cls or base is BaseLoss or base is object:
                continue
            if issubclass(base, BaseLoss):
                base_name = base.__name__
                break
        else:
            base_name = "BaseLoss"
        params = _get_loss_required_params(loss_cls)
        lines.append(f"| `{loss_cls.__name__}` | `{base_name}` | {', '.join(f'`{p}`' for p in params)} |")

    # Legend: discovered fields (transparency / debugging)
    lines.append("")
    lines.append("### Discovered fields per model (via source AST)")
    lines.append("")
    lines.append("| Model | Path | ModelOutput fields | ModelInput fields |")
    lines.append("|---|---|---|---|")
    for model_cls, name in models:
        f = field_cache[model_cls]
        for path in ("token", "text"):
            out_str = ", ".join(f"`{x}`" for x in sorted(f[f"output_{path}"])) or "*(none)*"
            in_str = ", ".join(f"`{x}`" for x in sorted(f[f"input_{path}"])) or "*(none)*"
            lines.append(f"| {name} | {path} | {out_str} | {in_str} |")

    lines.append("")
    return "\n".join(lines)


def main():
    md = generate_markdown()
    out_path = Path(__file__).resolve().parent.parent / "docs" / "compatibility_matrix.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

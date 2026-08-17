"""Insert `from __future__ import annotations` into every `tropt/*.py`.

Sphinx autodoc can't resolve TROPT's jaxtyping/pydantic forward refs without it
(https://github.com/sphinx-doc/sphinx/issues/11211). `common.py` is excluded:
its pydantic models use runtime jaxtyping field types that break when
stringified. The import goes *after* any module docstring so the docstring still
renders on the autodoc module pages.

Run against a throwaway copy of the tree — it rewrites files in place.

Usage: python docs/scripts/inject_annotations.py [package_dir]
"""

import ast
import pathlib
import sys

EXCLUDED = {"common.py"}


def inject(package_dir: str | pathlib.Path = "tropt") -> int:
    """Rewrite every eligible module under `package_dir`. Returns the count."""
    n = 0
    for path in pathlib.Path(package_dir).rglob("*.py"):
        if path.name in EXCLUDED:
            continue
        src = path.read_text(encoding="utf-8")
        if "from __future__ import annotations" in src:
            continue

        insert_at = 0
        try:
            first = (ast.parse(src).body or [None])[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
            ):
                insert_at = first.end_lineno
        except SyntaxError:
            insert_at = 0

        lines = src.splitlines(keepends=True)
        lines.insert(insert_at, "from __future__ import annotations\n")
        path.write_text("".join(lines), encoding="utf-8")
        n += 1
    return n


if __name__ == "__main__":
    print(f"Injected future annotations into {inject(*sys.argv[1:])} module(s).")

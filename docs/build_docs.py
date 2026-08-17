import glob
import os
import shutil
import stat
import subprocess
import sys
import time
import webbrowser


def _force_writable(func, path, _exc):
    """rmtree error hook: clear the read-only bit (git/GitHub files) and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# `onexc` replaced `onerror` in Python 3.12; the project supports 3.10+.
_RMTREE_HOOK = (
    {"onexc": _force_writable} if sys.version_info >= (3, 12)
    else {"onerror": _force_writable}
)


def robust_cleanup(path):
    """Delete `path`, retrying while Google Drive / Windows holds a lock.

    Returns True if the tree is gone, False if it stayed locked.
    """
    if not os.path.exists(path):
        return True

    print(f"Cleaning up {path}...")
    for _ in range(5):
        try:
            shutil.rmtree(path, **_RMTREE_HOOK)
            return True
        except OSError:
            time.sleep(0.5)  # Wait for Google Drive/Windows to release lock

    print(f"  Warning: Could not fully delete {path}. Some files are locked.")
    return False

def build_docs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use a unique temp dir if the default one is locked
    # This guarantees we never crash on FileExistsError
    default_temp = os.path.join(project_root, "_temp_build_env")

    if os.path.exists(default_temp) and not robust_cleanup(default_temp):
        print("  Default temp dir is locked. Using a new unique directory...")
        temp_root = os.path.join(project_root, f"_temp_build_env_{int(time.time())}")
    else:
        temp_root = default_temp

    src_original = os.path.join(project_root, "tropt")
    docs_original = os.path.join(project_root, "docs")

    src_copy = os.path.join(temp_root, "tropt")
    docs_copy = os.path.join(temp_root, "docs")

    final_build_dir = os.path.join(docs_original, "_build", "html")

    # ---------------------------------------------------------
    # 1. Copy Code & Docs
    # ---------------------------------------------------------
    print(f"Creating build environment in {temp_root}...")

    # Exclude all version control and cache files
    ignore_patterns = shutil.ignore_patterns(
        '.git', '.github', '.idea', '.vscode',
        '__pycache__', '*.pyc', '_build', 'Thumbs.db'
    )

    # dirs_exist_ok=True prevents crashing if the folder partially exists
    shutil.copytree(src_original, src_copy, ignore=ignore_patterns, dirs_exist_ok=True)
    shutil.copytree(docs_original, docs_copy, ignore=ignore_patterns, dirs_exist_ok=True)

    # ---------------------------------------------------------
    # 2. Inject 'from __future__ import annotations'
    # ---------------------------------------------------------
    # Same script CI runs (deploy_docs.yml), pointed at the throwaway copy.
    print("Injecting future annotations...")
    from docs.scripts.inject_annotations import inject
    print(f"  Injected into {inject(src_copy)} module(s).")

    # NOTE: inlining + sanitizing tropt/recipe_hub/README.md now happens in
    # conf.py (the `include-read` event), so it runs for every build path.

    # ---------------------------------------------------------
    # 3. Generate compatibility matrix
    # ---------------------------------------------------------
    print("Generating compatibility matrix...")
    try:
        from docs.scripts.generate_compat_matrix import generate_markdown
        md = generate_markdown()
        compat_path = os.path.join(docs_copy, "guides", "compatibility_matrix.md")
        with open(compat_path, "w", encoding="utf-8") as f:
            f.write(md)
        # Also write to original docs so it's available outside builds
        with open(os.path.join(docs_original, "guides", "compatibility_matrix.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print("  Compatibility matrix generated.")
    except Exception as e:
        print(f"  Warning: Could not generate compatibility matrix: {e}")

    # ---------------------------------------------------------
    # 4. Run Sphinx
    # ---------------------------------------------------------
    print("Running Sphinx...")
    build_cmd = [
        "sphinx-build",
        "-b", "html",
        docs_copy,
        os.path.join(docs_copy, "_build", "html")
    ]

    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Sphinx build failed.")
        return

    # ---------------------------------------------------------
    # 5. Copy Artifacts Back
    # ---------------------------------------------------------
    print(f"Copying build artifacts to {final_build_dir}...")

    # Clean destination (ignore errors here to avoid crashing the whole script)
    robust_cleanup(final_build_dir)

    try:
        shutil.copytree(
            os.path.join(docs_copy, "_build", "html"),
            final_build_dir,
            dirs_exist_ok=True
        )
    except Exception as e:
        print(f"Error copying artifacts: {e}")
        print(f"You can find the raw build at: {os.path.join(docs_copy, '_build', 'html')}")

    # ---------------------------------------------------------
    # 6. Cleanup Temp Dir
    # ---------------------------------------------------------
    robust_cleanup(temp_root)

    # Clean up any other stale temp dirs from previous runs
    for path in glob.glob(os.path.join(project_root, "_temp_build_env_*")):
        robust_cleanup(path)

    # ---------------------------------------------------------
    # 7. Open in Browser (Optional)
    # ---------------------------------------------------------
    index_path = os.path.join(final_build_dir, "index.html")
    if os.path.exists(index_path):
        print(f"Opening docs in browser: {index_path}")
        # 'file://' prefix is required for some browsers
        webbrowser.open(f"file://{os.path.abspath(index_path)}")

    print("Done! Docs updated successfully.")

if __name__ == "__main__":
    build_docs()

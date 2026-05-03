import os
import shutil
import subprocess
import sys
import stat
import time
import glob
import webbrowser

def make_writable(path):
    """Force a file/directory to be writable."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass

def robust_cleanup(path):
    """
    Aggressively cleans up a directory.
    1. Walks tree to fix permissions (handling read-only Git/GitHub files).
    2. Tries to delete.
    3. Returns True if successful, False if locked.
    """
    if not os.path.exists(path):
        return True

    print(f"Cleaning up {path}...")
    
    # 1. Force permissions first (Pre-emptive strike)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            make_writable(os.path.join(root, d))
        for f in files:
            make_writable(os.path.join(root, f))
    make_writable(path)

    # 2. Try deletion with retries
    for i in range(5):
        try:
            shutil.rmtree(path)
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
    print("Injecting future annotations...")
    for root, _, files in os.walk(src_copy):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if "from __future__ import annotations" not in content:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("from __future__ import annotations\n" + content)
                except Exception as e:
                    print(f"  Skipping {file}: {e}")

    # ---------------------------------------------------------
    # 2.5. Inline tropt/recipe_hub/README.md into docs/api/recipe_hub.rst
    # ---------------------------------------------------------
    print("Inlining recipe_hub/README.md into recipe_hub.rst...")
    recipe_hub_rst = os.path.join(docs_copy, "api", "recipe_hub.rst")
    placeholder = (
        ".. [[[ THIS WILL BE REPLACED WITH tropt/recipe_hub/README.md "
        "AT BUILD TIME ]]]"
    )
    # myst_parser's :parser: option lets an .rst file include and parse a
    # markdown file inline. Path is relative to recipe_hub.rst inside the
    # temp build dir (docs_copy/api/ → ../../tropt/recipe_hub/README.md).
    replacement = (
        ".. include:: ../../tropt/recipe_hub/README.md\n"
        "   :parser: myst_parser.sphinx_"
    )
    try:
        with open(recipe_hub_rst, "r", encoding="utf-8") as f:
            content = f.read()
        if placeholder not in content:
            print(f"  Warning: placeholder not found in {recipe_hub_rst}")
        else:
            content = content.replace(placeholder, replacement)
            with open(recipe_hub_rst, "w", encoding="utf-8") as f:
                f.write(content)
            print("  recipe_hub.rst inlined.")
    except Exception as e:
        print(f"  Warning: could not inline README: {e}")

    # ---------------------------------------------------------
    # 3. Generate compatibility matrix
    # ---------------------------------------------------------
    print("Generating compatibility matrix...")
    try:
        from scripts.generate_compat_matrix import generate_markdown
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
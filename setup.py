import io
import os
from setuptools import find_packages, setup

# Read the README for the long description
with io.open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tropt",
    version="0.0.1",
    description="A toolbox for optimizing discrete text triggers.",
    license="MIT",

    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Matan Ben-Tov",
    url="https://github.com/matanbt/text-trigger-opt-toolbox",

    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "torch>=2.4.0",
        "tqdm",
        "jaxtyping",
        "numpy>=2.0.0",
        "transformers>=4.45.0",
        "accelerate>=1.0.0",
        "hydra-core>=1.3.0",
        "omegaconf",
        "sentence-transformers>=5.1.0",
        "wandb",
        "livelossplot",
        "openai",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "pre-commit",
        ],
    },
    python_requires=">=3.10",
    include_package_data=True,
)

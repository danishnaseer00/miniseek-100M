from setuptools import setup, find_packages

setup(
    name="miniseek",
    version="0.1.0",
    description="Incremental LLM Research Project - Miniseek",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)

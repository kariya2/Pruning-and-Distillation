from setuptools import setup, find_packages

setup(
    name="distillation",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy<2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "pytest>=7.0.0",
    ],
) 
from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="derivkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "numdifftools"
    ],
    author="Nikolina Šarčević, Matthijs van der Wild",
    author_email="nikolina.sarcevic@gmail.com",
    description="Toolkit for estimating derivatives using hybrid polynomial fitting and stencil methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
)

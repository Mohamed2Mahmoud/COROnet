"""Setup script for the coronet package."""

from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).parent
_REQUIREMENTS = (_ROOT / "requirements.txt").read_text().splitlines()
_REQUIREMENTS = [line.strip() for line in _REQUIREMENTS if line.strip() and not line.startswith("#")]

setup(
    name="coronet",
    version="1.0.0",
    description="An object-oriented pipeline for coronary artery disease analysis on cardiac CT.",
    long_description=(_ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Mohamed Mahmoud",
    url="https://github.com/Mohamed2Mahmoud/COROnet",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10",
    install_requires=_REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "coronet=scripts.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

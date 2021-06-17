from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="happy",
    version="0.0.1",
    author="Claudia Vanea",
    description="The Histology Analysis Pipeline.py (HAPPY)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nellaker-group/HAPPY",
    packages=find_packages(include=["happy"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "eval=projects.placenta.eval:main",
            "train=projects.placenta.train:main",
            "nuc_train=projects.placenta.train:nuc_main",
            "cell_train=projects.placenta.train:cell_main",
        ]
    },
)

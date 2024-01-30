from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="tnreason",
    version="0.2",
    author="Alex Goessmann",
    author_email="alex.goessmann@web.de",
    description="A package for reasoning with tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    install_requires=[
        "numpy>=1.17", # was 1.23.4",
        "pandas>=1.0.0",
        "networkx",
        "pyyaml",
        "rdflib",
        "matplotlib"
    ],
    python_requires=">=3",
    license="AGPL-3.0",
    url="https://github.com/EnexaProject/enexa-tensor-reasoning",
    keywords="inductive reasoning, tensor networks, alternating least squares, markov logic networks"
)
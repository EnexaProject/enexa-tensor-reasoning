from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="tensor-reasoning",
    version="0.0",
    author="Alex Goeßmann",
    long_description=long_description,
    install_requires=[
        "numpy",
        "pandas",
        "pgmpy"
    ]
)
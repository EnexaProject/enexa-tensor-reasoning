from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="tnreason",
    version="0.0",
    author="Alex Goeßmann",
    description="A package for reasoning with tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.23.4",
        "pandas>=1.5.3",
        "pgmpy>=0.1.24"
    ],
    python_requires=">=3.8.5",
    py_modules=["tnreason"],
    package_dir={'':'tnreason'},
)
# tensor-reasoning

Represent Knowledge Graphs as tensors to perform logic calculus and regression.

## Usage

This module can be used for different tasks, as demonstrated in the `examples` folder. 

### Generation of Random Knowledge Graphs

Given a set of logical formulas a Markov Logic Network can be created using the Basis Calculus. One can then sample from the model to generate random data, in this case interpreted as a Random Knowledge Graph.

An example can be found in `examples/generation/generate_accounting_kg.py`

### Learning of logical formulas

Given a Knowledge Graph and positive and negative examples (each a pair of individuals), one can learn a logical formula true on the positive and false on the negative examples. To this end optimization via Alternating Least Squares has been implemented.
Examples can be found in `examples/learning/`.


## Submodules

The repository contains these submodules, each performing a dedicated task.

### Representation

On KG represented in turtle files:

`ttl_to_csv.py` Transform turtle file into a DataFrame containing facts.

`csv_to_cores.py` Transform Fact DataFrame into CoordinateCalculus 

### Logic

Coordinate Calculus: `CoordinateCalculus` main class for coordinate-based calculus of logical formulas.

Basis Calculus: `BasisCalculus` main class for basis-vector-based calculus of logical formulas.

Expression Calculus: Evaluation of expressions given dictionaries of `CoordinateCalculus`/`BasisCalculus` objects.

### Optimization

`generalized_als.py`: Performing the Alternating Least Squares.

### Learning

`expression_learning.py` Class `ExpressionLearner` is a wrapper of the representation and optimization module to learn logical formulas.

### Models

`create_mln.py` Creates a Markov Logic Network base
# tensor-reasoning

Represent Knowledge Graphs as tensors to perform logic calculus and regression.

## Usage

This module can be used for different tasks, as demonstrated in the `examples` folder. 

### Generation of Random Knowledge Graphs

Given a set of logical formulas a Markov Logic Network can be created using the Basis Calculus. One can then sample from the model to generate random data, in this case interpreted as a Random Knowledge Graph.

An example can be found in `examples/generation/generate_accounting_kg.py`.

### Learning of logical formulas

Given a Knowledge Graph and positive and negative examples (each a pair of individuals), one can learn a logical formula true on the positive and false on the negative examples. To this end optimization via Alternating Least Squares has been implemented.
Examples can be found in `examples/learning/`.


## Packages

### Logic

Coordinate Calculus: `CoordinateCalculus` main class for coordinate-based calculus of logical formulas.

Basis Calculus: `BasisCalculus` main class for basis-vector-based calculus of logical formulas.

Expression Calculus: Evaluation of expressions given dictionaries of `CoordinateCalculus`/`BasisCalculus` objects.

### Optimization

`generalized_als.py` Performs the Alternating Least Squares to solve tensor regression problems.

### Learning

`expression_learning.py` Optimizes formulas using Coordinate Calculus and the Alternating Least Squares.

`mln_learning.py` Learns Markov Logic Networks based on data.

### Models

`markov_logic_network.py` Creates a Markov Logic Network using Basis Calculus based on `pgmpy.models.MarkovNetwork`.

### Representation

On KG represented in turtle files:

`ttl_to_csv.py` Transform turtle file into a DataFrame containing facts.

`factdf_to_cores.py` Transforms the fact DataFrame into CoordinateCalculus Cores in the variable-based representation.

`pairdf_to_cores.py` Uses the pair DataFrame to initialize the targetCore.

`sampledf_to_cores.py` Transform sample DataFrame into CoordinateCalculus Cores in the atom-based representation.
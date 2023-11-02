# tensor-reasoning

Represent Knowledge Graphs as tensors to perform logic calculus and regression.

## Usage

Examples can be found in `demonstration`.

## Contents

The repository contains these submodules, each performing a dedicated task.

### Representation

On KG represented in turtle files:

`ttl_to_csv.py` Transform turtle file into a DataFrame containing facts.

`csv_to_cores.py` Transform Fact DataFrame into CoordinateCalculus 

### Logic

Coordinate Calculus: `CoordinateCalculus` main class for coordinate-based calculus of logical formulas.

Basis Calculus: `BasisCalculus` main class for basis-vector-based calculus of logical formulas.

Expression Calculus: Evaluation of expressions given dictionaries of `CoordinateCalculus`/`BasisCalculus` objects.

### ALS

`generalized_als.py`: Performing the Alternating Least Squares.

### Expression Learning

`expression_learning.py` Class `ExpressionLearner` is a wrapper of the representation and optimization module to learn logical formulas.

### Graphical Models

`create_mln.py` Creates a Markov Logic Network base
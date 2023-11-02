# TensorILP

Represent Knowledge Graphs as tensors to perform logic calculus and regression.

# Requirements

The latest versions of these packages have to be installed within the used python interpreter:

-`pgmpy` For generation of synthetic test data from probabilistic graphical models

-`pandas` Usage in generation of tensor cores from data

-`numpy` For performing logical calculus based on numpy routines


# Contents

The repository contains these submodules, each performing a dedicated task.

## Representation

On KG represented in turtle files:

`ttl_to_csv.py` Transform turtle file into a DataFrame containing facts.

`csv_to_cores.py` Transform Fact DataFrame into CoordinateCalculus 

## Logic

Coordinate Calculus: `CoordinateCalculus` main class for coordinate-based calculus of logical formulas.

Basis Calculus: `BasisCalculus` main class for basis-vector-based calculus of logical formulas.

Expression Calculus: Evaluation of expressions given dictionaries of `CoordinateCalculus`/`BasisCalculus` objects.

## ALS

`generalized_als.py`: Performing the Alternating Least Squares.

## Expression Learning

`expression_learning.py` Class `ExpressionLearner` is a wrapper of the representation and optimization module to learn logical formulas.

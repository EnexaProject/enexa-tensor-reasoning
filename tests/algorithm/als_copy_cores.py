from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

networkCores = encoding.create_raw_formula_cores(
    ["imp", "a1", "a2"]
)

from tnreason.algorithms import alternating_least_squares

#copied = als.copy_cores(networkCores, suffix="_fun", exceptionColors=["a1", "a2"])
#print(engine.contract({**networkCores, **copied}, openColors=["(imp_a1_a2)_fun", "(imp_a1_a2)"]).values)

learner = algorithms.ALS(networkCores=networkCores, importanceColors=["a1","a2"], targetCores=encoding.create_formulas_cores(
    {"f1": ["imp","a1","a2"]}
))

print(learner.importanceColors)
print(learner.compute_conOperator(["(imp_a1_a2)"],[2]).values)

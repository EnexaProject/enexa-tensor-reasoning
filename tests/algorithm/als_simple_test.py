from tnreason import engine
from tnreason import contraction

import numpy as np

matrix = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2, 2, 3)),
    colors = ["a1", "a2", "t"]
)

leftVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2)),
    colors = ["a1"]
)

rightVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2)),
    colors = ["a2"]
)

targetVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(3)),
    colors = ["t"]
)

from tnreason.algorithms import alternating_least_squares


optimizer = als.ALS(networkCores={"mat":matrix, "lvec":leftVector, "rvec":rightVector},
                    targetCores={"tar":targetVector},
                    importanceColors=["t"])

cores = optimizer.networkCores

print(cores.keys())
print(contraction.get_contractor("PgmpyVariableEliminator")(cores, openColors=["t"]).contract().values)

print("before",optimizer.compute_residuum())
residua = optimizer.alternating_optimization(["lvec","rvec"], computeResiduum=True)
print("after",optimizer.compute_residuum())
print(residua)

exit()
optimizer.networkCores.pop("lvec")
print(optimizer.networkCores.keys())
conOperator = optimizer.compute_conOperator(["a1"],[2])

conTarget = optimizer.compute_conTarget(["a1"],[2])
print(conOperator.values)
print(conOperator.colors)

print(conTarget.values)
print(conTarget.colors)



#optimizer.alternating_optimization(["lvec","rvec"])
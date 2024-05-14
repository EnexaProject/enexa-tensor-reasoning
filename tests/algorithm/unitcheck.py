from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.algorithms import alternating_least_squares

networkCores = {
    #**encoding.create_formulas_cores({"f1": ["imp", "a1", "a2"]})
    **encoding.create_raw_formula_cores(["imp", "a1", "a2"])
}

targetCores = {
    **als.copy_cores(encoding.create_formulas_cores({"f1": ["imp", "a1", "a2"]}), "_tar", ["a1", "a2"]),
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
}

optimizer = als.ALS(
    networkCores=networkCores,
    targetCores=targetCores,
    importanceColors=["a1", "a2"]
)



conOperator = optimizer.compute_conOperator(updateColors=["(imp_a1_a2)"], updateShape=[2])
print(conOperator.values)
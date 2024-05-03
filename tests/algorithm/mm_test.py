from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.algorithms import moment_matching

networkCores = {
    **encoding.create_formulas_cores({"f1":["imp", "a1", "a2"]})
}

targetCores = {
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(imp_a1_a2)"])
}

matcher = moment_matching.MomentMatcher(networkCores=networkCores,
                                        targetCores=targetCores)


matcher.ones_initialization()
matcher.matching_step("(imp_a1_a2)")
matcher.alternating_matching()
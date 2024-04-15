from tnreason import engine

import numpy as np

matrix = engine.get_core()(
    values=np.random.binomial(10, 0.5, size=(2, 2, 3)),
    colors=["a1", "a2", "t"]
)

leftVector = engine.get_core()(
    values=np.random.binomial(10, 0.5, size=(2)),
    colors=["a1"]
)

rightVector = engine.get_core()(
    values=np.random.binomial(10, 0.5, size=(2)),
    colors=["a2"]
)

from tnreason.network import gibbs

sampler = gibbs.Gibbs(networkCores={"mat": matrix, "lvec": leftVector, "rvec": rightVector},
                      importanceColors=["t"])
positions = sampler.alternating_sampling(["lvec", "rvec"])

print(matrix.values)
print(positions)
print(matrix.values[int(positions[-1][0]), int(positions[-1][1]), :])

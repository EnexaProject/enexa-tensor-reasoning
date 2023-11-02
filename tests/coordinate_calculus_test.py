from tensor_reasoning.logic import coordinate_calculus as cc

import numpy as np

core = cc.CoordinateCore(np.random.normal(size=(3,6,9)),["b","a","c"])

core.reorder_colors(["a","b","c"])
print(core.values.shape,core.colors)

core1 = cc.CoordinateCore(np.random.normal(size=(6,9,3)),["a","c","b"])
core.sum_with(core1)
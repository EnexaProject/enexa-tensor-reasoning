import numpy as np

def review_coreDict(coreDict):
    for coreKey in coreDict:
        print("## Investigating Core {} ##".format(coreKey))
        core = coreDict[coreKey]
        review_core(core)

def review_core(core):
    print("Squared Norm is {}".format(np.linalg.norm(core.values) ** 2))
    print("Shape is {}".format(core.values.shape))
    print("Colors are {}".format(core.colors))
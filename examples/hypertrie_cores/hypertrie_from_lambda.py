from tentris import tentris, Hypertrie

import numpy as np

from tnreason.engine import workload_to_tentris as tc

def ht_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="PolyEncoding"):
    values = Hypertrie(dtype=int, depth=len(inshape + outshape))
    for i in np.ndindex(*inshape):
        values[tuple(i) + tuple(function(*i))] = 1
    return tc.HypertrieCore(values=values, colors=incolors + outcolors, name=name)

def ht_tencoding_from_function(inshape, incolors, function, name="PolyEncoding", dtype=float):
    values = Hypertrie(dtype=dtype, depth=len(inshape))
    for i in np.ndindex(*inshape):
        values[tuple(i)] = float(function(*i))
    return tc.HypertrieCore(values=values, colors=incolors, name=name)



if __name__=="__main__":

    rCore = ht_rencoding_from_function([2,2],[2],["a","b"],["c"], function=lambda i, j : [i])
    print([value for value in rCore.values])
    tCore = ht_tencoding_from_function([2,2],["a","b"],function=lambda i, j : i)
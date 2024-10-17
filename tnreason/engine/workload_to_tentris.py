from tentris import tentris, Hypertrie

from tnreason.engine import workload_to_numpy as cor

from tnreason.engine import subscript_creation as subc

import numpy as np


def ht_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="PolyEncoding"):
    values = Hypertrie(dtype=int, depth=len(inshape + outshape))
    for i in np.ndindex(*inshape):
        values[tuple(i) + tuple(function(*i))] = 1
    return HypertrieCore(values=values, colors=incolors + outcolors, name=name)


def ht_tencoding_from_function(inshape, incolors, function, name="PolyEncoding", dtype=float):
    values = Hypertrie(dtype=dtype, depth=len(inshape))
    for i in np.ndindex(*inshape):
        values[tuple(i)] = float(function(*i))
    return HypertrieCore(values=values, colors=incolors, name=name)


def ht_from_rdf(path, tripleColors=["s", "p", "o"], name="KnowledgeGraphCore"):
    tStore = tentris.TripleStore()
    tStore.load_rdf_data(path)
    return HypertrieCore(values=tStore.hypertrie(), colors=tripleColors, name=name)


class HypertrieCore:
    def __init__(self, values, colors, name=None, shape=[]):
        if isinstance(values, Hypertrie):
            self.values = values
        elif isinstance(values, np.ndarray):
            self.values_from_numpy(array=values)
        else:
            raise ValueError("Values {} to initialize Hypertrie Core not understood!".format(values))
        self.colors = colors
        self.name = name

        self.shape = shape

    def __str__(self):
        return "## Hypertrie Core " + str(self.name) + "\nColors: " + str(self.colors)

    def __getitem__(self, item):
        return self.values[item]

    def get_shape(self):
        shape = np.zeros(self.values.depth)
        for entry in self.values:
            for i in range(len(shape)):
                if entry[0][i] + 1 > shape[i]:
                    shape[i] = entry[0][i] + 1
        self.shape = [int(dim) for dim in shape]

    def values_from_numpy(self, array):
        self.values = Hypertrie(dtype=array.dtype, depth=len(array.shape))
        for index in np.ndindex(array.shape):
            if array[index] != 0:
                self.values[index] = array[index]

    def values_to_numpy(self, calculateShape=False):
        if calculateShape:
            self.get_shape()
        numpyValues = np.zeros(shape=self.shape)
        for entry in self.values:
            numpyValues[tuple(entry[0])] = entry[1]
        return numpyValues

    def to_NumpyTensorCore(self):
        return cor.NumpyCore(values=self.values_to_numpy(), colors=self.colors, name=self.name)


class HypertrieContractor:
    def __init__(self, coreDict, openColors):
        for key in coreDict:
            if not isinstance(coreDict[key], HypertrieCore):
                if isinstance(coreDict[key], cor.NumpyCore):
                    coreDict[key] = HypertrieCore(coreDict[key].values, coreDict[key].colors, coreDict[key].name)
                else:
                    raise ValueError("Hypertrie Contractions works only for Hypertrie or Numpy Cores!")
        self.coreDict = coreDict
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        with tentris.einsumsum(subscript=substring, operands=[self.coreDict[key].values for key in coreOrder]) as e:
            resultValues = Hypertrie(dtype=e.result_dtype, depth=e.result_depth)
            e.try_extend_hypertrie(resultValues)

        return HypertrieCore(values=resultValues,
                             colors=[color for color in colorOrder if color in self.openColors])

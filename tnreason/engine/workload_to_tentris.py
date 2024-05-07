from tentris import tentris, Hypertrie

from tnreason.engine import workload_to_numpy as cor

from tnreason.engine import subscript_creation as subc

import numpy as np


class TentrisCore:
    def __init__(self, values, colors, name=None, inType="Hypertrie"):
        if inType == "Hypertrie":
            self.values = values
        elif inType == "Numpy":
            self.values_from_numpy(array=values)
        elif inType == "rdf":
            self.values_from_rdf(path=values)

        self.colors = colors
        self.name = name

    def values_from_rdf(self, path):
        tStore = tentris.TripleStore()
        tStore.load_rdf_data(path)
        self.values = tStore.hypertrie()

    def values_from_numpy(self, array):
        self.values = Hypertrie(dtype=array.dtype, depth=len(array.shape))
        for index, coordinate in enumerate(np.nditer([array, array.shape])):
            if coordinate != 0:
                self.values[np.unravel_index(index, array.shape)] = coordinate

    def values_to_numpy(self):
        depth = self.values.depth

        ## Get size -> Better to extend to coordinate dictionaries?
        size = np.zeros(depth)
        for entry in self.values:
            for i in range(depth):
                if entry[0][i] > size[i]:
                    size[i] = entry[0][i]

        numpyValues = np.empty(shape=[int(size[i]) for i in range(depth)])
        for entry in self.values:
            numpyValues[entry[0]] = entry[1]
        return numpyValues

    def to_NumpyTensorCore(self):
        return cor.NumpyCore(self.values_to_numpy(), self.colors, self.name)


class TentrisContractor:
    def __init__(self, coreDict, openColors):
        self.tentrisCores = {
            key: TentrisCore(values=coreDict[key].values, colors=coreDict[key].colors, name=coreDict[key].name, inType="Hypertrie") for
            key in coreDict
        }
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.tentrisCores, self.openColors)
        with tentris.einsumsum(subscript=substring, operands=[self.tentrisCores[key].values for key in coreOrder]) as e:
            resultValues = Hypertrie(dtype=e.result_dtype, depth=e.result_depth)
            e.try_extend_hypertrie(resultValues)

        return TentrisCore(values=resultValues,
                           colors=[color for color in colorOrder if color in self.openColors],
                           inType="Hypertrie")

if __name__ == "__main__":
    testCore1 = TentrisCore(np.array([[1.1, 2], [0.12, -1.1]]), colors=["a", "b"])
    testCore2 = TentrisCore(np.array([[1.1, 2], [0.12, -1.1]]), colors=["a", "c"])

    testContractor = TentrisContractor({"1": testCore1, "2":testCore2}, ["b","c"])
    print(testContractor.einsum().to_NumpyTensorCore().values)
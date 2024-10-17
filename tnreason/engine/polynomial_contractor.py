import numpy as np


def poly_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="PolyEncoding"):
    sliceList = [(1, {(incolors + outcolors)[k]: assignment for k, assignment in
                      enumerate(i + tuple([int(entry) for entry in function(*i)]))}) for i in np.ndindex(*inshape)]
    return PolynomialCore(values=SliceValues(slices=sliceList, shape=inshape + outshape), colors=incolors + outcolors,
                          name=name)


def poly_tencoding_from_function(inshape, incolors, function, name="PolyEncoding"):
    sliceList = [(function(*i), {incolors[k]: assignment for k, assignment in enumerate(i)}) for i in
                 np.ndindex(*inshape) if function(*i) != 0]
    return PolynomialCore(values=SliceValues(slices=sliceList, shape=inshape), colors=incolors, name=name)


class SliceValues:
    """
    Storing the polynomial by a list of tuples, each representing a weighted monomial by
        - value: Weight of the monomial
        - positionDict: Dictionary of variables in the polynomial,
            - each key is the name of a categorical variable X
            - its value k specifies the variable to X==k
    Each monomial seen as a tensor is specified by a weighted trivial slice.
    """

    def __init__(self, slices=[(1, dict())], shape=[]):
        self.slices = slices  # List of tuples (value, positionDict)
        self.shape = shape


class PolynomialCore:
    def __init__(self, values, colors, name=None):
        self.colors = colors
        self.name = name

        if isinstance(values, np.ndarray):
            self.ell_zero_initialize_from_numpy(values)
        else:
            self.values = values

    def __str__(self):
        return "## Sliced Core " + str(self.name) + " ##\nValues: " + str(self.values.slices) + "\nColors: " + str(
            self.colors)

    def ell_zero_initialize_from_numpy(self, arr):
        """
        Initialization of the slices by all nonzero coordinates of the array, resulting in ell_zero(arr) many monomials
        """
        slices = []
        for idx in np.ndindex(arr.shape):
            if arr[idx] != 0:
                slices.append(
                    (arr[idx], {self.colors[i]: subindex for i, subindex in enumerate(idx)})
                )
        self.values = SliceValues(slices, shape=arr.shape)

    def contract_with(self, core2):
        newColors = list(set(self.colors) | set(core2.colors))
        newShapes = [0 for color in newColors]
        for i, color in enumerate(self.colors):
            newShapes[newColors.index(color)] = self.values.shape[i]
        for i, color in enumerate(core2.colors):
            newShapes[newColors.index(color)] = core2.values.shape[i]

        return PolynomialCore(
            values=SliceValues(slice_contraction(self.values.slices, core2.values.slices),
                               newShapes),
            colors=newColors,
            name=str(self.name) + "_" + str(core2.name)
        )

    def drop_color(self, color):
        colorDim = self.values.shape.pop(self.colors.index(color))
        newSlices = []
        for (val, pos) in self.values.slices:
            if color in pos:
                pos.pop(color)
                newSlices.append((val, pos))
            else:
                newSlices.append((colorDim * val, pos))
        self.values.slices = newSlices
        self.colors.pop(self.colors.index(color))

    def add_identical_slices(self):
        newSlices = []
        alreadyFound = []
        while len(self.values.slices) != 0:
            val, pos = self.values.slices.pop()
            if pos not in alreadyFound:
                alreadyFound.append(pos)
                for (val2, pos2) in self.values.slices:
                    if pos == pos2:
                        val += val2
                newSlices.append((val, pos))
        self.values.slices = newSlices

    def multiply(self, weight):
        return PolynomialCore(values=SliceValues(
            slices=[(weight * val, pos) for (val, pos) in self.values.slices],
            shape=self.values.shape
        ), colors=self.colors, name=self.name)

    def sum_with(self, sumCore):
        return PolynomialCore(values=SliceValues(
            slices=self.values.slices + sumCore.values.slices,
            shape=self.values.shape
        ), colors=self.colors, name=self.name)

    def normalize(self):
        return self

    def enumerate_slices(self, enumerationColor="j"):
        self.colors = self.colors + [enumerationColor]
        self.values = SliceValues(
            [(entry[0], {**entry[1], enumerationColor: i}) for i, entry in enumerate(self.values.slices)],
            shape=self.values.shape + [len(self.values.slices)])

    def __getitem__(self, item):
        value = 0
        for entry in self.values.slices:
            if agreeing_dicts(entry[1], {color: item[i] for i, color in enumerate(self.colors)}):
                value += entry[0]
        return value

class GenericSliceContractor:

    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key: PolynomialCore(coreDict[key].values, coreDict[key].colors, name=key) for key in
                         coreDict}
        self.openColors = openColors

    def contract(self):
        ## Without optimization -> Can apply optimization from version0
        resultCore = PolynomialCore(SliceValues(slices=[(1, dict())], shape=[]), [], name="Contraction")
        for key in self.coreDict:
            resultCore = resultCore.contract_with(self.coreDict[key])
        for color in list(resultCore.colors):
            if color not in self.openColors:
                resultCore.drop_color(color)
        return resultCore


def slice_contraction(slices1, slices2):
    slices = []
    for (val1, pos1) in slices1:
        for (val2, pos2) in slices2:
            if agreeing_dicts(pos1, pos2):
                slices.append((val1 * val2, {**pos1, **pos2}))
    return slices


def agreeing_dicts(pos1, pos2):
    for key in pos1:
        if key in pos2:
            if pos1[key] != pos2[key]:
                return False
    return True

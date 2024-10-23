import numpy as np


def poly_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="PolyEncoding"):
    sliceList = [(1, {(incolors + outcolors)[k]: assignment for k, assignment in
                      enumerate(i + tuple([int(entry) for entry in function(*i)]))}) for i in np.ndindex(*inshape)]
    return PolynomialCore(values=sliceList, shape=inshape + outshape, colors=incolors + outcolors,
                          name=name)


def poly_tencoding_from_function(inshape, incolors, function, name="PolyEncoding"):
    sliceList = [(function(*i), {incolors[k]: assignment for k, assignment in enumerate(i)}) for i in
                 np.ndindex(*inshape) if function(*i) != 0]
    return PolynomialCore(values=sliceList, shape=inshape, colors=incolors, name=name)


class PolynomialCore:
    """
    :values: Storing the polynomial by a list of tuples, each representing a weighted monomial by
        - value: Weight of the monomial
        - positionDict: Dictionary of variables in the polynomial,
            - each key is the name of a categorical variable X
            - its value k specifies the variable to X==k
    Each monomial seen as a tensor is specified by a weighted trivial slice.
    """

    def __init__(self, values, colors, name=None, shape=None):
        self.colors = colors
        self.name = name

        if shape is not None:
            self.shape = shape

        if isinstance(values, np.ndarray):
            self.ell_zero_initialize_from_numpy(values)
        else:
            self.values = values

    def __str__(self):
        return "## Polynomial Core " + str(self.name) + " ##\nValues: " + str(self.values) + "\nColors: " + str(
            self.colors)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        value = 0
        for entry in self.values:
            if agreeing_dicts(entry[1], {color: item[i] for i, color in enumerate(self.colors)}):
                value += entry[0]
        return value

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
        self.values = slices
        self.shape = arr.shape

    def contract_with(self, core2):
        newColors = list(set(self.colors) | set(core2.colors))
        newShapes = [0 for color in newColors]
        for i, color in enumerate(self.colors):
            newShapes[newColors.index(color)] = self.shape[i]
        for i, color in enumerate(core2.colors):
            newShapes[newColors.index(color)] = core2.shape[i]

        return PolynomialCore(
            values=slice_contraction(self.values, core2.values),
            shape=newShapes,
            colors=newColors,
            name=str(self.name) + "_" + str(core2.name)
        )

    def reduce_colors(self, newColors):
        newValues = []
        for j in range(len(self.values)):
            newValues.append((np.prod([self.shape[k] for k, col in enumerate(self.colors) if
                                       col not in self.values[j][1] and col not in newColors]) * self.values[j][0],
                              {key: self.values[j][1][key] for key in self.values[j][1] if key in newColors}))
        self.values = newValues
        self.shape = [self.shape[k] for k, col in enumerate(self.colors) if col in newColors]
        self.colors = newColors


    def add_identical_slices(self):
        newSlices = []
        alreadyFound = []
        while len(self.values) != 0:
            val, pos = self.values.pop()
            if pos not in alreadyFound:
                alreadyFound.append(pos)
                for (val2, pos2) in self.values:
                    if pos == pos2:
                        val += val2
                newSlices.append((val, pos))
        self.values = newSlices

    def multiply(self, weight):
        return PolynomialCore(values=[(weight * val, pos) for (val, pos) in self.values],
                              shape=self.shape, colors=self.colors, name=self.name)

    def sum_with(self, sumCore):
        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: sumCore.shape[i] for i, color in enumerate(sumCore.colors)}}
        return PolynomialCore(values=self.values + sumCore.values,
                              shape=list(colorsShapeDict.values()), colors=list(colorsShapeDict.keys()),
                              name=self.name)

    def enumerate_slices(self, enumerationColor="j"):
        self.colors = self.colors + [enumerationColor]
        self.values = [(entry[0], {**entry[1], enumerationColor: i}) for i, entry in enumerate(self.values)]
        self.shape = self.shape + [len(self.values)]

    def reorder_colors(self, newColors):
        if set(self.colors) == set(newColors):
            self.colors = newColors
        else:
            raise ValueError("Reordering of Colors in Core {} not possible, since different!".format(self.name))


class PolynomialContractor:

    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = coreDict
        self.openColors = openColors

    def contract(self):
        ## Without optimization -> Can apply optimization from version0
        if len(self.coreDict) == 0:
            return PolynomialCore(values=[(1, dict())], shape=[], colors=self.openColors, name="Contraction")
        name, resultCore = self.coreDict.popitem()
        for key in self.coreDict:
            resultCore = resultCore.contract_with(self.coreDict[key])
        resultCore.reduce_colors(self.openColors)
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

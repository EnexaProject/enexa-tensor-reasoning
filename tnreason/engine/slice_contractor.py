import numpy as np


class SliceCore:
    """
    Core for representing Tensors with leg dimensions 2
    Tailored to slice sparsity (a generalization to l_0 sparsity allowing also trivial vectors instead of basis vectors as factors in the tensor products to be summed)
    Differs from NumpyCore by
        * self.values: A list of generalized basis tensors, each specified by a tuple of
            - value: weight of slice (nonzero) in float or int
            - set of colors, which are false
            - set of colors, which are true
    """

    def __init__(self, values, colors, name=None):
        self.colors = colors
        if isinstance(values, np.ndarray):
            self.ell_zero_initialize_from_numpy(values)
        else: ## Should then be directly in tuple format
            self.values = values

        self.name = name

    def ell_zero_initialize_from_numpy(self, arr):
        self.values = []
        for idx in np.ndindex(arr.shape):
            if arr[idx] != 0:
                self.values.append((arr[idx], set([self.colors[i] for i, x in enumerate(idx) if x == 0]),
                                    set([self.colors[i] for i, x in enumerate(idx) if x != 0])))
    def __str__(self):
        return "## Sliced Core " + str(self.name) + " ##\nValues: " + str(self.values) + "\nColors: " + str(self.colors)


class SliceContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key : SliceCore(coreDict[key].values, coreDict[key].colors, key) for key in coreDict}
        self.openColors = openColors

    def contract(self):
        allColors = set()
        for key in self.coreDict:
            allColors = allColors | set(self.coreDict[key].colors)

        values = [(1, set(), set())]
        for key in self.coreDict:
            values = slice_contraction(values, self.coreDict[key].values)
        values = reduce_colors(values, allColors - set(self.openColors))
        return SliceCore(values, self.openColors)


def reduce_colors(values, reduceColors):
    reducedValues = []
    for val, neg, pos in values:
        for reduceColor in reduceColors:
            if reduceColor not in pos and reduceColor not in neg:
                val = 2 * val  # Color dimension always 2
            elif reduceColor in pos:
                pos.remove(reduceColor)
            elif reduceColor in neg:
                neg.remove(reduceColor)
            else:
                print(reduceColor, pos, neg, reduceColor in pos, reduceColor in neg)
        reducedValues.append((val, pos, neg))
    return reducedValues


def slice_contraction(values1, values2):
    slices = []
    for slice1 in values1:
        for slice2 in values2:
            combined = combine_slices(slice1, slice2)
            if combined[0] != 0:
                slices.append(combined)
    return slices


def combine_slices(slice1, slice2):
    val1, neg1, pos1 = slice1
    val2, neg2, pos2 = slice2

    neg = neg1 | neg2
    pos = pos1 | pos2
    if len(pos & neg) > 0:  ## Check whether the slice zero
        return (0, [], [])
    else:
        return (val1 * val2, neg, pos)


if __name__ == "__main__":
    values1 = [
        (1.1, {"a", "b"}, {"c"}),
        (0.9, set(), {"d"})
    ]
    core1 = SliceCore(values1, ["a", "b", "c"])

    values2 = [
        (1.1, {"b"}, {"a", "c"}),
        (2, {"a"}, set())
    ]
    core2 = SliceCore(values2, ["a", "b", "c"])

    contracted = SliceContractor(coreDict={
        "c1": core1,
        "c2": core2
    }, openColors=["a", "b"]).contract()

    print(contracted)

    # print([e for e in slice_contraction(values1, values2)])

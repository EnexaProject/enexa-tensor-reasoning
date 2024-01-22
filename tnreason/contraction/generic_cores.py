import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']


class TensorCoreBase:
    def __init__(self, values, colors, name=None):
        if len(colors) != len(values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(colors) != len(set(colors)):
            raise ValueError("There are duplicate colors in Core {}!".format(name))

        self.values = values
        self.colors = colors
        self.name = name


class NumpyTensorCore(TensorCoreBase):

    def reduced_contraction(self, core1, reductionColors=[]):
        colorDict = {color: alphabet[i] for i, color in enumerate(np.unique(self.colors + core1.colors))}

        core0String = "".join([colorDict[color] for color in self.colors])
        core1String = "".join([colorDict[color] for color in core1.colors])
        premiseString = ",".join([core0String, core1String])

        outColors = [color for color in np.unique(self.colors + core1.colors) if color not in reductionColors]
        headString = "".join([colorDict[color] for color in outColors])

        contractionString = "->".join([premiseString, headString])

        outValues = np.einsum(contractionString, self.values, core1.values)

        return NumpyTensorCore(outValues, outColors, [self.name, "con", core1.name])

    def reduce_colors(self, reductionColors):
        colorDict = {color: alphabet[i] for i, color in enumerate(self.colors)}

        contractionString = "->".join([
            "".join([colorDict[color] for color in self.colors]),
            "".join([colorDict[color] for color in self.colors if color not in reductionColors])
        ])

        return NumpyTensorCore(np.einsum(contractionString, self.values),
                               [color for color in self.colors if color not in reductionColors], self.name)

    def get_values_as_array(self):
        return self.values


def change_type(cCore, targetType="NumpyTensorCore"):
    if targetType == "NumpyTensorCore":
        return NumpyTensorCore(cCore.values, cCore.colors, cCore.name)
    else:
        raise TypeError("Type {} not understood!".format(targetType))

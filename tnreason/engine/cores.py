import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']


class TensorCoreBase:
    def __init__(self, values, colors, name=None):
        if len(colors) != len(values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(colors) != len(set(colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

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

    def reduce_color(self, reductionColor):
        self.reduce_colors([reductionColor])

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

    def clone(self):
        return NumpyTensorCore(self.values.copy(), self.colors.copy(), self.name)  # ! Shallow Copies?

    ## For Sampling
    def normalize(self):
        return NumpyTensorCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS
    def reorder_colors(self, newColors):
        oldColors = self.colors.copy()
        oldValues = np.copy(self.values)

        colorDict = {oldColors[i]: alphabet[i] for i in range(len(oldColors))}
        old_string = "".join([colorDict[color] for color in oldColors])
        new_string = "".join([colorDict[color] for color in newColors])
        contraction_string = "->".join([old_string, new_string])

        newValues = np.einsum(contraction_string, oldValues)

        self.values = newValues
        self.colors = newColors


    def sum_with(self, sumCore):
        if set(self.colors)!=set(sumCore.colors):
            print(self.colors, sumCore.colors)
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, sumCore.name))
        else:
            self.reduce_colors(sumCore.colors)
            return NumpyTensorCore(self.values + sumCore.values, self.colors, self.name)

    def multiply(self, weight):
        return NumpyTensorCore(weight * self.values, self.colors, self.name)

def change_type(cCore, targetType="NumpyTensorCore"):
    if targetType == "NumpyTensorCore":
        return NumpyTensorCore(cCore.values, cCore.colors, cCore.name)
    else:
        raise TypeError("Type {} not understood!".format(targetType))

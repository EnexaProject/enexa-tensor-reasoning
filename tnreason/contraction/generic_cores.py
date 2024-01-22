import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']

class TensorCoreBase:
    def __init__(self, values, colors, name=None):
        if len(colors) != len(values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))

        self.values = values
        self.colors = colors
        self.name = name

class NumpyTensorCore(TensorCoreBase):

    def contract_with(self, core1, reductionColors=[]):
        colorDict = {color: alphabet[i] for i, color in enumerate(np.unique(self.colors + core1.colors))}

        core0String = "".join([colorDict[color] for color in self.colors])
        core1String = "".join([colorDict[color] for color in core1.colors])
        premiseString = ",".join([core0String,core1String])

        outColors = [color for color in np.unique(self.colors + core1.colors) if color not in reductionColors]
        headString = "".join([colorDict[color] for color in outColors])

        contractionString = "->".join([premiseString,headString])

        outValues = np.einsum(contractionString, self.values, core1.values)

        return NumpyTensorCore(outValues, outColors, [self.name, "con", core1.name])

    def get_values_as_array(self):
        return self.values



def initialize_from_cCore(cCore):
    return NumpyTensorCore(cCore.values, cCore.colors, cCore.name)




if __name__ == "__main__":


    values = np.random.binomial(1,0,size=(3,2))
    print(values.shape)
    tc = NumpyTensorCore(np.random.binomial(1,0,size=(3,2)), ["a","b"])
    tc2 = NumpyTensorCore(np.random.binomial(1,0,size=(3,2)), ["a","b"])

    contracted = tc.contract_with(tc2, ["a"])
    print(contracted.values.shape)
    print(contracted.colors)

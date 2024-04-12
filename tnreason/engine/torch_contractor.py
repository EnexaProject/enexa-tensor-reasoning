import torch as tor

from tnreason.engine import cores as cor

from tnreason.engine import subscript_creation as subc


class TorchCore:
    def __init__(self, values, colors, name=None, inType="Numpy"):
        if inType == "Numpy":
            self.values = tor.from_numpy(values)
        else:
            self.values = values
        self.colors = colors
        self.name = name

    def to_NumpyTensorCore(self):
        return cor.NumpyCore(self.values.numpy(), self.colors, self.name)


class TorchContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.torchCores = {
            key: TorchCore(values=coreDict[key].values, colors=coreDict[key].colors, name=coreDict[key].name) for
            key in coreDict
        }
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_substring(self.torchCores, self.openColors)
        return TorchCore(values=tor.einsum(substring,
                                           *[self.torchCores[key].values for key in coreOrder]
                                           ), colors=[color for color in colorOrder if color in self.openColors],
                         inType="Other")


if __name__ == "__main__":
    import tnreason.tensor.formula_tensors as ft

    cores = ft.FormulaTensor(["c", "and", ["not", ["a", "or", "b"]]]).get_cores()
    print(cores)

    contractor = TorchContractor(cores, openColors=["a", "b"])

    result = contractor.einsum()
    print(result.values)
    print(result.colors)

    print(result.to_NumpyTensorCore())

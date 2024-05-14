import torch as tor

from tnreason.engine import workload_to_numpy as cor
from tnreason.engine import subscript_creation as subc


class TorchCore:
    def __init__(self, values, colors, name=None, inType="NumpyTensorCore"):
        if inType == "NumpyTensorCore":
            self.values = tor.from_numpy(values)
        else:
            self.values = values
        self.colors = colors
        self.name = name

    def to_NumpyTensorCore(self):
        return cor.NumpyCore(self.values.numpy(), self.colors, self.name)


class TorchContractor:
    def __init__(self, coreDict, openColors):
        self.torchCores = {
            key: TorchCore(values=coreDict[key].values, colors=coreDict[key].colors, name=coreDict[key].name) for
            key in coreDict
        }
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.torchCores, self.openColors)
        return TorchCore(values=tor.einsum(substring,
                                           *[self.torchCores[key].values for key in coreOrder]
                                           ), colors=[color for color in colorOrder if color in self.openColors],
                         inType="Other")
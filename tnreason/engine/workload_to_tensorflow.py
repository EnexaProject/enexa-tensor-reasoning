import tensorflow as tf

from tnreason.engine import workload_to_numpy as cor
from tnreason.engine import subscript_creation as subc


class TensorFlowCore:
    def __init__(self, values, colors, name=None):
        self.values = tf.convert_to_tensor(values)
        self.colors = colors
        self.name = name

    def to_NumpyTensorCore(self):
        return cor.NumpyCore(self.values.numpy(), self.colors, self.name)


class TensorFlowContractor:
    def __init__(self, coreDict, openColors):
        self.tensorFlowCores = {
            key: TensorFlowCore(values=coreDict[key].values, colors=coreDict[key].colors, name=coreDict[key].name) for
            key in coreDict
        }
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.tensorFlowCores, self.openColors)
        return TensorFlowCore(values=tf.einsum(substring,
                                               *[self.tensorFlowCores[key].values for key in coreOrder]
                                               ), colors=[color for color in colorOrder if color in self.openColors])

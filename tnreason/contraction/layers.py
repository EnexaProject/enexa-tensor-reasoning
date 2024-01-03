from tnreason.logic import coordinate_calculus as cc
import numpy as np


class Layer:
    def __init__(self, coresDict={}, signsDict={}):
        self.signsDict = signsDict
        self.coresDict = coresDict

        self.colors = []
        for key in self.coresDict:
            self.colors = self.colors + self.coresDict[key].colors

    def compute_size(self):
        self.size = 0
        for key in self.coresDict:
            self.size += np.prod(self.coresDict[key].values.shape)
        return self.size

    def reduce_color(self, color):
        for key in self.coresDict:
            if color in self.coresDict[key].colors:
                self.coresDict[key] = self.coresDict[key].reduce_color(color)
        return self

    def switch_sign(self):
        for key in self.signsDict:
            self.signsDict[key] = -1 * self.signsDict[key]


def contract(layer0, layer1):
    contractedCoresDict = {}
    contractedSignsDict = {}
    for key0 in layer0.coresDict:
        for key1 in layer1.coresDict:
            contractedSignsDict[key0 + "_" + key1] = layer0.signsDict[key0] * layer1.signsDict[key1]
            contractedCoresDict[key0 + "_" + key1] = layer0.coresDict[key0].compute_and(layer1.coresDict[key1])
    return Layer(contractedCoresDict, contractedSignsDict)


if __name__ == "__main__":
    l = Layer({"a": cc.CoordinateCore(core_values=np.array([2]), core_colors=["b1"])}, {"a": 1})

    print(l.compute_size())
    print(l.reduce_color("b1"))
    print(contract(l, l).coresDict)

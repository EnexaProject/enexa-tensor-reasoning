from tnreason import engine
from tnreason import encoding

import numpy as np

answerCoreSuffix = "_answerCore"


class BinaryContractionPropagator:
    """
    Generalizing the Constraint Propagation towards propagated generic tensor cores (whereas ConstraintPropagator propagates vector cores)
        Avoid multiple elements in queue -> Usage of sets
    """

    def __init__(self, subNetworks, queryVariables, affectionDict=None):
        self.subNetworks = subNetworks

        self.queryVariables = queryVariables

        self.initialize_answerCores(queryVariables, find_all_color_shapes(subNetworks))

        if affectionDict is None:
            affectionDict = find_affection_by_colors(subNetworks, queryVariables)
        self.affectionDict = affectionDict

    def initialize_answerCores(self, queryVariables, colorShapes):
        if not all([all([color in colorShapes for color in queryVariables[key]]) for key in queryVariables]):
            raise ValueError("A query color is not appearing in the subnetworks!")

        self.answerCores = {key + answerCoreSuffix: engine.get_core()(
            values=np.ones(shape=[colorShapes[color] for color in queryVariables[key]]),
            colors=queryVariables[key]) for key in queryVariables}

    def initialize_queue(self):
        self.networkQueue = set()
        for queryKey in self.affectionDict:
            for netKey in self.affectionDict[queryKey]:
                self.networkQueue.add(netKey)

    def propagate(self):
        while not len(self.networkQueue) == 0:
            networkKey = self.networkQueue.pop()
            for queryKey in [key for key in self.queryVariables if networkKey in self.affectionDict[key]]:
                changed = self.update_answerCore(queryKey, networkKey)
                if changed:
                    for netKey in self.affectionDict[queryKey]:
                        self.networkQueue.add(netKey)

    def update_answerCore(self, queryKey, networkKey):
        contracted = engine.contract(coreDict={**self.subNetworks[networkKey],
                                               **{key + answerCoreSuffix: self.answerCores[key + answerCoreSuffix] for
                                                  key in self.affectionDict if
                                                  networkKey in self.affectionDict[key]}
                                               },
                                     openColors=self.queryVariables[queryKey])
        contractedShape = contracted.values.shape
        changed = False
        for i, coordinate in enumerate(contracted.values.flatten()):
            if self.answerCores[queryKey + answerCoreSuffix].values[np.unravel_index(i, contractedShape)] == 1 and \
                    contracted.values[np.unravel_index(i, contractedShape)] == 0:
                self.answerCores[queryKey + answerCoreSuffix].values[np.unravel_index(i, contractedShape)] = 0
                changed = True

        return changed


def find_affection_by_colors(subNetworks, queryVariables):
    subColors = {key: find_color_shapes(subNetworks[key]) for key in subNetworks}
    return {queryKey: [key for key in subNetworks if
                       any([color in subColors[key] for color in queryVariables[queryKey]])] for queryKey in
            queryVariables}


def find_color_shapes(network):
    colorShapes = {}
    for coreKey in network:
        for i, color in enumerate(network[coreKey].colors):
            colorShapes.update({color: network[coreKey].values.shape[i]})
    return colorShapes


def find_all_color_shapes(subNetworks):
    colorShapes = {}
    for netKey in subNetworks:
        colorShapes.update(find_color_shapes(subNetworks[netKey]))
    return colorShapes


if __name__ == "__main__":
    subNetworks = {"net1": encoding.create_formulas_cores({"f1": ["imp", "a", "b"]}),
                   "net2": encoding.create_formulas_cores({"f2": ["a"],
                                                           "f3": ["imp", "a", "c"]})}

    lCon = BinaryContractionPropagator(subNetworks, queryVariables={"q1": ["a"],
                                                                    "q3": ["c"],
                                                                    "q2": ["b", "c"]})

    lCon.initialize_queue()

    lCon.propagate()
    print(lCon.answerCores["q2" + answerCoreSuffix].values)

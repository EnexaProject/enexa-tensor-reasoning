from tnreason import encoding
from tnreason import engine

import numpy as np

messageCoreSuffix = "_messageCore"

class MPGibbs:
    def __init__(self, coresDict):
        self.networkCores = coresDict
        self.create_affectionDict()
        self.ones_initialization()

    def create_affectionDict(self):
        self.colorAffectionDict = {}
        self.colorDimDict = {}
        for coreKey in self.networkCores:
            for color in self.networkCores[coreKey].colors:
                if color in self.colorAffectionDict:
                    self.colorAffectionDict[color].append(coreKey)
                else:
                    self.colorAffectionDict[color] = [coreKey]
                    self.colorDimDict[color] = self.networkCores[coreKey].values.shape[
                        self.networkCores[coreKey].colors.index(color)]

    def ones_initialization(self):
        self.messageCores = {}
        for color in self.colorDimDict:
            self.messageCores[color + messageCoreSuffix] = encoding.create_trivial_core(color + messageCoreSuffix,
                                                                                        self.colorDimDict[color],
                                                                                        [color])

    def alternating_sampling(self, basisColors, sweepNum=10, sampleColors=None):
        if sampleColors is None:
            # When not specified otherwise, all colors are sampled
            sampleColors = self.colorDimDict.keys()
        assignments = {color: [] for color in basisColors}
        for sweep in range(sweepNum):
            for color in sampleColors:
                if color in basisColors:
                    assignments[color].append(self.resample(color, basis=True))
                else:
                    self.resample(color, basis=False)
        return assignments

    def resample(self, updateColor, basis=True):
        affectedCores = {coreKey: self.networkCores[coreKey] for coreKey in self.colorAffectionDict[updateColor]}
        affectedColors = []
        for coreKey in affectedCores:
            for color in affectedCores[coreKey].colors:
                if color not in affectedColors and color != updateColor:
                    affectedColors.append(color)

        distribution = engine.contract(
            coreDict={**affectedCores,
                      **{color + messageCoreSuffix: self.messageCores[color + messageCoreSuffix] for color in
                         affectedColors}},
            openColors=[updateColor]
        )
        if basis:
            if np.sum(distribution.values) == 0:
                print(f"Sampling {updateColor} on a vanishing distribution.")
                normedDistribution = 1 / distribution.values.shape[0] * np.ones(shape=distribution.values.shape)
            else:
                normedDistribution = distribution.values / np.sum(distribution.values)
            randomAssignment = np.where(np.random.multinomial(1, normedDistribution) == 1)[0][0]
            self.messageCores[updateColor + messageCoreSuffix] = encoding.create_basis_core(
                updateColor + messageCoreSuffix,
                self.colorDimDict[updateColor],
                [updateColor], randomAssignment)
            return randomAssignment
        else:
            self.messageCores[updateColor + messageCoreSuffix] = distribution


class SuperGibbs:
    def __init__(self, importanceDict):
        """
        importanceDict: keys specify the sampler and values are pairs of importance weight and the coresDict
        """
        self.samplers = {key: MPGibbs(importanceDict[key][1]) for key in importanceDict}

    def sample_hidden(self, visibleColors, sweepNum=1):
        # Sample the hidden colors of each sampler
        for key in self.samplers:
            self.samplers[key].alternating_sampling(
                basisColors = [],
                sweepNum=sweepNum,
                sampleColors = [color for color in self.samplers[key].colorDimDict if color not in visibleColors]
            )

    def sample_visible(self, visibleColors, sweepNum=1):
        ## Problem: How to compare the contributions -> Would need partition functions in the importanceDict!
           pass



if __name__ == "__main__":
    core1 = engine.get_core()(
        values=np.random.random(size=(5, 4, 2)),
        colors=["a", "b", "d"]
    )
    core2 = engine.get_core()(
        values=np.random.random(size=(5, 4)),
        colors=["a", "c"]
    )

    gibbser = MPGibbs({"c1": core1, "c2": core2})
    print(gibbser.alternating_sampling(["a", "b", "c"]))

    from tnreason import knowledge

    hybridKB = knowledge.HybridKnowledgeBase(
        facts={"f1": ["imp", "p", "q"],
               "f2": ["not", "q"]}
    )
    gibbser = MPGibbs(hybridKB.create_cores())
    print(gibbser.alternating_sampling(["p", "q"]))


    superGibbs = SuperGibbs({"positive" : (1,hybridKB.create_cores())})
    superGibbs.sample_hidden(["p"])
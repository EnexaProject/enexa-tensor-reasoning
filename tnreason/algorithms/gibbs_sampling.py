from tnreason import engine
from tnreason import encoding

import numpy as np


class Gibbs:
    """
    Implements Gibbs sampling with annealing support
        * networkCores: Main tensor network forming a Markov Network
        * importanceList, importanceColors: Reshaping the probability of networkCores, as in ALS
        * exponentiated: Whether raw probability (which can get negative) is exponentiated to be posified
    """

    def __init__(self, networkCores, importanceColors=[], importanceList=[({}, 1)], exponentiated=False,
                 contractionMethod=engine.defaultContractionMethod):
        self.networkCores = networkCores

        self.importanceColors = importanceColors
        self.importanceList = importanceList
        self.contractionMethod = contractionMethod

        self.exponentiated = exponentiated

    def ones_initialization(self, updateKeys, shapesDict, colorsDict=None):
        """
        Initialize all cores to be sampled with trivial cores
        """
        if colorsDict is None:
            colorsDict = {key: key for key in updateKeys}
        for updateKey in updateKeys:
            if updateKey in self.networkCores.keys():
                print("Warning: Existing core {} has been reinitialized in Gibbs!".format(updateKey))
            self.networkCores[updateKey] = encoding.create_trivial_core(updateKey, shapesDict[updateKey],
                                                                        colorsDict[updateKey])

    def annealed_sample(self, updateKeys, annealingPattern=[(10, 1)]):
        """
        Samples based on an annealing pattern.
        """
        for rep, (sweepNum, temperature) in enumerate(annealingPattern):
            sample = self.gibbs_sample(updateKeys=updateKeys, sweepNum=sweepNum, temperature=temperature)
        return sample

    def gibbs_sample(self, updateKeys, sweepNum=10, temperature=1):
        """
        Sample created to constant temperature.
        """
        positions = np.empty(shape=(sweepNum, len(updateKeys)))
        for sweep in range(sweepNum):
            for i, updateKey in enumerate(updateKeys):
                positions[sweep, i] = self.sample_core(updateKey, temperature=temperature)

        return {updateKeys[i]: positions[-1, i] for i in range(len(updateKeys))}

    def sample_core(self, updateKey, temperature):
        """
        Update a color core based on tempered marginal probability
        """
        ## Trivialize the core to be updated (serving as a placeholder)
        tbUpdated = self.networkCores.pop(updateKey)
        self.networkCores[updateKey] = encoding.create_trivial_core(updateKey, tbUpdated.values.shape, tbUpdated.colors)

        updateColors = tbUpdated.colors
        updateShape = tbUpdated.values.shape

        updateDistribution = engine.contract({**self.networkCores, **self.importanceList[0][0],
                                              "trivialCore": encoding.create_trivial_core("trivialCore", updateShape,
                                                                                          updateColors)},
                                             openColors=updateColors,
                                             method=self.contractionMethod).multiply(self.importanceList[0][1])
        updateDistribution.reorder_colors(updateColors)
        for importanceCores, weight in self.importanceList[1:]:
            updateDistribution = updateDistribution.sum_with(
                engine.contract({**self.networkCores, **importanceCores,
                                 "trivialCore": encoding.create_trivial_core("trivialCore", updateShape,
                                                                             updateColors)},
                                openColors=updateColors,
                                method=self.contractionMethod).multiply(weight))

        flattened = updateDistribution.values.flatten()
        if self.exponentiated:  # Used to posify the unnormalized probability
            flattened = np.exp(flattened)

        if np.sum(flattened) <= 0:  # Sample from uniform distribution, when vanishing unnormalized probability
            localProb = np.ones(shape=flattened.shape) / flattened.shape[0]
            print("Vanishing prob for update of core {}!".format(updateKey))
        else:
            if temperature != 1:
                localProb = flattened ** temperature / (np.sum(flattened ** temperature))
            else:
                localProb = flattened / np.sum(flattened)

        # Draw a basis vector from the distribution
        pos = np.where(np.random.multinomial(1, localProb) == 1)[0][0]
        self.networkCores[updateKey] = encoding.create_basis_core(updateKey, updateShape, updateColors, pos)
        return pos

from tnreason import engine

import numpy as np


def create_affectionDict(energyDict, colors):
    return {color: [energyKey for energyKey in energyDict if
                    any([color in energyDict[energyKey][0][coreKey].colors for coreKey in energyDict[energyKey][0]])]
            for color in colors}


class EnergyMeanField:
    def __init__(self, energyDict, colors=[], dimDict={}, partitionColorDict=None):
        self.energyDict = energyDict
        self.colors = colors

        self.affectionDict = create_affectionDict(energyDict, colors)
        self.dimDict = dimDict

        # Only distinction to Gibbs: MeanCores instead of samples turned into cores
        self.partitionColorDict = partitionColorDict
        self.meanCores = {parKey: engine.create_trivial_core(parKey, [self.dimDict[color] for color in
                                                                      self.partitionColorDict[parKey]],
                                                             partitionColorDict[parKey]).multiply(
            1 / np.prod([self.dimDict[color] for color in self.partitionColorDict[parKey]])) for parKey
            in partitionColorDict}

    def update_meanCore(self, upKey, temperature=1):

        oldMean = self.meanCores[upKey].clone()

        restMeanCores = {secKey: self.meanCores[secKey] for secKey in self.meanCores if secKey != upKey}
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in self.partitionColorDict[upKey]]))
        contracted = engine.contract(
            {**restMeanCores,
             **self.energyDict[affectedEnergyKeys[0]][0]
             }, openColors=self.partitionColorDict[upKey], dimDict=self.dimDict).multiply(
            self.energyDict[affectedEnergyKeys[0]][1])
        for energyKey in affectedEnergyKeys[1:]:
            contracted.sum_with(
                engine.contract({**restMeanCores,
                                 **self.energyDict[energyKey][0]},
                                openColors=self.partitionColorDict[upKey], dimDict=self.dimDict).multiply(
                    self.energyDict[energyKey][1])
            )

        contracted = contracted.multiply(1 / temperature)
        self.meanCores[upKey] = engine.get_core("NumpyTensorCore")(values=contracted.values,
                                                                   colors=contracted.colors).exponentiate().normalize()

        angle = engine.contract({"old": oldMean, "new": self.meanCores[upKey]}, openColors=[])
        return angle.values

    def anneal(self, temperatureList):
        angles = np.empty(shape=(len(temperatureList), len(self.partitionColorDict)))
        for i, temperature in enumerate(temperatureList):
            for j, upKey in enumerate(self.partitionColorDict):
                angles[i, j] = self.update_meanCore(upKey, temperature=temperature)
        return angles

    def draw_sample(self):
        """
        Draws a sample from the approximating independent distribution
        """
        sample = {}
        for coreKey in self.meanCores:
            sample.update(self.meanCores[coreKey].draw_sample(temperature=1))
        return sample


class EnergyGibbs:
    def __init__(self, energyDict, colors=[], dimDict={}):
        self.energyDict = energyDict
        self.colors = colors

        self.affectionDict = create_affectionDict(energyDict, colors)

        self.dimDict = dimDict
        self.sample = {}

    def initialize_sample_uniform(self):
        for color in self.colors:
            self.sample.update(
                engine.create_trivial_core(color + "_probCore", self.dimDict[color], [color]).draw_sample())

    def calculate_energy(self, upColors):
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in upColors]))
        sampleCores = {
            color + "_sampleCore": engine.create_basis_core(color + "_sampleCore", [self.dimDict[color]], [color],
                                                            (self.sample[color])) for
            color in self.sample if color not in upColors}
        contractedEnergy = engine.contract(coreDict={**self.energyDict[affectedEnergyKeys[0]][0], **sampleCores},
                                           openColors=upColors, dimDict=self.dimDict).multiply(
            self.energyDict[affectedEnergyKeys[0]][1])
        for energyKey in affectedEnergyKeys[1:]:
            contractedEnergy = contractedEnergy.sum_with(
                engine.contract({**self.energyDict[energyKey][0], **sampleCores}, openColors=upColors,
                                dimDict=self.dimDict).multiply(
                    self.energyDict[energyKey][1]))
        return contractedEnergy

    def sample_colors(self, colors, temperature=1):
        energy = self.calculate_energy(colors)
        self.sample.update(energy.draw_sample(asEnergy=True, temperature=temperature))

    def annealed_sample(self, temperatureList=[1], partitionColorDict=None):
        if partitionColorDict is None:
            partitionColorDict = {color: [color] for color in self.colors}
        for i, temperature in enumerate(temperatureList):
            for j, upKey in enumerate(partitionColorDict):
                self.sample_colors(partitionColorDict[upKey], temperature=temperature)

from tnreason import engine

import numpy as np

## Energy-based
gibbsMethodString = "gibbsSample"
meanFieldMethodString = "meanFieldSample"
energyMaximumMethodString = "exactEnergyMax"
energyOptimizationMethods = [gibbsMethodString, meanFieldMethodString, energyMaximumMethodString]


def optimize_energy(energyDict, colors=[], dimDict={}, method=gibbsMethodString,
                    temperatureList=[1 for i in range(10)], coreType=None, contractionMethod=None):
    if method == gibbsMethodString:
        sampler = EnergyGibbs(energyDict=energyDict, colors=colors, dimDict=dimDict, coreType=coreType,
                              contractionMethod=contractionMethod)
        sampler.annealed_sample(temperatureList)
        return sampler.sample
    elif method == meanFieldMethodString:
        approximator = EnergyMeanField(energyDict=energyDict, colors=colors, dimDict=dimDict, coreType=coreType,
                                       contractionMethod=contractionMethod)
        approximator.anneal(temperatureList=temperatureList)
        return approximator.draw_sample()
    elif method == energyMaximumMethodString:
        contracted = engine.create_trivial_core("contracted", [dimDict[color] for color in colors], colors,
                                                coreType=coreType)
        for energyKey in energyDict:
            contracted = contracted.sum_with(
                engine.contract(energyDict[energyKey][0], openColors=colors, dimDict=dimDict,
                                method=contractionMethod).multiply(
                    energyDict[energyKey][1]))
        return contracted.get_argmax()
    else:
        raise ValueError("Energy Optimization Method {} not implemented.".format(energyMaximumMethodString,
                                                                                 method))


class EnergyMeanField:
    def __init__(self, energyDict, colors=[], dimDict={}, partitionColorDict=None, coreType=None,
                 contractionMethod=None):
        self.energyDict = energyDict
        self.colors = colors

        self.affectionDict = create_affectionDict(energyDict, colors)
        self.dimDict = dimDict

        self.coreType = coreType
        self.contractionMethod = contractionMethod

        # Only distinction to Gibbs: MeanCores instead of samples turned into cores
        if partitionColorDict is None:
            self.partitionColorDict = {color: [color] for color in colors}
        else:
            self.partitionColorDict = partitionColorDict
        self.meanCores = {parKey: engine.create_trivial_core(parKey, [self.dimDict[color] for color in
                                                                      self.partitionColorDict[parKey]],
                                                             self.partitionColorDict[parKey],
                                                             coreType=coreType).multiply(
            1 / np.prod([self.dimDict[color] for color in self.partitionColorDict[parKey]])) for parKey
            in self.partitionColorDict}

    def update_meanCore(self, upKey, temperature=1):

        oldMean = self.meanCores[upKey].clone()

        restMeanCores = {secKey: self.meanCores[secKey] for secKey in self.meanCores if secKey != upKey}
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in self.partitionColorDict[upKey]]))
        contracted = engine.contract(
            {**restMeanCores,
             **self.energyDict[affectedEnergyKeys[0]][0]
             }, openColors=self.partitionColorDict[upKey], dimDict=self.dimDict,
            method=self.contractionMethod).multiply(
            self.energyDict[affectedEnergyKeys[0]][1])
        for energyKey in affectedEnergyKeys[1:]:
            contracted.sum_with(
                engine.contract({**restMeanCores,
                                 **self.energyDict[energyKey][0]},
                                openColors=self.partitionColorDict[upKey], dimDict=self.dimDict,
                                method=self.contractionMethod).multiply(
                    self.energyDict[energyKey][1])
            )
        self.meanCores[upKey] = contracted.multiply(1 / temperature).exponentiate().normalize()

        angle = engine.contract({"old": oldMean, "new": self.meanCores[upKey]}, openColors=[],
                                method=self.contractionMethod)
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
    def __init__(self, energyDict, colors=[], dimDict={}, coreType=None, contractionMethod=None):
        self.energyDict = energyDict
        self.colors = colors

        self.affectionDict = create_affectionDict(energyDict, colors)

        self.dimDict = dimDict
        self.sample = {}

        self.coreType = coreType
        self.contractionMethod = contractionMethod

    def initialize_sample_uniform(self):
        for color in self.colors:
            self.sample.update(
                engine.create_trivial_core(color + "_probCore", [self.dimDict[color]], [color],
                                           coreType=self.coreType).draw_sample())

    def calculate_energy(self, upColors):
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in upColors]))
        sampleCores = {
            color + "_sampleCore": engine.create_basis_core(color + "_sampleCore", [self.dimDict[color]], [color],
                                                            (self.sample[color]), coreType=self.coreType) for
            color in self.sample if color not in upColors}
        contractedEnergy = engine.contract(coreDict={**self.energyDict[affectedEnergyKeys[0]][0], **sampleCores},
                                           openColors=upColors, dimDict=self.dimDict,
                                           method=self.contractionMethod).multiply(
            self.energyDict[affectedEnergyKeys[0]][1])
        for energyKey in affectedEnergyKeys[1:]:
            contractedEnergy = contractedEnergy.sum_with(
                engine.contract({**self.energyDict[energyKey][0], **sampleCores}, openColors=upColors,
                                dimDict=self.dimDict, method=self.contractionMethod).multiply(
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


def create_affectionDict(energyDict, colors):
    return {color: [energyKey for energyKey in energyDict if
                    any([color in energyDict[energyKey][0][coreKey].colors for coreKey in energyDict[energyKey][0]])]
            for color in colors}

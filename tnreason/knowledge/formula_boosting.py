from tnreason import algorithms
from tnreason import encoding

from tnreason.knowledge import distributions

parameterCoreSuffix = "_parCore"


class FormulaBooster:
    def __init__(self, knowledgeBase, specDict):
        self.knowledgeBase = knowledgeBase
        self.specDict = specDict

    def find_candidate(self, sampleDf):
        networkCores = encoding.create_architecture(self.specDict["architecture"], self.specDict["headNeurons"])
        importanceColors = encoding.find_atoms(self.specDict["architecture"])

        empiricalDistribution = distributions.EmpiricalDistribution(sampleDf, importanceColors)

        importanceList = [
            (empiricalDistribution.create_cores(), 1 / empiricalDistribution.get_partition_function(importanceColors)),
            (self.knowledgeBase.create_cores(), -1 / self.knowledgeBase.get_partition_function(importanceColors))]

        colorDims = encoding.find_selection_dimDict(self.specDict["architecture"])
        updateShapes = {key + parameterCoreSuffix: colorDims[key] for key in colorDims}
        updateColors = {key + parameterCoreSuffix: [key] for key in colorDims}
        updateCoreKeys = list(updateShapes.keys())
        if self.specDict["method"] == "als":
            sampler = algorithms.ALS(networkCores=networkCores, importanceColors=importanceColors,
                                     importanceList=importanceList, targetCores={})
            sampler.random_initialize(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            sampler.alternating_optimization(updateKeys=updateCoreKeys, sweepNum=self.specDict["sweeps"])
            solutionDict = sampler.get_color_argmax(updateKeys=updateCoreKeys)

        elif self.specDict["method"] == "gibbs":
            sampler = algorithms.Gibbs(networkCores=networkCores, importanceColors=importanceColors,
                                       importanceList=importanceList)
            sampler.ones_initialization(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            if "annealingPattern" in self.specDict:
                sampleDict = sampler.annealed_sample(updateKeys=updateCoreKeys,
                                                     annealingPattern=self.specDict["annealingPattern"])
            elif "sweeps" in self.specDict:
                sampleDict = sampler.gibbs_sample(updateKeys=updateCoreKeys, sweepNum=self.specDict["sweeps"])
            else:
                raise ValueError("Bad parameter specification for Gibbs: {}".format(self.specDict))
            solutionDict = {key[:-len(parameterCoreSuffix)]: int(sampleDict[key]) for key in
                            sampleDict}  # Drop parameterCoreSuffix and ensure int output
        else:
            raise ValueError("Sampling Method {} not known!".format(self.specDict["method"]))

        self.candidates = encoding.create_solution_expression(self.specDict["architecture"], solutionDict)

    def test_candidates(self):
        if self.specDict["acceptanceCriterion"] == "always":
            return True

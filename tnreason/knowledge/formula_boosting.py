from tnreason import algorithms
from tnreason import encoding

from tnreason.knowledge import distributions

parameterCoreSuffix = "_parCore"

methodSelectionString = "method"
alsOptionString = "als"
gibbsOptionString = "gibbs"
annealPatternOptionString = "annealingPattern"

headNeuronString = "headNeurons"
architectureString = "architecture"
structureSweepsString = "sweeps"

acceptanceCriterionString = "acceptanceCriterion"

def check_boosting_dict(specDict):
    if methodSelectionString not in specDict:
        raise ValueError("Method not specified for Boosting a formula!")
    if headNeuronString not in specDict:
        raise ValueError("Head Neuron not specified for Boosting a formula!")
    if methodSelectionString not in specDict:
        raise ValueError("Architecture is not specified for Boosting a formula!")

class FormulaBooster:
    """
    Dedicates structure learning to the algorithm subpackage.
    """
    def __init__(self, knowledgeBase, specDict):
        self.knowledgeBase = knowledgeBase
        self.specDict = specDict

    def find_candidate(self, sampleDf):
        """
        Searches for a candidate assignment to the architecture with maximal alignment to the likelihood gradient.
        """
        networkCores = encoding.create_architecture(self.specDict[architectureString], self.specDict[headNeuronString])
        importanceColors = encoding.find_atoms(self.specDict[architectureString])

        empiricalDistribution = distributions.EmpiricalDistribution(sampleDf, importanceColors)

        importanceList = [
            (empiricalDistribution.create_cores(), 1 / empiricalDistribution.get_partition_function(importanceColors)),
            (self.knowledgeBase.create_cores(), -1 / self.knowledgeBase.get_partition_function(importanceColors))]

        colorDims = encoding.find_selection_dimDict(self.specDict[architectureString])
        updateShapes = {key + parameterCoreSuffix: colorDims[key] for key in colorDims}
        updateColors = {key + parameterCoreSuffix: [key] for key in colorDims}
        updateCoreKeys = list(updateShapes.keys())

        ## When alternating least squares used for structure learning
        if self.specDict[methodSelectionString] == alsOptionString:
            sampler = algorithms.ALS(networkCores=networkCores, importanceColors=importanceColors,
                                     importanceList=importanceList, targetCores={})
            sampler.random_initialize(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            sampler.alternating_optimization(updateKeys=updateCoreKeys, sweepNum=self.specDict[structureSweepsString])
            solutionDict = sampler.get_color_argmax(updateKeys=updateCoreKeys)
        ## When annealed Gibbs sampling used for structure learning
        elif self.specDict[methodSelectionString] == gibbsOptionString:
            sampler = algorithms.Gibbs(networkCores=networkCores, importanceColors=importanceColors,
                                       importanceList=importanceList,
                                       exponentiated=True)
            sampler.ones_initialization(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            if annealPatternOptionString in self.specDict:
                sampleDict = sampler.annealed_sample(updateKeys=updateCoreKeys,
                                                     annealingPattern=self.specDict[annealPatternOptionString])
            elif structureSweepsString in self.specDict:
                sampleDict = sampler.gibbs_sample(updateKeys=updateCoreKeys, sweepNum=self.specDict[structureSweepsString])
            else:
                raise ValueError("Bad parameter specification for Gibbs: {}".format(self.specDict))
            solutionDict = {key[:-len(parameterCoreSuffix)]: int(sampleDict[key]) for key in
                            sampleDict}  # Drop parameterCoreSuffix and ensure int output
        else:
            raise ValueError("Sampling Method {} not known!".format(self.specDict[methodSelectionString]))

        self.candidates = encoding.create_solution_expression(self.specDict[architectureString], solutionDict)

    def test_candidates(self):
        """
        Tests whether to accept the candidate.
        """
        if self.specDict["acceptanceCriterion"] == "always":
            return True
        else:
            raise ValueError("Acceptance Criterion {} not understood.".format(self.specDict[acceptanceCriterionString]))

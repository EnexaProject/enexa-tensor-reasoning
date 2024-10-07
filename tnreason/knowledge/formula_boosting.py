from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

from tnreason.knowledge import distributions

parameterCoreSuffix = "_parCore"

methodSelectionString = "method"

## Energy-based
gibbsMethodString = "gibbsSample"
meanFieldMethodString = "meanFieldSample"
gibbsAnnealingArgument = "annealingPattern"  # used in meanField and gibbs
energyMaximumMethodString = "exactEnergyMax"
## KLDivergence-based
klMaximumMethodString = "exactKLMax"

headNeuronString = "headNeurons"
architectureString = "architecture"
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
        Searches for a candidate formula
        """
        atomColors = encoding.find_atoms(self.specDict[architectureString])
        selectionColors = encoding.find_selection_colors(self.specDict[architectureString])

        empiricalDistribution = distributions.EmpiricalDistribution(sampleDf, atomColors)
        statisticCores = encoding.create_architecture(self.specDict[architectureString],
                                                      self.specDict[headNeuronString])

        energyDict = {"pos": ({**statisticCores, **empiricalDistribution.create_cores()},
                              1 / empiricalDistribution.get_partition_function(atomColors)),
                      "neg": ({**statisticCores, **self.knowledgeBase.create_cores()},
                              -1 / self.knowledgeBase.get_partition_function(atomColors))}
        dimDict = engine.get_dimDict(statisticCores)

        ## Energy optimization methods
        if self.specDict[methodSelectionString] == energyMaximumMethodString:
            solutionDict = engine.contract(energyDict["pos"][0], openColors=selectionColors).multiply(
                energyDict["pos"][1]).sum_with(
                engine.contract(energyDict["neg"][0], openColors=selectionColors).multiply(
                    energyDict["neg"][1])).get_argmax()
        elif self.specDict[methodSelectionString] == gibbsMethodString:
            sampler = algorithms.EnergyGibbs(energyDict=energyDict, colors=selectionColors, dimDict=dimDict)
            if gibbsAnnealingArgument in self.specDict:
                temperatureList = self.specDict[gibbsAnnealingArgument]
            else:
                temperatureList = [1 for i in range(10)]
            sampler.annealed_sample(temperatureList=temperatureList)
            solutionDict = sampler.sample
        elif self.specDict[methodSelectionString] == meanFieldMethodString:
            approximator = algorithms.EnergyMeanField(energyDict=energyDict, colors=selectionColors, dimDict=dimDict)
            if gibbsAnnealingArgument in self.specDict:
                temperatureList = self.specDict[gibbsAnnealingArgument]
            else:
                temperatureList = [1 for i in range(10)]
            approximator.anneal(temperatureList=temperatureList)
            solutionDict = approximator.draw_sample()
        ## Brute force KL Divergence method
        elif self.specDict[methodSelectionString] == klMaximumMethodString:
            posPhase = engine.contract(energyDict["pos"][0],
                                       openColors=encoding.find_selection_colors(
                                           self.specDict[architectureString])).multiply(energyDict["pos"][1])
            negPhase = engine.contract(energyDict["neg"][0],
                                       openColors=encoding.find_selection_colors(
                                           self.specDict[architectureString])).multiply(-energyDict["neg"][1])
            klDivergences = posPhase.calculate_coordinatewise_kl_to(negPhase)

            solutionDict = klDivergences.get_argmax()
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

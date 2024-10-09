from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

from tnreason.knowledge import distributions

parameterCoreSuffix = "_parCore"

headNeuronString = "headNeurons"
architectureString = "architecture"
acceptanceCriterionString = "acceptanceCriterion"
methodSelectionString = "method"  # Entry in specDict, either one of algorithms.energyOptimizationMethods or klMaximumMethodString
annealingArgumentString = "annealingPattern"  # used in meanField and gibbs

## KLDivergence-based
klMaximumMethodString = "exactKLMax"


def check_boosting_dict(specDict):
    if methodSelectionString not in specDict:
        raise ValueError("Method not specified for Boosting a formula!")
    if headNeuronString not in specDict:
        raise ValueError("Head Neuron not specified for Boosting a formula!")
    if methodSelectionString not in specDict:
        raise ValueError("Architecture is not specified for Boosting a formula!")


class Grafter:
    """
    Searches for best formula by the grafting heuristic: Formulation by an energy optimization problem
    Exceptional handling of KL Divergence: Distinguish between positive and negative phase
    when calculating coordinatewise KL divergence
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

        ## Energy optimization methods: Ignores constrastive structure of energyDict
        if self.specDict[methodSelectionString] in algorithms.energyOptimizationMethods:
            if annealingArgumentString in self.specDict:
                temperatureList = self.specDict[annealingArgumentString]
            else:
                temperatureList = [1 for i in range(10)]
            solutionDict = algorithms.optimize_energy(energyDict=energyDict, colors=selectionColors, dimDict=dimDict,
                                                      method=self.specDict[methodSelectionString],
                                                      temperatureList=temperatureList)

        ## Brute force KL Divergence method: Makes usage of the contrastive structure
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

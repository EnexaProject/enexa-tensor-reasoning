import pandas as pd

from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import formula_boosting as fb
from tnreason.knowledge import distributions as dist
from tnreason.knowledge import deductive as ded


class HybridLearner:
    """
    Intended to use for extending a Knowledge Base based on data.
    Iterating between:
        - structure learning: Using the FormulaBooster to learn new formulas
        - weight estimation: Using the EntropyMaximizer to adjust the weights to the formulas
    """

    def __init__(self, startKB):
        """
        startKB a knowledge.HybridKnowledgeBase instance representing the current knowledge to be extended.
        """
        self.hybridKB = startKB

    def get_knowledge_base(self):
        return self.hybridKB

    def boost_formula(self, specDict, sampleDf, stepName="_boostStep"):
        """
        Boosting with
        * specDict: Dictionary specification of the Hyperparameters of the Boosting Step:
            - method: Method for structure learning: als or gibbs supported
            - sweeps: Number of sweeps in structure learning
            - architecture: Collection of neurons
            - headNeurons: List of neuronKeys to be used for formula heads
            - calibrationSweeps: Number of sweeps in weight estimation
        * sampleDf: pd.DataFrame storing the data used for the boosting step
        * stepName: Specifies a name suffix for the learned formula to be stored in the HybridKnowledgeBase.
                    Needs to differ for each Step to avoid key conflicts.
        """
        booster = fb.FormulaBooster(self.hybridKB, specDict)
        booster.find_candidate(sampleDf)
        print("Learned formulas: {}".format(booster.candidates))
        if booster.test_candidates():
            print("Accepted formulas.")
            self.hybridKB.include(
                dist.HybridKnowledgeBase(weightedFormulas={
                    candidateKey + stepName: booster.candidates[candidateKey] + [0] for candidateKey in
                    booster.candidates}))
            if "calibrationSweeps" not in specDict:
                specDict["calibrationSweeps"] = 10
            self.calibrate_weights_on_data(specDict, sampleDf=sampleDf)

    def calibrate_weights_on_data(self, specDict, sampleDf, formulaKeys=None):
        if formulaKeys is None:
            formulaKeys = self.hybridKB.weightedFormulas.keys()

        tboFormulas = {
            key: self.hybridKB.weightedFormulas[key][:-1] for key in formulaKeys}

        empDistributionInferer = ded.InferenceProvider(dist.EmpiricalDistribution(sampleDf))
        satDict = {key: empDistributionInferer.ask(tboFormulas[key]) for key in tboFormulas}
        calibrator = wees.EntropyMaximizer(expressionsDict=tboFormulas, satisfactionDict=satDict,
                                           backCores=self.hybridKB.create_hard_cores())

        weightDict, factsDict = calibrator.alternating_optimization(sweepNum=specDict["calibrationSweeps"])
        for key in weightDict:
            self.hybridKB.weightedFormulas[key][-1] = weightDict[key][-1]
        for key in factsDict:
            formula = self.hybridKB.weightedFormulas.pop(key)
            if factsDict[key]:
                self.hybridKB.facts[key] = formula[:-1]
            else:
                self.hybridKB.facts[key] = ["not",formula[:-1]]
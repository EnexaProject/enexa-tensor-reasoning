from tnreason import knowledge

from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import formula_boosting as fb


class HybridLearner:

    def __init__(self, startKB):
        self.hybridKB = startKB

    def get_kb(self):
        return self.hybridKB

    def boost_formula(self, specDict, sampleDf, stepName="boostStep"):
        booster = fb.FormulaBooster(self.hybridKB, specDict)
        booster.find_candidate(sampleDf)
        if booster.test_candidate():
            self.hybridKB.include(
                knowledge.HybridKnowledgeBase(weightedFormulas={stepName:
                                                                    booster.candidate}))
            if "calibrationSweeps" not in specDict:
                specDict["calibrationSweeps"] = 10
            self.calibrate_weights_on_data(specDict, sampleDf=sampleDf)


    def calibrate_weights_on_data(self, specDict, sampleDf, formulaKeys=None):
        if formulaKeys is None:
            formulaKeys = self.hybridKB.weightedFormulas.keys()
        satDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict({
            key: self.hybridKB.weightedFormulas[key][:-1] for key in formulaKeys})

        calibrator = wees.EntropyMaximizer(self.hybridKB, satisfactionDict=satDict)

        solutionDict = calibrator.alternating_optimization(sweepNum=specDict["calibrationSweeps"])
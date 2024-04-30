from tnreason import knowledge

from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import formula_boosting as fb


class HybridLearner:

    def __int__(self, startKB):
        self.hybridKB = startKB
    def get_kb(self):
        return self.hybridKB

    def boost_formula(self, specDict, architectureDict, sampleDf):
        booster = fb.FormulaBooster(specDict)
        booster.find_candidate(architectureDict, sampleDf)

    def calibrate_weights_on_data(self, sampleDf, sweepNum=10):
        satDict = wees.EmpiricalDistribution(sampleDf).get_satisfactionDict({
            key: self.weightedFormulasDict[key][:-1] for key in self.weightedFormulasDict})
        wees.EntropyMaximizer(expressionsDict=self.hybridKB.weightedFormulasDict, satisfactionDict=satDict)

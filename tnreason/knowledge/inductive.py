from tnreason import knowledge

from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import formula_boosting as fb


class HybridLearner:

    def __init__(self, startKB):
        self.hybridKB = startKB

    def get_kb(self):
        return self.hybridKB

    def boost_formula(self, specDict, sampleDf, stepName="_boostStep"):
        booster = fb.FormulaBooster(self.hybridKB, specDict)
        booster.find_candidate(sampleDf)
        print("Learned formulas: {}".format(booster.candidates))
        if booster.test_candidates():
            print("Accepted formulas.")
            self.hybridKB.include(
                knowledge.HybridKnowledgeBase(weightedFormulas={
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

        satDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(tboFormulas)
        calibrator = wees.EntropyMaximizer(expressionsDict=tboFormulas, satisfactionDict=satDict,
                                           backCores=self.hybridKB.create_cores(hardOnly=True))

        solutionDict = calibrator.alternating_optimization(sweepNum=specDict["calibrationSweeps"])
        print(solutionDict)

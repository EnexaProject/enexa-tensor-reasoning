from tnreason.logic import expression_utils as eu
from tnreason.logic import expression_generation as eg

from tnreason.model import tensor_model as tm
from tnreason.model import infer_mln as imln

import numpy as np
import pandas as pd


class SamplerBase:
    def __init__(self, expressionsDict):
        self.expressionsDict = expressionsDict
        self.atoms = eu.get_all_variables([self.expressionsDict[formulaKey][0] for formulaKey in self.expressionsDict])
        self.marginalizedDict = None

    def compute_marginalized_distributions(self):
        tensorRepresented = tm.TensorRepresentation(self.expressionsDict, headType="expFactor")
        self.marginalizedDict = {atomKey: tensorRepresented.marginalized_contraction([atomKey]).normalize()
                                 for atomKey in self.atoms}

    def create_independent_sample(self):
        if self.marginalizedDict is None:
            self.compute_marginalized_distributions()
        return {atomKey: np.random.multinomial(1, self.marginalizedDict[atomKey].values)[0] == 0 for atomKey in
                self.atoms}

    ## To be outsourced to LogicModel!
    def infer_expressionsDict(self, evidenceDict={}):
        inferedExpressionsDict = {}
        for key in self.expressionsDict:
            inferedFormula = imln.infer_expression(self.expressionsDict[key][0], evidenceDict)
            if inferedFormula not in ["Thing", "Nothing"]:
                inferedFormula = eg.remove_double_not(inferedFormula)
                inferedExpressionsDict[key] = [inferedFormula, self.expressionsDict[key][1]]
        return inferedExpressionsDict


class GibbsSampler(SamplerBase):
    def create_sampleDf(self, sampleNum, chainLength=10, outType="int64"):
        sampleDf = pd.DataFrame(columns=self.atoms)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf, pd.DataFrame(self.gibbs_sample(chainLength=chainLength), index=[samplePos])])
        return sampleDf.astype(outType)

    def gibbs_sample(self, chainLength):
        sampleDict = self.create_independent_sample()
        for sweep in range(chainLength):
            for updateAtomKey in self.atoms:
                miniSampler = SamplerBase(self.infer_expressionsDict(
                    {atomKey: sampleDict[atomKey] for atomKey in self.atoms if atomKey != updateAtomKey}))
                if updateAtomKey not in miniSampler.atoms:
                    miniSampler = SamplerBase({updateAtomKey: [str(updateAtomKey), 0]})
                sampleDict[updateAtomKey] = miniSampler.create_independent_sample()[updateAtomKey]
        return sampleDict


if __name__ == "__main__":
    learnedFormulaDict = {
        "f0": ["A1", 10],
        "f1": [["not", ["A2", "and", "A3"]], 10],
        "f2": ["A2", -10]
    }
    sampler = GibbsSampler(learnedFormulaDict)
    sampler.compute_marginalized_distributions()

    print(sampler.create_sampleDf(100, 20)["A1"])

from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.model import tensor_model as tm
from tnreason.model import logic_model as lm
from tnreason.model import model_visualization as mv

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

    def infer_expressionsDict(self, evidenceDict={}):
        logRep = lm.LogicRepresentation(self.expressionsDict)
        logRep.infer(evidenceDict=evidenceDict, simplify=True)
        return logRep.expressionsDict

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None, savePath=None):
        return mv.visualize_model(self.expressionsDict,
                                  strengthMultiplier=strengthMultiplier,
                                  strengthCutoff=strengthCutoff,
                                  fontsize=fontsize,
                                  showFormula=showFormula,
                                  evidenceDict=evidenceDict,
                                  pos=pos,
                                  savePath=savePath,
                                  show=False)


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


class ExactSampler:
    def __init__(self, expressionsDict, margAtoms=None):
        self.expressionsDict = expressionsDict
        self.atoms = eu.get_all_variables([self.expressionsDict[formulaKey][0] for formulaKey in self.expressionsDict])

        if margAtoms is None:
            margAtoms = self.atoms
        self.compute_distributionCore(margAtoms)

    def compute_distributionCore(self, margAtoms):
        tensorRepresented = tm.TensorRepresentation(self.expressionsDict, headType="expFactor")
        self.distributionCore = tensorRepresented.marginalized_contraction(margAtoms).normalize()

    def compute_superposedSampleCore(self, sampleNum):
        return cc.CoordinateCore(np.random.multinomial(sampleNum, self.distributionCore.values.flatten()).reshape(
            self.distributionCore.values.shape), self.distributionCore.colors)


if __name__ == "__main__":
    learnedFormulaDict = {
        "f0": ["A1", 1],
        "f1": [["not", ["A2", "and", "A3"]], 1],
        "f2": ["A2", -1]
    }

    sampler = GibbsSampler(learnedFormulaDict)
    pos = None
    for rep in range(10):
        sample = sampler.gibbs_sample(chainLength=10)
        pos = sampler.visualize(evidenceDict=sample, pos=pos, savePath=None)#"./demonstration/visualizations/gibbs{}.png".format(rep))

    sampler.compute_marginalized_distributions()
    print(sampler.create_sampleDf(100, 20))

    exit()

    exactSampler = ExactSampler(learnedFormulaDict)
    sampleNums = [1, 10, 100, 1000, 10000]
    for sampleNum in sampleNums:
        print(sampleNum, np.linalg.norm(
            exactSampler.distributionCore.values - exactSampler.compute_superposedSampleCore(
                sampleNum).values / sampleNum))

from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.model import tensor_model as tm
from tnreason.model import logic_model as lm
from tnreason.model import model_visualization as mv

import numpy as np
import pandas as pd


class SamplerBase:
    def __init__(self, expressionsDict, factsDict={}):
        self.expressionsDict = expressionsDict.copy()
        self.factsDict = factsDict.copy()

        self.atoms = list(eu.get_all_variables([expressionsDict[formulaKey][0] for formulaKey in expressionsDict] +
                                               [factsDict[key] for key in factsDict]))

        self.marginalizedDict = None

    def change_temperature(self, temperature):
        self.expressionsDict = {key: [self.expressionsDict[key][0], self.expressionsDict[key][1] / temperature]
                                for key in self.expressionsDict}

    def compute_marginalized_distributions(self, variablesList=[]):
        marginalAtoms = list(set(self.atoms) | set(variablesList))
        tensorRepresented = tm.TensorRepresentation(self.expressionsDict, self.factsDict, headType="expFactor")
        self.marginalizedDict = {atomKey: tensorRepresented.marginalized_contraction([atomKey]).normalize()
                                 for atomKey in marginalAtoms}

    def create_independent_sample(self, variableList=[]):
        if len(variableList) == 0:
            variableList = self.atoms

        if self.marginalizedDict is None:
            self.compute_marginalized_distributions(variableList)

        for atomKey in variableList:
            assert len(self.marginalizedDict[atomKey].colors) == 1, "Marginalization failed for atom {}.".format(
                atomKey)

        return {atomKey: np.random.multinomial(1, self.marginalizedDict[atomKey].values)[0] == 0 for atomKey in
                variableList}

    def infer_formulas_and_facts(self, evidenceDict={}):
        logRep = lm.LogicRepresentation(self.expressionsDict, self.factsDict)
        logRep.infer(evidenceDict=evidenceDict, simplify=True)
        return logRep.get_formulas_and_facts()

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None, savePath=None, show=False):
        return mv.visualize_model(self.expressionsDict,
                                  strengthMultiplier=strengthMultiplier,
                                  strengthCutoff=strengthCutoff,
                                  fontsize=fontsize,
                                  showFormula=showFormula,
                                  evidenceDict=evidenceDict,
                                  pos=pos,
                                  savePath=savePath,
                                  show=show)


class GibbsSampler(SamplerBase):
    def create_sampleDf(self, sampleNum, chainLength=10, outType="int64"):
        sampleDf = pd.DataFrame(columns=self.atoms)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf, pd.DataFrame(self.gibbs_sample(chainLength=chainLength), index=[samplePos])])
        return sampleDf.astype(outType)

    def gibbs_step(self, evidenceDict, updateAtomKey, temperature=1):
        stepSampler = SamplerBase(*self.infer_formulas_and_facts(
            {atomKey: evidenceDict[atomKey] for atomKey in evidenceDict if atomKey != updateAtomKey}
        ))
        stepSampler.change_temperature(temperature)
        if updateAtomKey not in stepSampler.atoms:
            stepSampler = SamplerBase({updateAtomKey: [str(updateAtomKey), 0]})
        return stepSampler.create_independent_sample()[updateAtomKey]

    def gibbs_sample(self, chainLength, variableList=None, temperature=1):
        if variableList is None:
            variableList = self.atoms
        sampleDict = self.create_independent_sample(variableList)
        for sweep in range(chainLength):
            for updateAtomKey in variableList:
                sampleDict[updateAtomKey] = self.gibbs_step(sampleDict, updateAtomKey, temperature=temperature)
        return sampleDict

    def simulated_annealing_gibbs(self, variableList, annealingPattern):
        self.compute_marginalized_distributions(variableList)
        sampleDict = self.create_independent_sample(variableList)
        for chainLength, temperature in annealingPattern:
            for sweep in range(chainLength):
                for updateAtomKey in variableList:
                    sampleDict[updateAtomKey] = self.gibbs_step(sampleDict, updateAtomKey, temperature=temperature)
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
    evidenceDict = sampler.gibbs_sample(10)
    sampler.visualize(evidenceDict=evidenceDict, show=True)

    exit()

    pos = None
    for rep in range(10):
        sample = sampler.gibbs_sample(chainLength=10)
        pos = sampler.visualize(evidenceDict=sample, pos=pos,
                                savePath=None)  # "./demonstration/visualizations/gibbs{}.png".format(rep))

    sampler.compute_marginalized_distributions()
    print(sampler.create_sampleDf(100, 20))

    exit()

    exactSampler = ExactSampler(learnedFormulaDict)
    sampleNums = [1, 10, 100, 1000, 10000]
    for sampleNum in sampleNums:
        print(sampleNum, np.linalg.norm(
            exactSampler.distributionCore.values - exactSampler.compute_superposedSampleCore(
                sampleNum).values / sampleNum))

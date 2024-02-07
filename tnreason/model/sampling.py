from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.model import tensor_model as tm
from tnreason.model import logic_model as lm
from tnreason.model import model_visualization as mv

from tnreason.contraction import core_contractor as coc

import numpy as np
import pandas as pd


class SamplerBase:
    def __init__(self, expressionsDict, factsDict={}, categoricalConstraintsDict={}):
        self.expressionsDict = expressionsDict.copy()
        self.factsDict = factsDict.copy()
        self.categoricalConstraintsDict = categoricalConstraintsDict.copy()

        self.atoms = list(eu.get_all_variables([expressionsDict[formulaKey][0] for formulaKey in expressionsDict] +
                                               [factsDict[key] for key in factsDict]))

        self.marginalizedDict = None

    def change_temperature(self, temperature):
        self.expressionsDict = {key: [self.expressionsDict[key][0], self.expressionsDict[key][1] / temperature]
                                for key in self.expressionsDict}

    def compute_marginalized_distributions(self, variablesList=[]):
        marginalAtoms = list(set(self.atoms) | set(variablesList))
        tensorRepresented = tm.TensorRepresentation(self.expressionsDict, self.factsDict,
                                                    categoricalConstraintsDict=self.categoricalConstraintsDict,
                                                    headType="expFactor")

        self.marginalizedDict = {atomKey: tensorRepresented.marginalized_contraction([atomKey]).normalize()
                                 for atomKey in marginalAtoms}

        for atomKey in marginalAtoms:
            if (np.sum(self.marginalizedDict[atomKey].values) - 1) > 0.001:
                print("Marginalization failed in atom {}!".format(atomKey))
                self.marginalizedDict[atomKey] = cc.CoordinateCore(np.array([0.5, 0.5]),
                                                                   self.marginalizedDict[atomKey].colors,
                                                                   self.marginalizedDict[atomKey].name)

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

    def gibbs_step(self, evidenceDict, updateAtomKeys, temperature=1):

        inferedFormulas, inferedFacts = self.infer_formulas_and_facts(
            {atomKey: evidenceDict[atomKey] for atomKey in evidenceDict if atomKey not in updateAtomKeys})
        inferedFacts = {**inferedFacts,
                        **{atomKey + "_posEvidence": atomKey for atomKey in evidenceDict if
                           atomKey not in updateAtomKeys and bool(evidenceDict[atomKey])},
                        **{atomKey + "_negEvidence": ["not", atomKey] for atomKey in evidenceDict if
                           atomKey not in updateAtomKeys and not bool(evidenceDict[atomKey])}}
        inferedFormulas = {**inferedFormulas,
                           **{updateAtomKey + "_ensure": [updateAtomKey, 0] for updateAtomKey in updateAtomKeys}}

        stepSampler = SamplerBase(inferedFormulas,
                                  factsDict=inferedFacts,
                                  categoricalConstraintsDict=self.categoricalConstraintsDict)

        stepSampler.change_temperature(temperature)

        return stepSampler.create_independent_sample(updateAtomKeys)

    def gibbs_sample(self, chainLength, variableList=None, temperature=1):
        if variableList is None:
            variableList = self.atoms
        sampleDict = self.create_independent_sample(variableList)
        for sweep in range(chainLength):
            for updateAtomKey in variableList:
                sampleDict[updateAtomKey] = self.gibbs_step(sampleDict, [updateAtomKey], temperature=temperature)[
                    updateAtomKey]
        return sampleDict

    def simulated_annealing_gibbs(self, variableList, annealingPattern):
        self.compute_marginalized_distributions(variableList)
        sampleDict = self.create_independent_sample(variableList)
        for chainLength, temperature in annealingPattern:
            for sweep in range(chainLength):
                for updateAtomKey in variableList:
                    sampleDict[updateAtomKey] = self.gibbs_step(sampleDict, [updateAtomKey], temperature=temperature)[
                        updateAtomKey]
        return sampleDict


class CategoricalGibbsSampler(SamplerBase):

    def turn_to_categorical(self):
        categoricalAtomSet = {atomKey for catKey in
                              self.categoricalConstraintsDict for atomKey in self.categoricalConstraintsDict[catKey]}
        for atomKey in self.atoms:
            if atomKey not in categoricalAtomSet:
                categoricalAtomSet.add(atomKey)
                self.categoricalConstraintsDict[atomKey + "_cat"] = [atomKey]

    def calculate_categorical_probability(self, categoricalKey, catEvidenceDict={}):
        restEvidenceDict = {
            **{catEvidenceDict[key]: 1 for key in catEvidenceDict if
               len(self.categoricalConstraintsDict[key]) > 1 and key != categoricalKey},
            **{self.categoricalConstraintsDict[key][0]: catEvidenceDict[key]
               for key in catEvidenceDict if len(self.categoricalConstraintsDict[key]) == 1 and key != categoricalKey}
        }
        variables = self.categoricalConstraintsDict[categoricalKey]
        if len(variables) > 1:
            marginalProb = np.empty(len(variables))
            for i, variable in enumerate(variables):
                tRep = tm.TensorRepresentation(self.expressionsDict,
                                               factsDict={**self.factsDict,
                                                          **{evidenceKey + "_evidence": ["not", evidenceKey] for
                                                             evidenceKey in variables
                                                             if evidenceKey != variable},
                                                          **{key + "_evidence": key for key in restEvidenceDict if
                                                             bool(restEvidenceDict[key])},
                                                          **{key + "_evidence": ["not", key] for key in restEvidenceDict
                                                             if not
                                                             bool(restEvidenceDict[key])}
                                                          },
                                               categoricalConstraintsDict=self.categoricalConstraintsDict,
                                               headType="expFactor")
                marginalProb[i] = coc.CoreContractor(tRep.all_cores(), openColors=[variable]).contract().values[1]

            marginalProb = 1 / np.sum(marginalProb) * marginalProb
        else:
            tRep = tm.TensorRepresentation({**self.expressionsDict,
                                            "added": [variables[0], 0]
                                            },
                                           factsDict={**self.factsDict,
                                                      **{key + "_evidence": key for key in restEvidenceDict if
                                                         bool(restEvidenceDict[key])},
                                                      **{key + "_evidence": ["not", key] for key in restEvidenceDict
                                                         if not
                                                         bool(restEvidenceDict[key])}},
                                           categoricalConstraintsDict=self.categoricalConstraintsDict,
                                           headType="expFactor")
            marginalProb = coc.CoreContractor(tRep.all_cores(), openColors=variables).contract().normalize().values
        return marginalProb

    def gibbs_step(self, categoricalKey, catEvidenceDict={}, temperature=1):
        marginalProb = self.calculate_categorical_probability(categoricalKey, catEvidenceDict)
        marginalProb = heat_prob(marginalProb, temperature)

        if len(self.categoricalConstraintsDict[categoricalKey]) > 1:
            return self.categoricalConstraintsDict[categoricalKey][
                np.where(np.random.multinomial(1, marginalProb) == 1)[0][0]]
        else:
            return np.random.multinomial(1, marginalProb)[0] == 0

    def gibbs_sampling(self, chainSize=10):
        self.turn_to_categorical()
        sampleDict = {key: self.gibbs_step(key) for key in self.categoricalConstraintsDict}

        for repetition in range(chainSize):
            for catKey in self.categoricalConstraintsDict:
                sampleDict[catKey] = self.gibbs_step(catKey, sampleDict)
        return self.catSample_to_standardSample(sampleDict)

    def catSample_to_standardSample(self, sampleDict):
        returnDict = {}
        for key in sampleDict:
            if len(self.categoricalConstraintsDict[key]) > 1:
                returnDict = {**returnDict,
                              **{atomKey: False for atomKey in self.categoricalConstraintsDict[key]}}
                returnDict[sampleDict[key]] = True
            elif len(self.categoricalConstraintsDict[key]) == 1:
                returnDict[self.categoricalConstraintsDict[key][0]] = sampleDict[key]
        return returnDict
    def simulated_annealing_gibbs(self, variableList, annealingPattern):
        self.turn_to_categorical()

        optimizationKeys = []
        for variable in variableList:
            found = False
            for catKey in self.categoricalConstraintsDict:
                if variable in self.categoricalConstraintsDict[catKey] and catKey not in optimizationKeys:
                    optimizationKeys.append(catKey)
                    found = True
            if not found:
                self.categoricalConstraintsDict[variable+"_emptyAdd"] = [variable]
                optimizationKeys.append(variable+"_emptyAdd")

        sampleDict = {key: self.gibbs_step(key) for key in optimizationKeys}

        for chainSize, temperature in annealingPattern:
            for sweep in range(chainSize):
                for catKey in optimizationKeys:
                    sampleDict[catKey] = self.gibbs_step(catKey, sampleDict, temperature=temperature)

        return self.catSample_to_standardSample(sampleDict)

def heat_prob(probArray, temperature):
    choiceNum = probArray.size
    for i in range(choiceNum):
        if probArray[i] > 0:
            probArray[i] = np.exp(1/temperature * np.log(probArray[i]))
    sum = np.sum(probArray)
    if np.isnan(probArray).any() or probArray[probArray<0].any() or sum==0:
        print("Bad sampling weights {} found and replaced by uniform.".format(probArray))
        return 1/choiceNum * np.ones(choiceNum)

    return 1/sum * probArray

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

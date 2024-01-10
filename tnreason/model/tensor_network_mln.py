from tnreason.logic import expression_utils as eu

from tnreason.contraction import core_contractor as coc
from tnreason.contraction import expression_evaluation as ee

from tnreason.model import logic_model as lm

import numpy as np
import pandas as pd


## Shifted main sampling functionality to sampling

class TensorMLN:
    def __init__(self, expressionsDict, formulaCoreDict=None):
        self.expressionsDict = expressionsDict
        self.atomKeys = eu.get_all_variables([self.expressionsDict[formulaKey][0] for formulaKey in self.expressionsDict])
        self.formulaCoreDict = formulaCoreDict

    def infer_on_evidenceDict(self, evidenceDict={}):
        inferedExpressionsDict = {}
        for key in self.expressionsDict:
            inferedFormula = lm.infer_expression(self.expressionsDict[key][0], evidenceDict)
            if inferedFormula not in ["Thing", "Nothing"]:
                inferedFormula = lm.reduce_double_not(inferedFormula)
                inferedExpressionsDict[key] = [inferedFormula, self.expressionsDict[key][1]]
        return TensorMLN(inferedExpressionsDict)

    def reduce_double_formulas(self):
        checkedKeys = []
        reducedExpressionDict = {}
        for key in self.expressionsDict:
            if key not in checkedKeys:
                checkedKeys.append(key)
                keyFormula, keyWeight = self.expressionsDict[key]
                for otherKey in self.expressionsDict:
                    if otherKey not in checkedKeys and lm.equality_check(keyFormula, self.expressionsDict[otherKey][0]):
                        checkedKeys.append(otherKey)
                        keyWeight = keyWeight + self.expressionsDict[otherKey][1]
                reducedExpressionDict[key] = [keyFormula, keyWeight]
        self.expressionsDict = reducedExpressionDict

    def initialize_formulaCoreDict(self):
        self.formulaCoreDict = {
            formulaKey: ee.ExpressionEvaluator(self.expressionsDict[formulaKey][0],
                                                initializeBasisCores=True).create_formula_factor().weighted_exponentiation(
                self.expressionsDict[formulaKey][1])
            for formulaKey in self.expressionsDict}

    def compute_marginalized(self, marginalKeys, contractionMethod="formulaDict", optimizationMethod="GreedyHeuristic"):
        if contractionMethod == "formulaDict":
            if self.formulaCoreDict is None:
                self.initialize_formulaCoreDict()
            contractionDict = self.formulaCoreDict.copy()
        elif contractionMethod == "basisCalculus":
            contractionDict = {}
            for expression in self.expressionsDict:
                pass
        else:
            raise ValueError("Contraction Method {} not understood!".format(contractionMethod))
        contractor = coc.CoreContractor(contractionDict, openColors=marginalKeys)
        return contractor.contract(optimizationMethod=optimizationMethod).normalize()

    ## To be implemented: Here we need Tensor Network contractions
    def create_independent_atom_sample(self, atomSampleKey):
        marginalProbCore = self.compute_marginalized([atomSampleKey])
        return np.random.multinomial(1, marginalProbCore.values)[0] == 0

    def create_independent_sample(self):
        return {atomKey: self.create_independent_atom_sample(atomKey) for atomKey in self.atomKeys}

    def gibbs(self, repetitionNum=10, verbose=True):
        sampleDict = self.create_independent_sample()
        for repetitionPos in range(repetitionNum):
            if verbose:
                print("## Gibbs Iteration {} ##".format(repetitionPos))
            for refinementAtomKey in sampleDict:
                samplerMLN = self.infer_on_evidenceDict(
                    {key: sampleDict[key] for key in sampleDict if key != refinementAtomKey})
                samplerMLN.reduce_double_formulas()
                if refinementAtomKey not in samplerMLN.atomKeys:
                    print("Warning: Infered MLN is empty on key {}".format(refinementAtomKey))
                    samplerMLN = TensorMLN({refinementAtomKey: [str(refinementAtomKey), 0]})
                sampleDict[refinementAtomKey] = samplerMLN.create_independent_atom_sample(refinementAtomKey)
            if verbose:
                print("SampleDict is {}".format(sampleDict))
        return sampleDict

    def generate_sampleDf(self, sampleNum, method="Gibbs10"):
        df = pd.DataFrame(columns=self.atomKeys)
        for ind in range(sampleNum):
            if method.startswith("Gibbs"):
                repetitionNum = int(method.split("Gibbs")[1])
                row_df = pd.DataFrame(self.gibbs(repetitionNum=repetitionNum, verbose=False), index=[ind])
                df = pd.concat([df, row_df])
        return df.astype("int64")


#def create_formulaCoreDict(expressionsDict):
#    return {formulaKey: ec.calculate_expressionCore(expressionsDict[formulaKey][0]).weighted_exponentiation(
#        expressionsDict[formulaKey][1])
#        for formulaKey in expressionsDict}





if __name__ == "__main__":
    example_expression_dict = {
        "e0": [["not", ["Unterschrank(z)", "and", ["not", "Moebel(z)"]]], 20],
        "e0.5": ["Moebel(z)", 4],
        "e0.625": ["Sledz", 4],
        "e0.626": [["not", ["Moebel(z)", "and", "Sledz"]], 5],
        "e0.75": [["not", [["Unterschrank(z)", "and", "Sledz"], "and", ["not", "Ausgangsrechnung(x)"]]], 2],
        "e1": [["not", "Ausgangsrechnung(x)"], 12],
        "e2": [[["not", "Ausgangsrechnung(x)"], "and", ["not", "Rechnung(x)"]], 14]
    }
    ## 1 = True, 0 = False
    example_evidence_dict = {
        "Unterschrank(z)": 1,
        "Ausgangsrechnung(x)": 1
    }

    tn_mln = TensorMLN(example_expression_dict)
    print(tn_mln.generate_sampleDf(int(100)))

    infered_mln = tn_mln.infer_on_evidenceDict(example_evidence_dict)

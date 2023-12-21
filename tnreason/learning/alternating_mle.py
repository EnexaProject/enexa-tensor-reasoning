import tnreason.logic.expression_utils as eu

import tnreason.representation.sampledf_to_cores as stoc

import tnreason.contraction.bc_contraction_generation as bcg
import tnreason.contraction.expression_evaluation as ee
import tnreason.contraction.core_contractor as coc

import numpy as np


class AlternatingMLE:
    def __init__(self, skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict={}):
        self.skeleton = skeletonExpression
        self.skeletonAtoms = eu.get_variables(skeletonExpression)
        self.candidatesDict = candidatesDict

        self.variableCoresDict = variableCoresDict
        self.learnedFormulaDict = learnedFormulaDict

    def generate_mln_core(self):
        self.rawCoreDict = bcg.generate_rawCoreDict(
            {formulaKey: self.learnedFormulaDict[formulaKey][0] for formulaKey in self.learnedFormulaDict})



    def random_initialize_variableCoresDict(self):
        for coreKey in self.variableCoresDict:
            self.variableCoresDict[coreKey].values = np.random.random(size=self.variableCoresDict[coreKey].values.shape)

    def create_variable_gradient(self, tboVariableKey):
        return {key: self.variableCoresDict[key] for key in self.variableCoresDict if key != tboVariableKey}

    def create_exponentiated_variables(self):
        ## To do: Generate coordinatewise exponentiated
        pass


    def create_fixedCores(self, sampleDf):
        self.dataNum = sampleDf.values.shape[0]
        self.fixedCoresDict = {}
        # Supports only and connectivity!
        for atomKey in self.skeletonAtoms:
            self.fixedCoresDict[atomKey] = stoc.create_fixedCore(sampleDf, self.candidatesDict[atomKey], ["j", atomKey], atomKey)


    def contract_truth_gradient(self, tboVariableKey):
        ## Contract the variable gradient with the fixedCores
        contractor = coc.CoreContractor(coreDict={**self.fixedCoresDict,**self.create_variable_gradient(tboVariableKey)},
                                        openColors=self.variableCoresDict[tboVariableKey].colors)
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract()

    def contract_gradient_exponential(self, tboVariableKey):
        ## To do: Generate contraction for term in first order condition
        pass

    def contract_double_gradient_exponential(self, tboVariableKey):
        ## To do: Generate contraction for term in Jacobi Matrix
        pass

    def newton_step(self, tboVariableKey):
        ## To do: Do Multidimensional Newton on First Order Condition
        pass

if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    skeletonExpression = ["P1", "and", "P2"]
    candidatesDict = {"P1": list(atomDict.keys()),
                      "P2": list(atomDict.keys()),
                      }

    variableCoresDict = {
        "l1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
        "l2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    }

    learnedFormulaDict = {
        "f0" : ["b", 10],
        "f1" : [["not",["a","and","b"]], 5],
        "f2" : ["c", 2]
    }


    optimizer = AlternatingMLE(skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict)
    optimizer.random_initialize_variableCoresDict()
    optimizer.generate_mln_core()

    import tnreason.model.generate_test_data as gtd

    sampleDf = gtd.generate_sampleDf(learnedFormulaDict,100)
    optimizer.create_fixedCores(sampleDf)

    print(optimizer.contract_truth_gradient("l1").values)

    print(optimizer.rawCoreDict.keys())
    print(optimizer.dataNum)
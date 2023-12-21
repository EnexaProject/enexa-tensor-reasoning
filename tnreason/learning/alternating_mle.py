import tnreason.logic.expression_utils as eu
import tnreason.contraction.bc_contraction_generation as bcg

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
        "f1" : [["not",["a","and","b"]], 5]
    }
    optimizer = AlternatingMLE(skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict)
    optimizer.random_initialize_variableCoresDict()
    optimizer.generate_mln_core()

    print(optimizer.rawCoreDict.keys())
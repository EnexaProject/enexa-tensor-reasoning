import numpy as np

from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import basis_calculus as bc


class ExpressionEvaluator:
    def __init__(self, expression, atomDict=None, sampleDf=None, initializeBasisCores=False):
        self.expression = expression
        if sampleDf is not None:
            self.create_atomDict_from_sampleDf(sampleDf)
        if atomDict is not None:
            self.atomDict = atomDict
        if initializeBasisCores:
            self.create_atomDict_basisCores()

    def create_atomDict_from_sampleDf(self, sampleDf):
        atomKeys = np.unique(eu.get_variables(self.expression))
        self.atomDict = {}
        for atomKey in atomKeys:
            if atomKey == "Thing":
                values = np.ones(sampleDf.shape[0])
            elif atomKey == "Nothing":
                values = np.zeros(sampleDf.shape[0])
            else:
                values = sampleDf[atomKey].astype("int64").values
            self.atomDict[atomKey] = cc.CoordinateCore(values, ["j"], atomKey)

    def create_atomDict_basisCores(self):
        atomKeys = np.unique(eu.get_variables(self.expression))
        self.atomDict = {}
        for atomKey in atomKeys:
            ## How to include Thing/Nothing atomKeys?
            self.atomDict[atomKey] = bc.BasisCore(np.eye(2), [atomKey, "head"], headcolor="head", name=atomKey)

    def evaluate(self):
        return self.calculate_core(self.expression)

    def evaluate_on_sampleDf(self, sampleDf):
        self.create_atomDict_from_sampleDf(sampleDf)
        return self.calculate_core(self.expression)

    def create_formula_factor(self):
        return self.calculate_core(self.expression).calculate_truth().reduce_identical_colors().to_coordinate()

    ## Included here from expression_calculus: Benefit is avoiding copies of the atomDict.
    def calculate_core(self, expression):
        if type(expression) == str:
            return self.atomDict[expression]
        elif expression[0] == "not":
            return self.calculate_core(expression[1]).negate()
        elif expression[1] == "and":
            return self.calculate_core(expression[0]).compute_and(
                self.calculate_core(expression[2]))
        else:
            raise ValueError("Expression {} not understood.".format(expression))


if __name__ == "__main__":
    coordinateDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "q", "z"], name="b"),
    }

    # expression = ["a", "and", "b"]
    expression = ["not", "a"]

    evaluator = ExpressionEvaluator(expression, atomDict=coordinateDict)
    result = evaluator.evaluate()
    print(result.colors, result.values.shape, np.sum(result.values))

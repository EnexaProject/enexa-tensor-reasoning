from tnreason.optimization import satisfaction_counter as sc

from tnreason.logic import expression_utils as eu

from tnreason.contraction import core_contractor as coc
from tnreason.contraction import expression_evaluation as ee

import numpy as np

from tnreason.logic import coordinate_calculus as cc
from tnreason.representation import sampledf_to_cores as stoc

from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft


class WeightEstimator:
    '''
    formulaDict: formulaKey: [expression, satRate, empRate, weight]
    '''

    def __init__(self, formulaList=[], startWeightsDict={}, sampleDf=None):
        ## Have weight saved in self.formulaDict and in self.formulaTensors
        self.formulaDict = {"f" + str(i): [formulaKey, 0, 0, 0] for i, formulaKey in enumerate(formulaList)}
        for key in self.formulaDict:
            if key in startWeightsDict:
                self.formulaDict[key][3] = startWeightsDict[key]
            else:
                startWeightsDict[key] = 0

        self.formulaTensors = tm.TensorRepresentation(
            {formulaKey: [self.formulaDict[formulaKey][0], self.formulaDict[formulaKey][3]] for formulaKey in
             self.formulaDict})

        if sampleDf is not None:
            self.load_sampleDf(sampleDf)
        else:
            self.dataTensor = None

    def add_formula(self, key, expression, weight=1):
        self.formulaTensors.add_expression(expression, weight, formulaKey=key)

        empRate = coc.CoreContractor(
            {**self.formulaTensors.get_cores([key], headType="truthEvaluation"),
             **self.dataTensor.get_cores()}).contract().values / self.dataTensor.dataNum
        satRate = coc.CoreContractor(
            {**self.formulaTensors.get_cores([key], headType="truthEvaluation")}
        ).contract().values / 2 ** len(eu.get_variables(expression))
        self.formulaDict[key] = [expression, satRate, empRate, weight]

    def load_sampleDf(self, sampleDf):
        self.dataTensor = ft.DataTensor(sampleDf)

    def calculate_independent_satRates(self):
        for formulaKey in self.formulaDict:
            self.formulaDict[formulaKey][1] = coc.CoreContractor(
                {**self.formulaTensors.get_cores([formulaKey], headType="truthEvaluation")
                 }).contract().values / 2 ** len(eu.get_variables(self.formulaDict[formulaKey][0]))

    def calculate_empRates(self):
        for formulaKey in self.formulaDict:
            self.formulaDict[formulaKey][2] = coc.CoreContractor(
                {**self.formulaTensors.get_cores([formulaKey], headType="truthEvaluation"),
                 **self.dataTensor.get_cores()}).contract().values / self.dataTensor.dataNum

    def independent_estimation(self, calculateEmp=True, calculateSat=True, cut=100):
        if calculateEmp:
            self.calculate_empRates()
        if calculateSat:
            self.calculate_independent_satRates()

        self.formulaTensors.update_heads({formulaKey: cutoff_weight(
            solve_rate_equation(self.formulaDict[formulaKey][1], self.formulaDict[formulaKey][2]), cut) for formulaKey
            in self.formulaDict})

        for formulaKey in self.formulaDict:
            self.formulaDict[formulaKey][3] = cutoff_weight(
                solve_rate_equation(self.formulaDict[formulaKey][1], self.formulaDict[formulaKey][2]), cut)

    def alternating_optimization(self, sweepNum):
        weightTracker = np.empty((sweepNum + 1, len(self.formulaDict)))
        self.independent_estimation()

        weightTracker[0] = [self.formulaDict[formulaKey][3] for formulaKey in self.formulaDict]
        for sweepPos in range(sweepNum):
            for i, formulaKey in enumerate(self.formulaDict):
                self.formula_optimization(formulaKey)
                weightTracker[sweepPos + 1, i] = self.formulaDict[formulaKey][3]
        return weightTracker

    def formula_optimization(self, tboFormulaKey, maxWeight=100):
        conDict = self.formulaTensors.get_cores(headType="expFactor")

        oldLength = len(conDict)
        conDict = {key: conDict[key] for key in conDict if key != tboFormulaKey + "_head"}
        newLength = len(conDict)

        if newLength != oldLength - 1:
            print(conDict.keys(), tboFormulaKey + "_head")
            raise ValueError("HeadCore of formula {} not found!".format(tboFormulaKey))

        negativeExpWeight, positiveExpWeight = coc.CoreContractor(conDict, openColors=[
            tboFormulaKey + "_" + str(self.formulaDict[tboFormulaKey][0])]).contract().values

        if positiveExpWeight == 0 or negativeExpWeight ==0:
            self.formulaDict[tboFormulaKey][3] = 0
        else:
            negPosQuotient = negativeExpWeight / positiveExpWeight
            empRate = self.formulaDict[tboFormulaKey][2]
            self.formulaDict[tboFormulaKey][3] = min(np.log(negPosQuotient * (empRate / (1 - empRate))), maxWeight)

        self.formulaTensors.update_heads({tboFormulaKey: self.formulaDict[tboFormulaKey][3]})

    def get_weights(self):
        return {key: self.formulaDict[key][3] for key in self.formulaDict}

def calculate_satRate_bc(expression):
    return ee.ExpressionEvaluator(expression, initializeBasisCores=True).create_formula_factor().count_satisfaction()


def solve_rate_equation(satRate, empRate):
    return -np.log(((1 - empRate) * satRate) / (empRate * (1 - satRate)))


def regularize_empRate(empRate, regFactor):
    return regFactor * (empRate - 0.5) + 0.5


def cutoff_weight(weight, cutoff):
    if weight > cutoff:
        return cutoff
    else:
        return weight


## Old, used only in SampleBasedMLNLearner
def calculate_weight(expression, sampleDf, filterCore=None, regFactor=1, verbose=False, check=True, cut=20):
    atoms = eu.get_variables(expression)
    atomDict = {
        atom: cc.CoordinateCore(stoc.sampleDf_to_universal_core(sampleDf, [atom]).flatten(), ["j"])
        for atom in atoms
    }
    satRate = calculate_satRate(expression)

    if check:
        assert satRate == calculate_satRate_bc(expression), "Saturation Rate does not coincide with Basis Calculus!"

    empRate = regularize_empRate(calculate_empRate(expression, atomDict, filterCore), regFactor)
    weight = cutoff_weight(solve_rate_equation(satRate, empRate), cut)
    if verbose:
        print("## Calculationg the weight of {} ##".format(expression))
        print("World satisfaction rate: {}".format(satRate))
        print("Data satifcation rate: {}".format(empRate))
        print("Calculated weight: {}".format(weight))
    return empRate, satRate, weight


def calculate_empRate(expression, atomDict, filterCore=None):
    expressionCore = ee.ExpressionEvaluator(expression, atomDict=atomDict).evaluate()
    if filterCore is None:
        expressionResults = expressionCore.values.flatten()
        empSatNum = np.sum(expressionResults)
        dataNum = expressionResults.shape[0]
    else:
        expressionCore = expressionCore.compute_and(filterCore)
        empSatNum = np.sum(expressionCore.values)
        dataNum = np.sum(filterCore.values)
    return empSatNum / dataNum


def partition_function(satRate, w):
    return (satRate * np.exp(w)) / (1 - satRate + satRate * np.exp(w))


def calculate_satRate(expression):
    variables = np.unique(eu.get_variables(expression))
    modelNum = 2 ** len(variables)
    satNum = sc.count_satisfaction(expression)
    return satNum / modelNum


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    from matplotlib import pyplot as plt

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    expressionList = [["b", "and", ["a", "and", ["not", "c"]]],
                      "a",
                      ["not", ["b", "and", "c"]]]

    # expressionList = ["a", "b", "c"] # Needs to be constant in that case

    from tnreason.model import generate_test_data as gtd

    sampleDf = gtd.generate_sampleDf({
        "e1": ["a", 2],
        "e2": ["b", 2],
        "e3": ["c", 2],
    },
        100)

    estimator = WeightEstimator(expressionList, sampleDf=sampleDf)
    # estimator.add_formula("e4", ["a", "and", ["not", "b"]], 1)
    #    estimator.independent_estimation(atomDict)
    weightTracker = estimator.alternating_optimization(10)

    plt.imshow(weightTracker, cmap="coolwarm")
    plt.colorbar()
    plt.show()
    exit()
    ## Independent Calculation (Old)
    expression = ["b", "and", ["a", "and", ["not", "c"]]]

    filterCore = cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 10, 10, 7, 7, 5)),
                                   ["l1", "l2", "l3", "y", "q", "z"], name="c")

    checkEmpRate = calculate_empRate(expression, atomDict, filterCore)
    checkSatRate = calculate_satRate(expression)
    result = calculate_weight(expression, atomDict, filterCore)

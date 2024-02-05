from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft

from tnreason.contraction import core_contractor as coc

from tnreason import knowledge

import numpy as np


class EntropyMaximizer:
    def __init__(self, formulaList=[], formulaDict=None,
                 satisfactionDict={},
                 factDict={}):
        if formulaDict is not None:
            self.formulaDict = formulaDict
        else:
            self.formulaDict = {
                "f" + str(i): [formula, 0] for i, formula in enumerate(formulaList)
            }
        self.formulaTensors = tm.TensorRepresentation(self.formulaDict)

        self.factDict = factDict
        self.factTensors = tm.TensorRepresentation({key: [factDict[key], None] for key in factDict},
                                                   headType="truthEvaluation")
        self.satisfactionDict = satisfactionDict

    def to_hybrid_kb(self):
        return knowledge.HybridKnowledgeBase(weightedFormulasDict=self.formulaDict,
                                             factsDict=self.factDict)

    def add_formula(self, expression, empRate=None, weight=0, key=None, isFact=False):
        if key is None:
            key = "f" + str(len(self.formulaDict.keys()) + len(self.factDict.keys()))
        if isFact:
            self.factDict[key] = expression
            self.factTensors.add_expression(expression, weight, key)
        else:
            self.formulaDict[key] = [expression, weight]
            self.formulaTensors.add_expression(expression, weight, key)

        self.satisfactionDict[key] = empRate

    def drop_formula(self, formulaKey):
        if formulaKey in self.factDict:
            self.factDict.pop(formulaKey)
            self.factTensors.drop_expression(formulaKey)
        elif formulaKey in self.formulaDict:
            self.formulaDict.pop(formulaKey)
            self.formulaTensors.drop_expression(formulaKey)

    def get_weights(self, formulaKeys=None):
        if formulaKeys is None:
            formulaKeys = self.formulaDict.keys()
        return {formulaKey: self.formulaDict[formulaKey][1] for formulaKey in formulaKeys}

    def fact_identification(self, thresholdWeight=100):
        for key in self.formulaDict.copy():
            if self.formulaDict[key][1] >= thresholdWeight:
                formula, weight = self.formulaDict.pop(key)
                self.formulaTensors.drop_expression(key)

                self.factDict[key] = formula
                self.factTensors.add_expression(formula, weight=None, formulaKey=key)

    def soften_facts(self, softWeight=100):
        for key in self.factDict.copy():
            fact = self.factDict.pop(key)
            self.factTensors.drop_expression(key)

            self.formulaDict[key] = [fact, softWeight]
            self.factTensors.add_expression(fact, weight=softWeight, formulaKey=key)

    def independent_estimation(self):
        ## Reset the weights to zero and do one optimization sweep
        self.formulaDict = {key: [self.formulaDict[key], 0] for key in self.formulaDict}
        for key in self.formulaDict:
            self.formula_optimization(key)

    def alternating_optimization(self, sweepNum=10):
        weightTracker = np.empty((sweepNum + 1, len(self.formulaDict)))

        weightTracker[0] = [self.formulaDict[formulaKey][1] for formulaKey in self.formulaDict]
        for sweepPos in range(sweepNum):
            for i, formulaKey in enumerate(self.formulaDict):
                self.formula_optimization(formulaKey)
                weightTracker[sweepPos + 1, i] = self.formulaDict[formulaKey][1]
        return weightTracker

    def formula_optimization(self, tboFormulaKey, maxWeight=100):
        hardContractionDict = self.formulaTensors.get_cores(headType="expFactor")
        assert tboFormulaKey + "_head" in hardContractionDict.keys(), \
            "Head Core of Formula {} to be optimized not found in Tensor Representation.".format(tboFormulaKey)
        contractionDict = {
            **{key: hardContractionDict[key] for key in hardContractionDict if key != tboFormulaKey + "_head"},
            **self.factTensors.get_cores(headType="truthEvaluation")
        }

        negativeProbability, positiveProbability = coc.CoreContractor(
            contractionDict,
            openColors=[tboFormulaKey + "_" + str(self.formulaDict[tboFormulaKey][0])]
        ).contract().values

        empRate = self.satisfactionDict[tboFormulaKey]

        if positiveProbability == 0 or negativeProbability == 0:
            ## Formula is entailed or contradicting, and should not be added from a KB perspective!
            self.formulaDict[tboFormulaKey][1] = 0
        elif empRate == 1:
            # Formulas which are everywhere true, will get maximum weight
            self.formulaDict[tboFormulaKey][1] = maxWeight
        elif empRate == 0:
            # Formulas which are nowhere true, should be negated and added as facts!
            # Here: Ignore them
            self.formulaDict[tboFormulaKey][1] = 0
        else:
            self.formulaDict[tboFormulaKey][1] = min(
                np.log((negativeProbability / positiveProbability) * (empRate / (1 - empRate))), maxWeight)

        self.formulaTensors.update_heads({tboFormulaKey: self.formulaDict[tboFormulaKey][1]})


class EmpiricalCounter:
    def __init__(self, sampleDf):
        self.dataTensor = ft.DataTensor(sampleDf)

    def get_empirical_satisfaction(self, formula):
        return coc.CoreContractor({
            **self.dataTensor.get_cores(),
            **ft.FormulaTensor(formula,
                               headType="truthEvaluation").get_cores()}).contract().values / self.dataTensor.dataNum

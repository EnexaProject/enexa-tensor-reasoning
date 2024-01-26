from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft

from tnreason.contraction import core_contractor as coc

import numpy as np


class EntropyMaximizer:
    def __init__(self, formulaList=[], formulaDict=None,
                 satisfactionDict={}, sampleDf=None,
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

        if sampleDf is not None:
            self.calculate_satisfaction(sampleDf)
        else:
            self.satisfactionDict = satisfactionDict

    def calculate_satisfaction(self, sampleDf):
        dataTensor = ft.DataTensor(sampleDf)
        self.satisfactionDict = {
            key: calculate_satisfaction(dataTensor, self.formulaDict[key][0])
            for key in self.formulaDict
        }

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

        weightTracker[0] = [self.formulaDict[formulaKey][3] for formulaKey in self.formulaDict]
        for sweepPos in range(sweepNum):
            for i, formulaKey in enumerate(self.formulaDict):
                self.formula_optimization(formulaKey)
                weightTracker[sweepPos + 1, i] = self.formulaDict[formulaKey][3]
        return weightTracker

    def formula_optimization(self, tboFormulaKey, maxWeight=100):
        empRate = self.satisfactionDict[tboFormulaKey]
        if empRate == 1:
            self.formulaDict[tboFormulaKey][1] = maxWeight
        elif empRate == 0:
            self.formulaDict[tboFormulaKey][1] = 0
        else:
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

            if positiveProbability == 0 or negativeProbability == 0:
                ## Formula is entailed or contradicting, and should not be added from a KB perspective!
                self.formulaDict[tboFormulaKey][1] = 0
            else:
                self.formulaDict[tboFormulaKey][1] = min(
                    np.log((negativeProbability / positiveProbability) * (empRate / (1 - empRate))), maxWeight)

        self.formulaTensors.update_heads({tboFormulaKey: self.formulaDict[tboFormulaKey][1]})


def calculate_satisfaction(dataTensor, formula):
    return coc.CoreContractor({**dataTensor.get_cores(),
                               **ft.FormulaTensor(formula,
                                                  headType="truthEvaluation").get_cores()}).contract().values / dataTensor.dataNum
from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft

from tnreason.logic import expression_utils as eu
from tnreason.logic import expression_generation as eg

from tnreason.contraction import core_contractor as coc

import numpy as np


class FormulaSamplingBase:
    def __init__(self, skeletonExpression, candidatesDict, knownFormulasDict={}, knownFactsDict={}, sampleDf=None):
        self.skeletonExpression = skeletonExpression
        self.placeHolders = candidatesDict.keys()
        self.candidatesDict = candidatesDict

        self.formulaTensors = tm.TensorRepresentation(knownFormulasDict, knownFactsDict, headType="expFactor")
        self.assignment = self.uniform_sample()

        if sampleDf is not None:
            self.load_sampleDf(sampleDf)

    def load_sampleDf(self, sampleDf):
        self.dataTensor = ft.DataTensor(sampleDf)

    def get_result(self):
        return eg.replace_atoms(self.skeletonExpression, self.assignment)

    def uniform_sample(self, placeHolders=None):
        if placeHolders is None:
            placeHolders = self.placeHolders
        return {phKey: np.random.choice(self.candidatesDict[phKey]) for phKey in placeHolders}

    def compute_local_probability(self, tboPlaceholderKey, temperature=1):
        self.assignment.pop(tboPlaceholderKey)

        probabilities = np.empty(shape=len(self.candidatesDict[tboPlaceholderKey]))
        for i, candidate in enumerate(self.candidatesDict[tboPlaceholderKey]):
            testFormula = eg.replace_atoms(self.skeletonExpression, {**self.assignment, tboPlaceholderKey: candidate})
            probabilities[i] = self.compute_formula_probability(testFormula) ** temperature

        if np.linalg.norm(probabilities) > 0:
            return probabilities / np.sum(probabilities)
        else:
            return [1 / len(probabilities) for i in range(len(probabilities))]

    def compute_formula_probability(self, formula, verbose=False):
        candidateTensor = ft.FormulaTensor(
            formula,
            headType="truthEvaluation")

        dataTerm = coc.CoreContractor({
            **self.formulaTensors.get_cores(),
            **self.dataTensor.get_cores(),
            **candidateTensor.get_cores()
        }).contract().values / self.dataTensor.dataNum

        partition = coc.CoreContractor({
            **self.formulaTensors.get_cores(),
            **candidateTensor.get_cores()
        }).contract().values

        if dataTerm == 0 and verbose:
            print("Formula {} nether true!".format(candidateTensor.expression))

        return dataTerm / partition


class GibbsFormulaSampler(FormulaSamplingBase):
    def gibbs_step(self, key, temperature=1):
        localProb = self.compute_local_probability(key, temperature=temperature)
        self.assignment[key] = self.candidatesDict[key][np.where(np.random.multinomial(1, localProb) == 1)[0][0]]

    def gibbs(self, chainSize=10):
        for chainPos in range(chainSize):
            for key in self.placeHolders:
                self.gibbs_step(key)

    def gibbs_simulated_annealing(self, temperaturePattern, restart=True):
        if restart:
            self.assignment = self.uniform_sample()
        ## temperaturePattern : List of tuples (chainSize, temperature)
        for chainSize, temperature in temperaturePattern:
            for chainPos in range(chainSize):
                for key in self.placeHolders:
                    self.gibbs_step(key, temperature=temperature)

from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft

from tnreason.logic import expression_utils as eu
from tnreason.logic import expression_generation as eg

from tnreason.contraction import core_contractor as coc

import numpy as np


class FormulaSamplingBase:
    def __init__(self, skeletonExpression, candidatesDict, knownFormulaDict={}, knownFactDict={}, sampleDf = None):
        self.skeletonExpression = skeletonExpression
        self.placeHolders = eu.get_variables(skeletonExpression)
        self.candidatesDict = candidatesDict

        self.formulaTensors = tm.TensorRepresentation(knownFormulaDict, headType="expFactor")
        self.factTensors = tm.TensorRepresentation({key: [knownFactDict[key], None] for key in knownFactDict},
                                                   headType="truthEvaluation")
        self.assignment = self.uniform_sample()

        if sampleDf is not None:
            self.load_sampleDf(sampleDf)

    def load_sampleDf(self, sampleDf):
        affectedAtoms = set()
        for placeHolder in self.placeHolders:
            affectedAtoms.update(self.candidatesDict[placeHolder])

        self.dataTensor = ft.DataTensor(sampleDf[affectedAtoms])

    def uniform_sample(self, placeHolders=None):
        if placeHolders is None:
            placeHolders = self.placeHolders
        return {phKey: np.random.choice(self.candidatesDict[phKey]) for phKey in placeHolders}

    def compute_local_probability(self, tboPlaceholderKey, temperature=1):
        self.assignment.pop(tboPlaceholderKey)

        probabilities = np.empty(shape=len(self.candidatesDict[tboPlaceholderKey]))
        for i, candidate in enumerate(self.candidatesDict[tboPlaceholderKey]):
            candidateTensor = ft.FormulaTensor(eg.replace_atoms(self.skeletonExpression, {**self.assignment, tboPlaceholderKey: candidate}), headType="truthEvaluation")

            dataTerm = coc.CoreContractor({
                **self.dataTensor.get_cores(),
                **candidateTensor.get_cores()
            }).contract().values / self.dataTensor.dataNum

            partition = coc.CoreContractor({
                **self.formulaTensors.get_cores(),
                **self.factTensors.get_cores(),
                **candidateTensor.get_cores()
            }).contract().values

            probabilities[i] = (dataTerm / partition) ** temperature
            if dataTerm == 0:
                print("Formula {} nether true!".format(candidateTensor.expression))

        if np.linalg.norm(probabilities)>0:
            return probabilities/np.sum(probabilities)
        else:
            return [1/len(probabilities) for i in range(len(probabilities))]

class GibbsFormulaSampler(FormulaSamplingBase):
    def gibbs_step(self, key):
        localProb = self.compute_local_probability(key, temperature=1)
        self.assignment[key] = self.candidatesDict[key][np.random.multinomial(1, localProb)[0]]

    def gibbs(self, chainSize=10):
        for chainPos in range(chainSize):
            for key in self.placeHolders:
                self.gibbs_step(key)



if __name__ == "__main__":
    skeleton = ["P0", "and", "P1"]
    candidatesDict = {
        "P0": ["sikorka", "sledz"],
        "P1": ["jaszczur", "szczeniak", "piskle"]
    }

    from tnreason.model import generate_test_data as gtd
    sampleDf = gtd.generate_sampleDf({
        "f1": [["sikorka", "and", ["not","piskle"]], 20],
        "f2": [["sledz", "and", ["not","szczeniak"]], 20],
        "f3": [["jaszczur", "and", "sikorka"], 20],
    }, 10)

    fSampler = GibbsFormulaSampler(skeleton, candidatesDict, sampleDf=sampleDf)
    print(fSampler.assignment)

    fSampler.gibbs(10)
    print(fSampler.assignment)

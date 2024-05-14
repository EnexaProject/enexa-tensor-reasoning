from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

import pandas as pd

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


class InferenceProvider:
    """
    Answering queries on a distribution by contracting its cores.
    """

    def __init__(self, distribution, contractionMethod=engine.defaultContractionMethod):
        """
        * distribution: Needs to support create_cores() and get_partition_function()
        """
        self.distribution = distribution
        self.contractionMethod = contractionMethod

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return entailedString
        elif probability == 0:
            return contradictingString
        else:
            return contingentString

    def ask(self, queryFormula, evidenceDict={}):

        contracted = engine.contract(
            coreDict={
                **self.distribution.create_cores(),
                **encoding.create_evidence_cores(evidenceDict),
                **encoding.create_raw_formula_cores(queryFormula)
            },
            method=self.contractionMethod, openColors=[encoding.get_formula_color(queryFormula)]).values

        return contracted[1] / (contracted[0] + contracted[1])

    def query(self, variableList, evidenceDict={}):
        return engine.contract(method=self.contractionMethod, coreDict={
            **self.distribution.create_cores(),
            **encoding.create_trivial_cores([variable for variable in variableList if
                                             variable not in self.distribution.atoms and variable not in evidenceDict]),
            **encoding.create_evidence_cores(evidenceDict),
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = distributionCore.get_maximal_index()
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_sample(self, variableList, annealingPattern=[(10, 1)]):
        sampler = algorithms.Gibbs(self.distribution.create_cores())

        sampler.ones_initialization(updateKeys=variableList, shapesDict={variable: 2 for variable in variableList},
                                    colorsDict={variable: [variable] for variable in variableList})

        return sampler.annealed_sample(updateKeys=variableList, annealingPattern=annealingPattern)

    def draw_samples(self, sampleNum, variableList=None, annealingPattern=[(10, 1)], outType="int64"):
        if variableList is None:
            variableList = self.distribution.atoms
        sampleDf = pd.DataFrame(columns=variableList)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf,
                 pd.DataFrame(self.annealed_sample(variableList=variableList, annealingPattern=annealingPattern),
                              index=[samplePos])])
        return sampleDf.astype(outType)

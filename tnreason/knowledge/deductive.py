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
            **encoding.create_evidence_cores(evidenceDict),
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        return distributionCore.get_argmax()

    def forward_sample(self, variableList, dimDict={}):
        return algorithms.ForwardSampler(self.distribution.create_cores(), dimDict=dimDict).draw_forward_sample(
            variableList)

    def draw_samples(self, sampleNum, variableList=None, outType="int64"):
        if variableList is None:
            variableList = self.distribution.atoms
        sampleDf = pd.DataFrame(columns=variableList)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf,
                 pd.DataFrame(self.forward_sample(variableList=variableList),
                              index=[samplePos])])
        return sampleDf.astype(outType)

    ## Energy-based: Creates always a full sample!
    # Can be used for sample (temperatures converging to 1) or for maximum search (temperature converging to 0)
    ## TO DO: Add support for evidenceDict
    def energy_based_sample(self, method, temperatureList=[1 for i in range(10)]):
        return algorithms.optimize_energy(energyDict={**self.distribution.get_energy_dict()},
                                          colors=list(self.distribution.get_dimension_dict().keys()),
                                          dimDict=self.distribution.get_dimension_dict(),
                                          method=method,
                                          temperatureList=temperatureList)

from tnreason import encoding
from tnreason import engine

class EmpiricalDistribution:
    def __init__(self, sampleDf, atomKeys=None):
        self.create_from_sampleDf(sampleDf, atomKeys)

    def create_from_sampleDf(self, sampleDf, atomKeys=None):
        if atomKeys is None:
            atomKeys = list(sampleDf.columns)
        self.dataCores = encoding.create_data_cores(sampleDf, atomKeys)
        self.dataNum = sampleDf.values.shape[0]

    def get_empirical_satisfaction(self, expression):
        return engine.contract(method="NumpyEinsum",
                               coreDict={**self.dataCores, **encoding.create_raw_formula_cores(expression)},
                               openColors=[encoding.get_formula_color(expression)]).values[1] / self.dataNum

    def get_satisfactionDict(self, expressionsDict):
        return {key: self.get_empirical_satisfaction(expressionsDict[key]) for key in expressionsDict}

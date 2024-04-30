from tnreason import encoding
from tnreason import engine


class EmpiricalDistribution:
    def __init__(self, sampleDf, atomKeys=None):
        if atomKeys is None:
            atomKeys = list(sampleDf.columns)
        self.sampleDf = sampleDf
        self.dataNum = sampleDf.values.shape[0]
        self.atoms = atomKeys

    def create_cores(self):
        return encoding.create_data_cores(self.sampleDf, self.atoms)

    def get_empirical_satisfaction(self, expression):
        return engine.contract(method="NumpyEinsum",
                               coreDict={**self.create_cores(), **encoding.create_raw_formula_cores(expression)},
                               openColors=[encoding.get_formula_color(expression)]).values[1] / (
            self.get_partition_function(encoding.get_variables(expression)))

    def get_satisfactionDict(self, expressionsDict):
        return {key: self.get_empirical_satisfaction(expressionsDict[key]) for key in expressionsDict}

    def get_partition_function(self, allAtoms):
        unseenAtomNum = len([atom for atom in allAtoms if atom not in self.atoms])
        return (self.dataNum * (2 ** unseenAtomNum))

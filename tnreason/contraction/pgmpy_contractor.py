from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination

from tnreason.logic import coordinate_calculus as cc


class PgmpyVariableEliminator:
    def __init__(self, coreDict={}, openColors=[]):
        self.model = MarkovNetwork()
        self.add_factors_from_coresDict(coreDict)

        self.openColors = openColors

    def add_factors_from_coresDict(self, coresDict):
        for coreKey in coresDict:
            for col1 in coresDict[coreKey].colors:
                self.model.add_node(col1)
                for col2 in coresDict[coreKey].colors:
                    if col2 != col1:
                        self.model.add_edge(col1, col2)
            self.model.add_factors(
                DiscreteFactor(coresDict[coreKey].colors, coresDict[coreKey].values.shape, coresDict[coreKey].values)
            )

    def contract(self):
        result = VariableElimination(self.model).query(evidence={}, variables=self.openColors)
        return cc.CoordinateCore(result.values, result.variables)

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination

from tnreason.engine import numpy_contractor as cor
from tnreason.engine import engine_visualization as cv


class PgmpyVariableEliminator:
    def __init__(self, coreDict={}, openColors=[], visualize=False):
        if visualize:
            self.visualize(coreDict)

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
        return cor.NumpyCore(result.values, result.variables)

    def visualize(self, coreDict):
        cv.draw_contractionDiagram(coreDict)
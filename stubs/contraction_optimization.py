from tnreason import engine

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']

class ContractionOptimization:
    """
    To manipulate the contraction order and use the other contractors as atomic providers.
    """

    def __init__(self, coreDict={}, openColors=[], variableNestedList=None,
                 coreType="NumpyTensorCore"):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

        self.variableNestedList = variableNestedList
        self.coreType = coreType

    def create_naive_variableNestedList(self, together=True):
        closedColors = []
        for key in self.coreDict:
            for color in self.coreDict[key].colors:
                if color not in closedColors and color not in self.openColors:
                    closedColors.append(color)
        if together:
            self.variableNestedList = [closedColors]
        else:
            self.variableNestedList = [[color] for color in closedColors]

    def recursive_contraction(self, method="NumpyEinsum"):
        for variables in self.variableNestedList:
            self.contraction_step(variables, method=method)

    def contraction_step(self, variables, method):
        affectedKeys = [key for key in self.coreDict if not set(self.coreDict[key].colors).isdisjoint(set(variables))]
        contractionCores = {key: self.coreDict.pop(key) for key in affectedKeys}

        openVariables = []
        for key in contractionCores:
            for color in contractionCores[key].colors:
                if color not in variables and color not in openVariables:
                    openVariables.append(color)

        self.coreDict["contracted_" + str(variables)] = engine.contract(method=method, coreDict=contractionCores,
                                                                        openColors=openVariables)
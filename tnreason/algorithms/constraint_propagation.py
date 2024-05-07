from tnreason import engine

import numpy as np

from queue import Queue

domainCoreSuffix = "_domainCore"


class ConstraintPropagator:
    """
    Updates binary domain cores based on local contractions.
    """
    def __init__(self, binaryCoresDict, domainCoresDict=None, verbose=False):
        self.verbose = verbose
        self.binaryCoresDict = binaryCoresDict

        if domainCoresDict is None:
            self.initialize_domainCoresDict()
        else:
            self.domainCoresDict = domainCoresDict

        self.coreQueue = Queue()
        self.create_affectionDict()
        self.assignments = None

    def initialize_domainCoresDict(self):
        self.domainCoresDict = {}
        for coreKey in self.binaryCoresDict:
            for i, color in enumerate(self.binaryCoresDict[coreKey].colors):
                if color + domainCoreSuffix not in self.domainCoresDict:
                    self.domainCoresDict[color + domainCoreSuffix] = engine.get_core()(
                        np.ones(self.binaryCoresDict[coreKey].values.shape[i]),
                        [color],
                        color + domainCoreSuffix)

    def create_affectionDict(self):
        self.colorAffectionDict = {}
        for coreKey in self.binaryCoresDict:
            for color in self.binaryCoresDict[coreKey].colors:
                if color in self.colorAffectionDict:
                    self.colorAffectionDict[color].append(coreKey)
                else:
                    self.colorAffectionDict[color] = [coreKey]

    def propagate_cores(self, coreOrder=None):
        if coreOrder is None:
            coreOrder = list(self.binaryCoresDict.keys())
        for coreKey in coreOrder:
            self.coreQueue.put(coreKey)
        while not self.coreQueue.empty():
            self.propagation_step(self.coreQueue.get())

    def propagation_step(self, coreKey):
        if self.verbose:
            print("Propagating core {}.".format(coreKey))
        changedColors = []
        for color in self.binaryCoresDict[coreKey].colors:
            contracted = engine.contract(coreDict=
                                         {coreKey: self.binaryCoresDict[coreKey],
                                          **{otherColor + domainCoreSuffix: self.domainCoresDict[
                                              otherColor + domainCoreSuffix] for otherColor in
                                             self.binaryCoresDict[coreKey].colors if otherColor != color}},
                                         openColors=[color]
                                         ).values
            for i in range(len(contracted)):
                if contracted[i] == 0 and self.domainCoresDict[color + domainCoreSuffix].values[i] == 1:
                    self.domainCoresDict[color + domainCoreSuffix].values[i] = 0
                    if color not in changedColors:
                        changedColors.append(color)
                    if np.sum(self.domainCoresDict[color + domainCoreSuffix].values) == 0:
                        raise ValueError(
                            "Inconsistent Knowledge Base: Color {} has no possible assignments!".format(color))

        for changedColor in changedColors:
            for key in self.colorAffectionDict[changedColor]:
                if key not in list(self.coreQueue.queue) and key != coreKey:
                    self.coreQueue.put(key)

    def find_assignments(self):
        self.assignments = {
            self.domainCoresDict[key].colors[0]: np.where(self.domainCoresDict[key].values == 1)[0][0]
            for key in self.domainCoresDict if np.sum(self.domainCoresDict[key].values) == 1}
        return self.assignments

    def find_redundant_cores(self):
        if self.assignments is None:
            self.find_assignments()
        return {key for key in self.binaryCoresDict if
                all([color in self.assignments for color in self.binaryCoresDict[key].colors])}

    def find_variable_cone(self, variables,
                           variableShapes={}):  ## Add variables to domainCoreDict when they are not there!
        for variable in variables:
            if variable + domainCoreSuffix not in self.domainCoresDict:
                self.domainCoresDict[variable + domainCoreSuffix] = engine.get_core()(
                    np.ones(variableShapes[variable]), [variable], variable + domainCoreSuffix)
        variablesQueue = Queue()
        for variable in variables:
            variablesQueue.put(variable)
        coneCores = {}
        while not variablesQueue.empty():
            variable = variablesQueue.get()
            coneCores[variable + domainCoreSuffix] = self.domainCoresDict[variable + domainCoreSuffix]
            if np.sum(self.domainCoresDict[variable + domainCoreSuffix].values) == 1:
                pass
            else:
                for key in self.binaryCoresDict:
                    if variable in self.binaryCoresDict[key].colors:
                        coneCores[key] = self.binaryCoresDict[key]
                        for color in self.binaryCoresDict[key].colors:
                            if color != variable and color + domainCoreSuffix not in coneCores:
                                coneCores[color + domainCoreSuffix] = self.domainCoresDict[color + domainCoreSuffix]
        return coneCores

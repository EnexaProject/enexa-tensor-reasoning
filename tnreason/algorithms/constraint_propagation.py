from tnreason import engine

import numpy as np

from queue import Queue

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"

domainCoreSuffix = "_domainCore"


class ConstraintPropagator:
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
                    self.domainCoresDict[color + domainCoreSuffix] = engine.get_core(coreType=defaultCoreType)(
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
            contracted = engine.contract(method=defaultContractionMethod, coreDict=
            {coreKey: self.binaryCoresDict[coreKey],
             **{otherColor + domainCoreSuffix: self.domainCoresDict[otherColor + domainCoreSuffix] for otherColor in
                self.binaryCoresDict[coreKey].colors if otherColor != color}},
                                         openColors=[color]
                                         ).values
            for i in range(len(contracted)):
                if contracted[i] == 0 and self.domainCoresDict[color + domainCoreSuffix].values[i] == 1:
                    self.domainCoresDict[color + domainCoreSuffix].values[i] = 0
                    if color not in changedColors:
                        changedColors.append(color)

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

    def find_evidence_and_redundant_cores(self):
        evidenceDict = {}
        multipleAssignmentColors = []
        for colorKey in self.domainCoresDict:
            color = colorKey[:-11]
            if color not in evidenceDict and color not in multipleAssignmentColors:
                assignmentsNum = np.sum(self.domainCoresDict[colorKey].values)
                if assignmentsNum == 0:
                    raise ValueError("Inconsistent Knowledge Base: Color {} has no possible assignments!".format(color))
                elif assignmentsNum == 1:
                    evidenceDict[color] = np.where(self.domainCoresDict[colorKey].values == 1)[0][0]
                    if self.verbose:
                        print("Color {} has only possible assignment {}.".format(color,
                                                                                 evidenceDict[color]))
                else:
                    multipleAssignmentColors.append(color)
                    if self.verbose:
                        print("Color {} has multiple assignments.".format(color))
        redundantCores = [coreKey for coreKey in list(self.binaryCoresDict.keys()) if
                          all(color in evidenceDict for color in self.binaryCoresDict[coreKey].colors)]
        remainingCores = [coreKey for coreKey in list(self.binaryCoresDict.keys()) if coreKey not in redundantCores]
        return evidenceDict, multipleAssignmentColors, \
            redundantCores, remainingCores

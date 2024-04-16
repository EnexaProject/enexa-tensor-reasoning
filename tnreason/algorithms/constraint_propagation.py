from tnreason import engine

import numpy as np

from queue import Queue

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"


class ConstraintPropagator:
    def __init__(self, binaryCoresDict, domainCoresDict=None, verbose=True):
        self.verbose = verbose
        self.binaryCoresDict = binaryCoresDict

        if domainCoresDict is None:
            self.initialize_domainCoresDict()
        else:
            self.domainCoresDict = domainCoresDict

        self.coreQueue = Queue()
        self.create_affectionDict()

    def initialize_domainCoresDict(self):
        self.domainCoresDict = {}
        for coreKey in self.binaryCoresDict:
            for i, color in enumerate(self.binaryCoresDict[coreKey].colors):
                if color + "_domainCore" not in self.domainCoresDict:
                    self.domainCoresDict[color + "_domainCore"] = engine.get_core(coreType=defaultCoreType)(
                        np.ones(self.binaryCoresDict[coreKey].values.shape[i]),
                        [color],
                        color + "_domainCore")

    def create_affectionDict(self):
        self.colorAffectionDict = {}
        for coreKey in self.binaryCoresDict:
            for color in self.binaryCoresDict[coreKey].colors:
                if color in self.colorAffectionDict:
                    self.colorAffectionDict[color].append(coreKey)
                else:
                    self.colorAffectionDict[color] = [coreKey]

    def propagate_cores(self):
        for coreKey in list(self.binaryCoresDict.keys()):
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
             **{otherColor + "_domainCore": self.domainCoresDict[otherColor + "_domainCore"] for otherColor in
                self.binaryCoresDict[coreKey].colors if otherColor != color}},
                                         openColors=[color]
                                         ).values
            colorChanged = False
            for i in range(len(contracted)):
                if contracted[i] == 0 and self.domainCoresDict[color + "_domainCore"].values[i] == 1:
                    self.domainCoresDict[color + "_domainCore"].values[i] = 0
                    colorChanged = True
            if colorChanged:
                changedColors.append(color)

        for changedColor in changedColors:
            for key in self.colorAffectionDict[changedColor]:
                if key not in list(self.coreQueue.queue) and key != coreKey:
                    self.coreQueue.put(key)

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

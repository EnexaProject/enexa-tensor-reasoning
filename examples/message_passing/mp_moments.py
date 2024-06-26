from tnreason import encoding
from tnreason import engine

import numpy as np

messageCoreSuffix = "_messageCore"
headCoreSuffix = "_headCore"  ## Should be same as in encoding


class MPMomentMatcher:
    def __init__(self, expressionsDict, empircalMeanDict):
        self.expressionsDict = expressionsDict
        self.empiricalMeanDict = empircalMeanDict

        self.atoms = encoding.get_all_atoms(expressionsDict)
        self.messageCores = {
            atom + messageCoreSuffix: encoding.create_trivial_core(atom + messageCoreSuffix, [2], [atom])
            for atom in self.atoms}

        self.directedCores = dict()
        for key in expressionsDict:
            self.directedCores.update(encoding.create_raw_formula_cores(expressionsDict[key]))

        self.headCores = dict()
        for key in expressionsDict:
            self.headCores.update(encoding.create_head_core(expressionsDict[key], "expFactor", 0))

        # To controll convergence
        self.weightsDict = {key: [] for key in expressionsDict}

    def message_passing(self, circulationNumber=10):
        for circPos in range(circulationNumber):
            self.upward_messages()
            self.adjust_headCores()
            self.downward_messages()

    def upward_messages(self):
        readyColors = self.atoms.copy()
        weightingCores = set(self.directedCores.keys()).copy()
        readyCores = find_ready_cores_in_direction(weightingCores, self.directedCores, readyColors)

        while len(readyCores) != 0:
            updateCoreKey = readyCores.pop()
            weightingCores.remove(updateCoreKey)

            if len(self.directedCores[updateCoreKey].colors) == 3:
                inColors = self.directedCores[updateCoreKey].colors[:1]
                outColor = self.directedCores[updateCoreKey].colors[2]
            elif len(self.directedCores[updateCoreKey].colors) == 2:
                inColors = [self.directedCores[updateCoreKey].colors[0]]
                outColor = self.directedCores[updateCoreKey].colors[1]

            self.messageCores[outColor + messageCoreSuffix] = engine.contract(
                {**{inColor + messageCoreSuffix: self.messageCores[inColor + messageCoreSuffix] for inColor in
                    inColors},
                 updateCoreKey: self.directedCores[updateCoreKey]
                 }, openColors=[outColor]).normalize()

            readyColors.append(outColor)

            if len(readyCores) == 0:
                readyCores = find_ready_cores_in_direction(weightingCores, self.directedCores, readyColors)

    def adjust_headCores(self, maxWeight=100):
        for key in self.expressionsDict:
            headColor = encoding.get_formula_color(expressionsDict[key])
            negValue, posValue = self.messageCores[headColor + messageCoreSuffix].values
            if negValue == 0 or self.empiricalMeanDict[key] == 0:
                weight = -maxWeight
            elif posValue == 0 or self.empiricalMeanDict[key] == 1:
                weight = maxWeight
            else:
                ## Need Cases: negValue / posValue 0 / 1
                weight = np.log(
                    (negValue / posValue) * (self.empiricalMeanDict[key] / (1 - self.empiricalMeanDict[key])))
            self.weightsDict[key].append(weight)
            self.headCores.update(encoding.create_head_core(expressionsDict[key], "expFactor", weight))

    def downward_messages(self):
        sentMessages = [self.headCores[key].colors[0] for key in self.headCores]
        weightingMessages = set([self.messageCores[key].colors[0] for key in self.messageCores if
                                 self.messageCores[key].colors[0] not in sentMessages])
        readyMessages = find_ready_messages(weightingMessages, sentMessages, self.directedCores)

        while len(readyMessages) != 0:
            messageKey = readyMessages.pop()
            weightingMessages.remove(messageKey)

            self.messageCores[messageKey + messageCoreSuffix] = engine.contract(
                {**find_parent_cores(messageKey, self.directedCores),
                 **{key+messageCoreSuffix: self.messageCores[key+messageCoreSuffix] for key in find_parent_messages(messageKey, self.directedCores) +
                    find_sibling_messages(messageKey, self.directedCores)}},
                openColors=[messageKey]
            )

            sentMessages.append(messageKey)
            if len(readyMessages) != 0:
                readyMessages = find_ready_messages(weightingMessages, sentMessages, self.directedCores)


def find_parent_cores(message, coresDict):
    return {key: coresDict[key] for key in coresDict if message in coresDict[key].colors[:-1]}


def find_parent_messages(message, coresDict):
    return [coresDict[key].colors[-1] for key in coresDict if message in coresDict[key].colors[:-1]]


def find_sibling_messages(message, coresDict):
    siblings = []
    for key in coresDict:
        if message in coresDict[key].colors[:-1]:
            siblings = siblings + [color for color in coresDict[key].colors[:-1] if color != message]
    return siblings


def find_ready_messages(weightingMessages, sentMessages, coresDict):
    readyMessages = []
    for message in weightingMessages:
        if all([parMes in sentMessages for parMes in find_parent_messages(message, coresDict)]):
            readyMessages.append(message)
    return readyMessages


def find_ready_colors(weightingColors, directedCoresDict, readyCores):
    readyColors = []
    for color in weightingColors:
        if not any(
                [color in directedCoresDict[key].colors[:-1] and key not in readyCores for key in directedCoresDict]):
            readyColors.append(color)
    return readyColors


def find_ready_cores_in_direction(weightingCores, directedCoresDict, readyColors):
    readyCores = []
    for coreKey in weightingCores:
        if len(directedCoresDict[coreKey].colors) == 3:
            if directedCoresDict[coreKey].colors[0] in readyColors and directedCoresDict[coreKey].colors[
                1] in readyColors:
                readyCores.append(coreKey)
        elif len(directedCoresDict[coreKey].colors) == 2:
            if directedCoresDict[coreKey].colors[0] in readyColors:
                readyCores.append(coreKey)
    return readyCores


if __name__ == "__main__":
    expressionsDict = {
        "e1": ["imp", "p", "q"],
        "e2": ["and", "q", "r"],
        "e3": ["q"]
    }

    empiricalMeanDict = {
        "e1": 0.8,
        "e2": 0.6,
        "e3": 0.1,
    }

    matcher = MPMomentMatcher(expressionsDict, empiricalMeanDict)
    matcher.message_passing(circulationNumber=10)

    from matplotlib import pyplot as plt
    for key in matcher.weightsDict:
        plt.scatter(range(len(matcher.weightsDict[key])), matcher.weightsDict[key], label=key, marker="+")
    plt.title("Weight Development during optimization")
    plt.legend()
    plt.show()

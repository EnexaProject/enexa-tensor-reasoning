import numpy as np


class ContractionOptimizer:
    def __init__(self, coreColorDict, colorDimDict, coreList=None):
        self.coreColorDict = coreColorDict  ## Colors of each core
        self.colorDimDict = colorDimDict  ## Dimension of each color

        if coreList is None:
            self.set_random_coreList()
        else:
            self.coreList = coreList

        self.currentSet = []

    def set_coreList(self, coreList):
        self.coreList = coreList

    def set_random_coreList(self):
        self.coreList = np.array(list(self.coreColorDict.keys()))
        np.random.shuffle(self.coreList)

    def fine_openColors_at(self, pos, coreList=None):  # Contraction Size of coreList up to position pos
        if coreList is None:
            coreList = self.coreList
        contractedCores = coreList[:pos]
        uncontractedCores = coreList[pos:]

        contractedColors = compute_colors_in_cores(contractedCores, self.coreColorDict)
        restColors = compute_colors_in_cores(uncontractedCores, self.coreColorDict)

        openColors = []
        for color in contractedColors:
            if color in restColors:
                openColors.append(color)

        openShape = [self.colorDimDict[color] for color in openColors]
        openSize = np.prod(openShape)

        return openColors, openShape, openSize

    def evaluate_coreList(self, coreList=None):
        if coreList is None:
            coreList = self.coreList
        self.openColors = []
        self.openShapes = []
        self.openSizes = []
        for pos in range(len(coreList) + 1):
            colors, shape, size = self.fine_openColors_at(pos, coreList)
            self.openColors.append(colors)
            self.openShapes.append(shape)
            self.openSizes.append(size)
        return self.openShapes

    def compute_heuristic_at(self, pos, newColorPenalty=1, colorCloseScore=1):
        openColors, _, _ = self.fine_openColors_at(pos)
        heuristic = np.ones(len(self.coreList[pos:]))

        ## Compute color close score
        for color in openColors:
            colorCount = len([core for core in self.coreList[pos:] if color in self.coreColorDict[core]])
            for i, core in enumerate(self.coreList[pos:]):
                if color in self.coreColorDict[core]:
                    heuristic[i] = heuristic[i] * ((colorCloseScore * self.colorDimDict[color]) / colorCount)

        ## Compute new color penalty
        for i, core in enumerate(self.coreList[pos:]):
            for color in self.coreColorDict[core]:
                if color not in openColors:
                    heuristic[i] = heuristic[i] / (newColorPenalty * self.colorDimDict[color])
        return heuristic

    def optimize_using_heuristic(self, newColorPenalty=1, colorCloseScore=1, verbose=False):
        if verbose:
            openSizes = self.evaluate_coreList()
            print(
                "Before optimization have a footprint {} using order {}.".format(np.sum(openSizes), self.coreList))
        for pos in range(len(self.coreList)):
            maxPos = np.argmax(
                self.compute_heuristic_at(pos, newColorPenalty=newColorPenalty, colorCloseScore=colorCloseScore))
            self.coreList[pos], self.coreList[pos + maxPos] = self.coreList[pos + maxPos], self.coreList[pos]

        if verbose:
            openSizes = self.evaluate_coreList()
            print(
                "After optimization have a footprint {} using order {}.".format(np.sum(openSizes), self.coreList))

    def random_modification(self, temp=1, criterion="memory"):
        modCoreList = self.coreList.copy()

        previousShapes = self.evaluate_coreList(modCoreList)
        previousScore = np.sum([np.prod(shape) for shape in previousShapes])

        modPos1 = np.random.randint(0, len(self.coreList))
        modPos2 = np.random.randint(0, len(self.coreList))

        modCoreList[modPos1], modCoreList[modPos2] = modCoreList[modPos2], modCoreList[modPos1]

        if criterion == "memory":
            newShapes = self.evaluate_coreList(modCoreList)
            newScore = np.sum([np.prod(shape) for shape in newShapes])

            acceptanceProb = 1 / (1 + np.exp(temp * (newScore - previousScore)))
        else:
            raise ValueError("Criterion {} not understood in randomized coreList optimization.".format(criterion))

        accept = np.random.choice([0, 1], p=[1 - acceptanceProb, acceptanceProb])
        if accept:
            self.coreList = modCoreList
        return accept and modPos1 != modPos2

    def metropolis(self, repetitions=10, temp=1, criterion="memory", verbose=True):
        if verbose:
            firstShapes = self.evaluate_coreList()
            firstScore = np.sum([np.prod(shape) for shape in firstShapes])
            modCounter = 0
        for repetition in range(repetitions):
            modified = self.random_modification(temp=temp, criterion=criterion)
            if modified and verbose:
                modCounter += 1
                print("## Random modification to {} accepted".format(self.coreList))
        if verbose:
            print("Of {} modifications {} have been accepted.".format(repetitions, modCounter))
            newShapes = self.evaluate_coreList()
            newScore = np.sum([np.prod(shape) for shape in newShapes])
            print("The criterion {} changed from {} to {}.".format(criterion, firstScore, newScore))


def compute_colors_in_cores(cores, coreColorDict):
    colors = []
    for core in cores:
        for color in coreColorDict[core]:
            if color not in colors:
                colors.append(color)
    return colors


if __name__ == "__main__":
    cCDict = {
        "C1": ["x", "y", "z", "q", "r"],
        "C2": ["x", "y", "z", "q", "r"],
        "C3": ["x"],
        "C4": ["y"],
        "C5": ["y"]
    }

    cDDict = {
        "x": 2,
        "y": 3,
        "z": 1,
        "q": 20,
        "r": 30
    }

    optim = ContractionOptimizer(cCDict, cDDict)
    print(optim.coreList)
    optim.optimize_using_heuristic(colorCloseScore=10)
    print(optim.coreList)

    optim.metropolis(repetitions=int(1e4))
    optim.random_modification()

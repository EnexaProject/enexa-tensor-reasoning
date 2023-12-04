import numpy as np

class ContractionOptimizer:
    def __init__(self, coreColorDict, colorDimDict, coreList = None):
        self.coreColorDict = coreColorDict ## Colors of each core
        self.colorDimDict = colorDimDict ## Dimension of each color

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

    def fine_openColors_at(self, pos): # Contraction Size of coreList up to position pos
        contractedCores = self.coreList[:pos]
        uncontractedCores = self.coreList[pos:]

        contractedColors = compute_colors_in_cores(contractedCores, self.coreColorDict)
        restColors = compute_colors_in_cores(uncontractedCores, self.coreColorDict)

        openColors = []
        for color in contractedColors:
            if color in restColors:
                openColors.append(color)

        openShape = [self.colorDimDict[color] for color in openColors]
        openSize = np.prod(openShape)

        return openColors, openShape, openSize

    def evaluate_coreList(self):
        self.openColors = []
        self.openShapes = []
        self.openSizes = []
        for pos in range(len(self.coreList)+1):
            colors, shape, size = self.fine_openColors_at(pos)
            self.openColors.append(colors)
            self.openShapes.append(shape)
            self.openSizes.append(size)

    def compute_heuristic_at(self, pos, newColorPenalty = 1, colorCloseScore = 1):
        openColors, _, _ = self.fine_openColors_at(pos)
        heuristic = np.ones(len(self.coreList[pos:]))

        ## Compute color close score
        for color in openColors:
            colorCount = len([core for core in self.coreList[pos:] if color in self.coreColorDict[core]])
            for i, core in enumerate(self.coreList[pos:]):
                if color in self.coreColorDict[core]:
                   heuristic[i] = heuristic[i] * ((colorCloseScore*self.colorDimDict[color])/colorCount)

        ## Compute new color penalty
        for i, core in enumerate(self.coreList[pos:]):
            for color in self.coreColorDict[core]:
                if color not in openColors:
                    heuristic[i] = heuristic[i] / (newColorPenalty*self.colorDimDict[color])
        return heuristic

    def optimize_using_heuristic(self, newColorPenalty = 1, colorCloseScore = 1, verbose = True):
        if verbose:
            self.evaluate_coreList()
            print("Before optimization have a footprint {} using order {}.".format(np.sum(self.openSizes),self.coreList))
        for pos in range(len(self.coreList)):
            maxPos = np.argmax(self.compute_heuristic_at(pos, newColorPenalty= newColorPenalty, colorCloseScore = colorCloseScore))
            self.coreList[pos], self.coreList[pos+maxPos] = self.coreList[pos+maxPos], self.coreList[pos]

        if verbose:
            self.evaluate_coreList()
            print("After optimization have a footprint {} using order {}.".format(np.sum(self.openSizes),self.coreList))


def compute_colors_in_cores(cores, coreColorDict):
    colors = []
    for core in cores:
        for color in coreColorDict[core]:
            if color not in colors:
                colors.append(color)
    return colors

if __name__ == "__main__":
    cCDict = {
        "C1" : ["x","y","z","q","r"],
        "C2" : ["x","y","z","q","r"],
        "C3" : ["x"],
        "C4" : ["y"],
        "C5" : ["y"]
    }

    cDDict = {
        "x" : 2,
        "y" : 3,
        "z" : 1,
        "q" : 20,
        "r" : 30
    }

    optim = ContractionOptimizer(cCDict,cDDict)
    print(optim.coreList)
    optim.optimize_using_heuristic(colorCloseScore=10)
    print(optim.coreList)
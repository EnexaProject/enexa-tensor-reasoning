from matplotlib import pyplot as plt
import numpy as np

from tnreason.optimization import contraction_optimization as co


class CoreContractor:
    """
    coreDict: list of CoordinateCores
    contractionList: list of colors
    instructionList: list of contraction instructions: either and with additional core or reduce a color. First entry must be add to start with
    """

    def __init__(self, coreDict={}, coreList=None, instructionList=None, openColors=[]):
        self.coreDict = coreDict
        self.coreList = coreList
        self.instructionList = instructionList
        self.openColors = openColors

    def exponentiate_with_weight(self, weightDict, exeptionKeys=[]):
        for coreKey in self.coreDict:
            self.coreDict[coreKey] = self.coreDict[coreKey].weighted_exponentiation(weightDict[coreKey])

    def optimize_coreList(self):
        # Generate the coreColorDict and colorDimDict for ContractionOptimizer
        coreColorDict = {}
        colorDimDict = {}
        for coreKey in self.coreDict:
            coreColorDict[coreKey] = self.coreDict[coreKey].colors.copy()
            for i, color in enumerate(self.coreDict[coreKey].colors):
                if color not in colorDimDict:
                    colorDimDict[color] = self.coreDict[coreKey].values.shape[i]
        optimizer = co.GreedyHeuristicOptimizer(coreColorDict, colorDimDict)
        # Optimize coreList i.e. order of contraction
        optimizer.optimize()
        self.coreList = optimizer.coreList

    def create_instructionList_from_coreList(self, verbose=False):
        if self.coreList is None:
            self.coreList = list(self.coreDict.keys())
        # Find all colors
        colorList = []
        for key in self.coreDict:
            for color in self.coreDict[key].colors:
                if color not in colorList:
                    colorList.append(color)
        # Find cores after which color can be reduced
        self.coreList = list(self.coreList)
        self.coreList.reverse()
        reduceDict = {key: [] for key in self.coreDict}
        for color in colorList:
            if color not in self.openColors:
                for key in self.coreList:
                    if color in self.coreDict[key].colors:
                        reduceDict[key].append(color)
                        break
        self.coreList.reverse()
        # Create the instructionList
        self.instructionList = []
        for key in self.coreList:
            self.instructionList.append(["add", key])
            for color in reduceDict[key]:
                self.instructionList.append(["reduce", color])
        if verbose:
            print("The instructionList is {} and colorList {}.".format(self.instructionList,
                                                                       {key: self.coreDict[key].colors for key in
                                                                        self.coreDict}))

    def evaluate_sizes_instructionList(self, show=True):
        shapeList = [[]]
        colorList = [[]]
        for instruction in self.instructionList:
            shapes = shapeList[-1].copy()
            colors = colorList[-1].copy()
            if instruction[0] == "add":
                for i, color in enumerate(self.coreDict[instruction[1]].colors):
                    if color not in colors:
                        colors.append(color)
                        shapes.append(self.coreDict[instruction[1]].values.shape[i])
            elif instruction[0] == "reduce":
                popindex = colors.index(instruction[1])
                shapes.pop(popindex)
                colors.pop(popindex)
            shapeList.append(shapes)
            colorList.append(colors)
        sizeList = []
        for shapes in shapeList:
            sizeList.append(np.prod(shapes))

        if show:
            plt.title("Tensor Sizes During the Contraction", fontsize=15)
            plt.scatter(range(1, len(sizeList)), sizeList[1:], marker="+")
            plt.xticks(range(1, len(sizeList)), [str(ins) for ins in self.instructionList])
            plt.xlabel("Instructions")
            plt.ylabel("Number of Coordinates of the contracted")
            plt.ylim([0, 1.1 * max(sizeList)])
            plt.show()

        return sizeList, shapeList, colorList

    def contract(self, verbose=False):
        if self.instructionList is None:
            self.create_instructionList_from_coreList()
        contracted = self.coreDict[self.instructionList[0][1]]
        for instruction in self.instructionList[1:]:
            if verbose:
                print("## Doing {} ##".format(instruction))
            if instruction[0] == "add":
                contracted = contracted.compute_and(self.coreDict[instruction[1]])
            elif instruction[0] == "reduce":
                contracted = contracted.reduce_color(instruction[1])
            else:
                raise ValueError("Instruction {} not understood.".format(instruction))
        if verbose and len(contracted.values.shape) > 0:
            print("Missing contraction colors are {}.".format(contracted.colors))
        return contracted

    def contract_color(self, color):
        affectedCores, affectedKeys = find_affected(self.coreDict, color)

        contracted = self.coreDict[affectedKeys[0]]
        for i, key in enumerate(affectedKeys[1:]):
            contracted = contracted.compute_and(self.coreDict[key])

        contracted.count_on_color(color)

        self.coreDict = {key: self.coreDict[key] for key in self.coreDict if key not in affectedKeys}
        self.coreDict["contracted_" + color] = contracted

        return contracted


def find_affected(coreDict, color):
    affectedCores = []
    affectedKeys = []
    for key in coreDict:
        if color in coreDict[key].colors:
            affectedCores.append(coreDict[key])
            affectedKeys.append(key)
    return affectedCores, affectedKeys


if __name__ == "__main__":
    import tnreason.logic.coordinate_calculus as cc
    import numpy as np


    def random_basis():
        vector = np.zeros((2))
        if np.random.binomial(n=1, p=0.5) == 1:
            vector[0] = 1
        else:
            vector[1] = 1
        return vector


    def calculate_random_basis_core(shape):
        shapeProduct = np.prod(shape)
        core = np.zeros([2, shapeProduct])
        for i in range(shapeProduct):
            core[:, i] = random_basis()
        return core.reshape([2] + shape)


    coordinateDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "q", "z"], name="b"),
    }

    weightDict = {
        "a": 2.134,
        "b": 1.72345
    }
    contractor = CoreContractor(coordinateDict,
                                None,
                                [["add", "a"], ["add", "b"], ["reduce", "x"], ["reduce", "y"], ["reduce", "q"]])

    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()
    print(contractor.instructionList)  ## Hast to reduce colors only after last usage.

    contractorWithOpen = CoreContractor(coordinateDict, instructionList=None, coreList=None, openColors=["x"])
    contractorWithOpen.create_instructionList_from_coreList()
    print(contractorWithOpen.instructionList)  ## Has to be without openColor Reduction, i.e. "x".
    contractorWithOpen.evaluate_sizes_instructionList(show=True)

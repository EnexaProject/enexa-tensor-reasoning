from matplotlib import pyplot as plt

from tnreason.contraction import contraction_optimization as co
from tnreason.contraction import bc_contraction_generation as cg
from tnreason.contraction import contraction_visualization as cv

from tnreason.logic import coordinate_calculus as cc

class ChainContractorBase:
    """
    coreDict: Dictionary of CoordinateCores
    coreList: Order of coreDict keys for contraction
    instructionList: list of contraction instructions: either "and" with additional core or "reduce" with a color. First entry must be add to start with.
    """

    def __init__(self, coreDict={}, coreList=None, instructionList=None, openColors=[]):
        self.coreDict = coreDict
        self.coreList = coreList
        self.instructionList = instructionList
        self.openColors = openColors

    def optimize_coreList(self, method="GreedyHeuristic"):
        # Generate the coreColorDict and colorDimDict for ContractionOptimizer
        coreColorDict = {}
        colorDimDict = {}
        for coreKey in self.coreDict:
            coreColorDict[coreKey] = self.coreDict[coreKey].colors.copy()
            for i, color in enumerate(self.coreDict[coreKey].colors):
                if color not in colorDimDict:
                    colorDimDict[color] = self.coreDict[coreKey].values.shape[i]
        if method == "GreedyHeuristic":
            optimizer = co.GreedyHeuristicOptimizer(coreColorDict, colorDimDict, globallyOpenColors=self.openColors)
            # Optimize coreList i.e. order of contraction
            optimizer.optimize()
        elif method == "Dijkstra":
            optimizer = co.DijkstraOptimizer(coreColorDict, colorDimDict, globallyOpenColors=self.openColors)
            optimizer.optimize()
        else:
            raise ValueError("Optimization Method {} not understood!".format(method))
        self.coreList = optimizer.coreList

    def get_reduceDict_from_coreList(self):
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
        return reduceDict


class ChainTensorContractor(ChainContractorBase):
    ## When NumpyTensorCores in coreDict
    def contract(self):
        reduceDict = self.get_reduceDict_from_coreList()
        contracted = self.coreDict[self.coreList[0]].reduce_colors(reduceDict[self.coreList[0]])
        for coreKey in self.coreList[1:]:
            contracted = contracted.reduced_contraction(self.coreDict[coreKey], reduceDict[coreKey])
        return contracted


class CoreContractor(ChainContractorBase):
    def create_instructionList_from_coreList(self, verbose=False):
        reduceDict = self.get_reduceDict_from_coreList()
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

    def visualize(self, title="Contraction Diagram", useInstructionList=True):
        pos = None
        if useInstructionList and self.instructionList is not None:
            pos = cv.get_positions_from_instructions(self.instructionList, self.openColors)
        elif useInstructionList and self.instructionList is None:
            print("Warning: Cannot use InstructionList for visualization, since not initialized!")
        cv.draw_contractionDiagram(self.coreDict, title=title, pos=pos)

    def contract(self, optimizationMethod=None, verbose=False):
        if len(self.coreDict)==0:
            return "EmptyCore"
        if optimizationMethod is None or optimizationMethod == "GreedyHeuristic":
            self.optimize_coreList()
            self.create_instructionList_from_coreList()
        else:
            raise ValueError("Optimization Method {} not supported!".format(optimizationMethod))
        contracted = self.coreDict[self.instructionList[0][1]]
        for instruction in self.instructionList[1:]:
            if verbose:
                print("## Doing {} ##".format(instruction))
            if instruction[0] == "add":
                contracted = contracted.compute_and(self.coreDict[instruction[1]])
            elif instruction[0] == "reduce":
                contracted = contracted.reduce_color(instruction[1])
            elif instruction[0] == "exp":
                contracted = contracted.exponenentiate()
            else:
                raise ValueError("Instruction {} not understood.".format(instruction))
        if verbose and len(contracted.values.shape) > 0:
            print("Missing contraction colors are {}.".format(contracted.colors))
        return contracted

    ## Unused!

    def generate_coreDict_from_formulaList(self, formulaList):
        ## This should not be the job ob coreContractor, but of a representator!
        for formula in formulaList:
            self.coreDict = {**self.coreDict, **cg.generate_factor_dict(formula)}

    def contract_color(self, color):
        affectedCores, affectedKeys = find_affected(self.coreDict, color)

        contracted = self.coreDict[affectedKeys[0]]
        for i, key in enumerate(affectedKeys[1:]):
            contracted = contracted.compute_and(self.coreDict[key])

        contracted.count_on_color(color)

        self.coreDict = {key: self.coreDict[key] for key in self.coreDict if key not in affectedKeys}
        self.coreDict["contracted_" + color] = contracted

        return contracted

    def numpy_einsum_contract(self):
        subscripts = self.create_contraction_subscripts()

        contractedValues = np.einsum(subscripts, *[self.coreDict[coreKey].values for coreKey in self.coreList])
        return cc.CoordinateCore(contractedValues, self.openColors)

    def create_contraction_subscripts(self):
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']

        colorDict = {}
        i = 0
        for coreKey in self.coreDict:
            for color in self.coreDict[coreKey].colors:
                if color not in colorDict:
                    colorDict[color] = alphabet[i]
                    i += 1

        colorList = []
        for coreKey in self.coreList:
            colorList.append("".join([colorDict[color] for color in self.coreDict[coreKey].colors]))

        lhs = ",".join(colorList)
        rhs = "".join([colorDict[color] for color in self.openColors])

        return lhs + "->" + rhs


def find_affected(coreDict, color):
    affectedCores = []
    affectedKeys = []
    for key in coreDict:
        if color in coreDict[key].colors:
            affectedCores.append(coreDict[key])
            affectedKeys.append(key)
    return affectedCores, affectedKeys


## To replace Optimization Calculus
class NegationTolerantCoreContractor:
    ## In CoreDict
    def __init__(self, coreDict={}, coreList=None, instructionList=None, openColors=[]):
        self.coreDict = coreDict  ## Now each key gives a list of list [core, ignorecolors]
        self.coreList = coreList
        self.instructionList = instructionList
        self.openColors = openColors

    def contract(self):

        contractedList = self.coreDict[self.instructionList[0][1]]
        print(len(contractedList))
        for instruction in self.instructionList[1:]:
            if instruction[0] == "add":
                contractedList = compute_list_and(contractedList, self.coreDict[instruction[1]])
                #            elif instruction[0] == "reduce":
                #                contracted = contracted.reduce_color(instruction[1])
                print(instruction, len(contractedList))
        return contractedList


def compute_list_and(leftList, rightList):
    preResultList = []

    ## Add the non Constants
    for leftCore, leftIgnoreColors in leftList:
        for rightCore, rightIgnoreColors in rightList:
            if (not is_constant(leftIgnoreColors, rightCore.colors)) and (
                    not is_constant(leftIgnoreColors, rightCore.colors)):
                preResultList.append([leftCore.compute_and(rightCore), leftIgnoreColors + rightIgnoreColors])

    ## Add the constants from both sides
    for leftCore, leftIgnoreColors in leftList:
        firstRightCore, firstRightIgnoreColors = rightList[0]
        if is_constant(leftIgnoreColors, firstRightCore.colors):
            preResultList.append([leftCore, leftIgnoreColors])
    for rightCore, rightIgnoreColors in rightList:
        firstLeftCore, firstLeftIgnoreColors = leftList[0]
        if is_constant(rightIgnoreColors, firstLeftCore.colors):
            preResultList.append([rightCore, rightIgnoreColors])

    ## Sum the cores with same colors
    resultList = []
    while len(preResultList) > 0:
        core1, ignoreColors1 = preResultList.pop()
        for core2, ignoreColors2 in preResultList.copy():
            if color_equivalence(core1, core2):
                core1 = core1.sum_with(core2)
                preResultList.pop(preResultList.index([core2, ignoreColors2]))
        resultList.append([core1, ignoreColors1])
    return resultList


def is_constant(ignoreColors, testColors):
    for color in ignoreColors:
        if color in testColors:
            return True
    return False


def color_equivalence(core1, core2):
    for color in core1.colors:
        if color not in core2.colors:
            return False
    for color in core2.colors:
        if color not in core1.colors:
            return False
    return True


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
                                [["add", "a"], ["add", "b"], ["reduce", "x"], ["reduce", "y"]],
                                ["q"]
                                )

    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()
    contractor.visualize()
    exit()

    contractor.create_instructionList_from_coreList()
    print(contractor.instructionList)  ## Hast to reduce colors only after last usage.

    contractorWithOpen = CoreContractor(coordinateDict, instructionList=None, coreList=None, openColors=["x"])
    contractorWithOpen.create_instructionList_from_coreList()
    print(contractorWithOpen.instructionList)  ## Has to be without openColor Reduction, i.e. "x".
    # contractorWithOpen.evaluate_sizes_instructionList(show=True)

    print(contractor.create_contraction_subscripts())
    print(contractor.numpy_einsum_contract().values)

    contractor = NegationTolerantCoreContractor(
        coreDict={
            "a": [
                [cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y", "z"], name="a"), ["y"]],
                [cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "r", "z"], name="a"), ["y2"]]
            ],
            "b": [[cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y2", "z"], name="a2"),
                   ["y2"]],
                  [cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "r", "z"], name="a"),
                   ["y2"]]
                  ],
        },
        instructionList=[["add", "a"], ["add", "b"], ["reduce", "x"], ["reduce", "y"]],
    )
    result = contractor.contract()
    print(result)

    exit()

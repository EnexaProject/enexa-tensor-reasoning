from tnreason.contraction import contraction_visualization as cv


class TreeContractorBase:
    def __init__(self, coreDict={}, contractionTree=[], openColors=[]):
        self.coreDict = coreDict
        self.contractionTree = contractionTree
        self.openColors = openColors

        self.reductionDict = self.get_reductionDict()

    def optimize_contractionTree(self):
        ## To Do: Implement Algorithms
        pass

    def get_reductionDict(self):
        self.reductionDict = initialize_reductionDict(self.contractionTree)
        for color in get_colors(self.contractionTree, self.coreDict):
            if color not in self.openColors:
                reducePos = find_last_usage(color, self.contractionTree, self.coreDict)
                self.reductionDict[str(reducePos)].append(color)

    def visualize(self):
        cv.draw_contractionDiagram(self.coreDict)


class TreeTensorContractor(TreeContractorBase):
    def contract(self):
        return self.contraction_step(self.contractionTree)

    def contraction_step(self, contractionTree):
        reductionColors = self.reductionDict[str(contractionTree)]
        if isinstance(contractionTree, str):
            return self.coreDict[contractionTree].reduce_colors(reductionColors)
        else:
            contracted = self.contraction_step(contractionTree[0])
            for subTree in contractionTree[1:-1]:
                contracted = contracted.reduced_contraction(self.contraction_step(subTree), [])
            return contracted.reduced_contraction(self.contraction_step(contractionTree[-1]), reductionColors)


def initialize_reductionDict(contractionTree):
    if isinstance(contractionTree, str):
        return {contractionTree: []}
    reductionDict = {str(contractionTree): []}
    for subTree in contractionTree:
        reductionDict = {**reductionDict, **initialize_reductionDict(subTree)}
    return reductionDict


def get_coreKeys(contractionTree):
    if isinstance(contractionTree, str):
        return [contractionTree]
    coreList = []
    for subTree in contractionTree:
        for core in get_coreKeys(subTree):
            if core not in coreList:
                coreList.append(core)
    return coreList


def get_colors(contractionTree, coreDict):
    colorList = []
    for coreKey in get_coreKeys(contractionTree):
        for color in coreDict[coreKey].colors:
            if color not in colorList:
                colorList.append(color)
    return colorList


def find_last_usage(color, contractionTree, coreDict):
    if isinstance(contractionTree, str):
        return contractionTree
    elif sum([color in get_colors(subTree, coreDict) for subTree in contractionTree]) > 1:
        return contractionTree
    else:
        for subTree in contractionTree:
            if color in get_colors(subTree, coreDict):
                return find_last_usage(subTree)


if __name__ == "__main__":
    contree = ["c1", ["c2", "c3"], "c4"]
    from tnreason.contraction import generic_cores as gc

    from tnreason.logic import coordinate_calculus as cc

    import numpy as np

    coreDict = {
        "c1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(3, 2)), ["a", "b"]),
        "c2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"]),
        "c3": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"]),
        "c4": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"])
    }

    tensorCoreDict = {
        key: gc.change_type(coreDict[key]) for key in coreDict
    }

    contractor = TreeTensorContractor(tensorCoreDict, contree, ["c", "a"])
    contractor.get_reductionDict()
    print(contractor.contract().values)
    contractor.visualize()

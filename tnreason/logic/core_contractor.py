
class CoreContractor:
    """
    coreDict: list of CoordinateCores
    contractionList: list of colors
    instructionList: list of contraction instructions: either and with additional core or reduce a color. First entry must be add to start with
    """
    def __init__(self, coreDict={}, instructionList=[]):
        self.coreDict = coreDict
        self.instructionList = instructionList

    def exponentiate_with_weight(self, weightDict, exeptionKeys=[]):
        for coreKey in self.coreDict:
            self.coreDict[coreKey] = self.coreDict[coreKey].weighted_exponentiation(weightDict[coreKey])

    def create_instructionList_from_coreList(self, coreList=None):
        if coreList is None:
            coreList = list(self.coreDict.keys())
        # Find all colors
        colorList = []
        for key in self.coreDict:
            for color in self.coreDict[key].colors:
                if color not in colorList:
                    colorList.append(color)
        # Find core after which color can be reduced
        coreList.reverse()
        reduceDict = {key: [] for key in self.coreDict}
        for color in colorList:
            for key in coreList:
                if color in self.coreDict[key].colors:
                    reduceDict[key].append(color)
                    break
        # Create the instructionList
        self.instructionList = []
        for key in reduceDict:
            self.instructionList.append(["add", key])
            for color in reduceDict[key]:
                self.instructionList.append(["reduce", color])

    def contract(self,verbose=False):
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
        if verbose and len(contracted.values.shape)>0:
            print("Missing contraction colors are {}.".format(contracted.colors))
        return contracted


    def contract_color(self, color):
        affectedCores, affectedKeys = find_affected(self.coreDict, color)

        contracted = self.coreDict[affectedKeys[0]]
        for i, key in enumerate(affectedKeys[1:]):
            contracted = contracted.compute_and(self.coreDict[key])

        contracted.count_on_color(color)

        self.coreDict = {key: self.coreDict[key] for key in self.coreDict if key not in affectedKeys}
        self.coreDict["contracted_"+color] = contracted

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
    contractor = CoreContractor(coordinateDict, [["add","a"],["add","b"],["reduce","x"],["reduce","y"],["reduce","q"]])
    contractor.create_instructionList_from_coreList()
    contractor.exponentiate_with_weight(weightDict)
    print(contractor.contract().values)
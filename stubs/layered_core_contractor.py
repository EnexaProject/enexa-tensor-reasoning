from tnreason.contraction import layers as lay
from tnreason.contraction import layered_contraction_generation as lcg

class LayeredCoreContractor:

    def __init__(self, layerDict={}, layerList=None, instructionList=None, openColors=[]):
        self.layerDict = layerDict
        self.layerList = layerList

        self.instructionList = instructionList
        self.openColors = openColors

    def contract(self):
        contracted = self.layerDict[self.instructionList[0][1]]
        for instruction in self.instructionList[1:]:
            if instruction[0] == "add":
                contracted = lay.contract(contracted, self.layerDict[instruction[1]])
            elif instruction[0] == "reduce":
                contracted = contracted.reduce_color(instruction[1])
        return contracted

    ## From analogous method in standard CoreContractor
    def create_instructionList_from_layerList(self):

        # Find all colors
        colorList = []
        for key in self.layerDict:
            for color in self.layerDict[key].colors:
                if color not in colorList and color not in self.openColors:
                    colorList.append(color)

        # Find cores after which color can be reduced
        self.layerList = list(self.layerList)
        self.layerList.reverse()
        reduceDict = {key: [] for key in self.layerDict}
        for color in colorList:
            if color not in self.openColors:
                for key in self.layerList:
                    if color in self.layerDict[key].colors:
                        reduceDict[key].append(color)
                        break
        self.layerList.reverse()

        # Create the instructionList
        self.instructionList = []
        for key in reduceDict:
            self.instructionList.append(["add", key])
            for color in reduceDict[key]:
                self.instructionList.append(["reduce", color])

if __name__ == "__main__":
    layDict = lcg.generate_skeletonLayerDict(["a","and",["not","b"]], {"a": 2, "b": 10}, 4)
    contractor = LayeredCoreContractor(layDict, list(layDict.keys()))
    contractor.create_instructionList_from_layerList()
    print(contractor.contract().coresDict)
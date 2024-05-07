from tnreason import engine
import numpy as np

categoricalCoreSuffix = "_catCore"

def create_categorical_cores(categoricalsDict):
    """
    Creates a tensor network representing the constraints of
        * categoricalsDict: Dictionary of atom lists to each categorical variable
    """
    catCores = {}
    for catName in categoricalsDict.keys():
        catCores = {**catCores, **create_constraintCoresDict(categoricalsDict[catName], catName)}
    return catCores


def create_constraintCoresDict(atoms, catName):
    constraintCoresDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.zeros(shape=(len(atoms), 2))
        coreValues[:, 0] = np.ones(shape=(len(atoms)))
        coreValues[i, 0] = 0
        coreValues[i, 1] = 1
        constraintCoresDict[catName + "_" + atomKey + categoricalCoreSuffix] = engine.get_core()(
            coreValues, [catName, atomKey], name=catName + "_" + atomKey + categoricalCoreSuffix)
    return constraintCoresDict

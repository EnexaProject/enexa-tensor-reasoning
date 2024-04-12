from tnreason import engine
import numpy as np


def create_constraints(specDict):
    catCores = {}
    for catName in specDict.keys():
        catCores = {**catCores, **create_constraintCoresDict(specDict[catName], catName)}
    return catCores

def create_constraintCoresDict(atoms, name, coreType="NumpyTensorCore"):
    constraintCoresDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.zeros(shape=(len(atoms), 2))
        coreValues[:, 0] = np.ones(shape=(len(atoms)))
        coreValues[i, 0] = 0
        coreValues[i, 1] = 1
        constraintCoresDict[name + "_" + atomKey + "_catCore"] = engine.get_core(coreType=coreType)(
            coreValues,
            [
                name,
                atomKey],
            name=name + "_" + atomKey + "_catCore")
    return constraintCoresDict
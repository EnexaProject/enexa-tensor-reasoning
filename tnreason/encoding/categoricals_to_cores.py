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
    return {catName + "_" + atomName + categoricalCoreSuffix: create_single_atomization(catName, len(atoms), i, atomName)[
        catName + "_" + atomName + categoricalCoreSuffix] for i, atomName in enumerate(atoms)}


def create_single_atomization(catName, catDim, position, atomName=None):
    """
    Creates the relation encoding of the categorical X with its atomization to the position (int).
    If the resulting atom is not named otherwise, we call it X=position.
    """
    if atomName is None:
        atomName = catName + "=" + str(position)
    values = np.zeros(shape=(catDim, 2))
    values[:, 0] = np.ones(shape=(catDim))
    values[position, 0] = 0
    values[position, 1] = 1
    return {catName + "_" + atomName + categoricalCoreSuffix: engine.get_core()(
        values, [catName, atomName], name=catName + "_" + atomName + categoricalCoreSuffix
    )}


def create_atomization_cores(atomizationSpecs, catDimDict):
    atomizationCores = {}
    for atomizationSpec in atomizationSpecs:
        catName, position = atomizationSpec.split("=")
        atomizationCores.update(create_single_atomization(catName, catDimDict[catName], int(position)))
    return atomizationCores

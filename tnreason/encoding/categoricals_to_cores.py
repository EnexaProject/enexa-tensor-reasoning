from tnreason import engine

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
    return {
        catName + "_" + atomName + categoricalCoreSuffix: create_single_atomization(catName, len(atoms), i, atomName)[
            catName + "_" + atomName + categoricalCoreSuffix] for i, atomName in enumerate(atoms)}


def create_single_atomization(catName, catDim, position, atomName=None):
    """
    Creates the relation encoding of the categorical X with its atomization to the position (int).
    If the resulting atom is not named otherwise, we call it X=position.
    """
    if atomName is None:
        atomName = catName + "=" + str(position)
    atomizer = lambda catPos: [catPos == position]
    return {catName + "_" + atomName + categoricalCoreSuffix:
                engine.create_relational_encoding(inshape=[catDim], outshape=[2], incolors=[catName],
                                                  outcolors=[atomName],
                                                  function=atomizer, coreType=engine.defaultCoreType,
                                                  name=catName + "_" + atomName + categoricalCoreSuffix
                                                  )}


def create_atomization_cores(atomizationSpecs, catDimDict):
    atomizationCores = {}
    for atomizationSpec in atomizationSpecs:
        catName, position = atomizationSpec.split("=")
        atomizationCores.update(create_single_atomization(catName, catDimDict[catName], int(position)))
    return atomizationCores

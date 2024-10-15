from tnreason import engine

momentCoreSuffix = "_momentCore"
targetCoreSuffix = "_targetCore"


class MomentMatcher:
    """
    Fits alternatingly local cores to reproduce local expected Statistics.
    Equals to coordinate descent with optimal steplength of the likelihood.
        * targetCores: Vector Cores storing the local expected statistics to be matched
        * networkCores: Static Cores shaping the basis
    Generalizes the weight estimation (which is the special case of leg dimension 2 and fitting of exponentiated first coordinate.
    """

    def __init__(self, networkCores, targetCores, coreType=None, contractionMethod=None):
        self.networkCores = networkCores

        self.targetCores = {targetCores[key].colors[0] + targetCoreSuffix: targetCores[key] for key in targetCores}

        self.updateDimDict = {self.targetCores[key].colors[0]: self.targetCores[key].values.shape[0] for key in
                              self.targetCores}
        self.dimDict = engine.get_dimDict(self.networkCores)

        self.coreType = coreType
        self.contractionMethod = contractionMethod

    def ones_initialization(self):
        """
        varDimDict: Dictionary with keys the colors of the moments and the shape the dimension of the axis
        """
        self.networkCores.update(
            engine.create_trivial_cores(list(self.updateDimDict.keys()),
                                        shapeDict={key: [self.updateDimDict[key]] for key in self.updateDimDict},
                                        suffix=momentCoreSuffix,
                                        coreType=self.coreType
                                        )
        )

    def matching_step(self, updateColor):
        self.networkCores.pop(updateColor + momentCoreSuffix)

        self.networkCores[updateColor + momentCoreSuffix] = engine.create_tensor_encoding(
            inshape=[self.dimDict[updateColor]], incolors=[updateColor], function=solve_moment_equation(
                satVect=engine.contract(coreDict=self.networkCores, openColors=[updateColor],
                                        method=self.contractionMethod).values,
                empVect=self.targetCores[updateColor + targetCoreSuffix].values
            ), name=updateColor + momentCoreSuffix, coreType=self.coreType
        )
        # self.networkCores[updateColor + momentCoreSuffix] = engine.get_core()(
        #     values=solve_moment_equation(
        #         satVect=engine.contract(coreDict=self.networkCores, openColors=[updateColor]).values,
        #         empVect=self.targetCores[updateColor + targetCoreSuffix].values
        #     ), colors=[updateColor], name=updateColor + momentCoreSuffix)

        print(self.networkCores[updateColor + momentCoreSuffix].values)

    def alternating_matching(self, sweepNum=10, updateColors=None):
        if updateColors is None:
            updateColors = list(self.updateDimDict.keys())
        for sweep in range(sweepNum):
            for updateColor in updateColors:
                self.matching_step(updateColor)


def find_common_nonzero(vect1, vec2):
    """
    Searching for a reference position for quotients in moment match.
    """
    for i in range(vect1.shape[0]):
        if vect1[i] > 0 and vec2[i] > 0:
            return i
    else:
        return -1


def solve_moment_equation(satVect, empVect):
    """
    Solves the local expected statistics matching of
        * satVect: Marginal probability of color wrt alien cores
        * empVect: Desired marginal probability (expected statistics)
    To Do: Update, such that partition function constant and not one reference coordinate!
    """
    refPos = find_common_nonzero(satVect, empVect)
    if refPos == -1:
        print("Warning: Moments cannot be matched!")
        return lambda i: 1

    # return solVect
    return lambda i: (satVect[refPos] / empVect[refPos]) * (empVect[int(i)] / satVect[int(i)])

class SuperposedFormulaTensor:
    ## Shall be the central object to be optimized during MLE

    def __init__(self):
        self.parameterCoresDict = {} # former variableCoresDict
        self.worldCoresDict = {} # former fixedCoresDict / atomSelectorDict
        self.skeletonCoresDict = {} # new from lcg



    def create_skeletonCoreDict(self, skeletonExpression, placeHolderShapesDict, placeHolderColorsDict):
        # placeHolderDicts -> shapes and colors specify how skeletonTN looks like at each placeholder
        pass

    ## WorldCoresDict Generation: CandidatesDict required for interpretation of the 
    def create_worldCoresDict_from_sampleDf(self, candidatesDict, sampleDf):
        # candidatesDict gives interpretation of placeholder axes
        pass

    def create_worldCoresDict_from_enumeration(self, candidatesDict):
        pass
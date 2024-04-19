from tnreason import algorithms
from tnreason import encoding


class FormulaBooster:
    def __init__(self, knowledgeBase):
        self.knowledgeBase = knowledgeBase

    def find_candidate(self, architectureDict, specDict):
        networkCores = {**encoding.create_architecture(architectureDict)}
        importanceColors = encoding.find_atoms(architectureDict)
        importanceList = [({}, 1),
                          ({**self.knowledgeBase.create_cores()}, -1 / self.knowledgeBase.partitionFunction())]

        colorDims = encoding.find_selection_dimDict(architectureDict)
        updateShapes = {key + "_parCore": colorDims[key] for key in colorDims}
        updateColors = {key + "_parCore": [key] for key in colorDims}
        updateCoreKeys = list(updateShapes.keys())
        print(colorDims)
        if specDict["method"] == "als":
            sampler = algorithms.ALS(networkCores=networkCores, importanceColors=importanceColors,
                                     importanceList=importanceList, targetCores={})
            sampler.random_initialize(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            sampler.alternating_optimization(updateKeys=updateCoreKeys, sweepNum=specDict["sweeps"])
            solutionDict = sampler.get_color_argmax(updateKeys=updateCoreKeys)

        elif specDict["method"] == "gibbs":
            sampler = algorithms.Gibbs(networkCores=networkCores, importanceColors=importanceColors,
                                       importanceList=importanceList)
            sampler.ones_initialization(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            solutionDict = sampler.gibbs_sample(updateKeys=updateCoreKeys, sweepNum=specDict["sweeps"])

        else:
            raise ValueError("Sampling Method {} not known!".format(specDict["method"]))

        return encoding.create_solution_expression(architectureDict, solutionDict)

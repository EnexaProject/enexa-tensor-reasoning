from tnreason import algorithms
from tnreason import encoding

parameterCoreSuffix = "_parCore"


class FormulaBooster:
    def __init__(self, knowledgeBase):
        self.knowledgeBase = knowledgeBase

    def find_candidate(self, architectureDict, specDict, sampleDf):
        networkCores = {**encoding.create_architecture(architectureDict)}
        importanceColors = encoding.find_atoms(architectureDict)
        importanceList = [({**encoding.create_data_cores(sampleDf, importanceColors)}, 1 / sampleDf.values.shape[0]),
                          ({**self.knowledgeBase.create_cores()}, -1 / self.knowledgeBase.partitionFunction())]

        colorDims = encoding.find_selection_dimDict(architectureDict)
        updateShapes = {key + parameterCoreSuffix: colorDims[key] for key in colorDims}
        updateColors = {key + parameterCoreSuffix: [key] for key in colorDims}
        updateCoreKeys = list(updateShapes.keys())
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
            if "annealingPattern" in specDict:
                sampleDict = sampler.annealed_sample(updateKeys=updateCoreKeys,
                                                     annealingPattern=specDict["annealingPattern"])
            elif "sweeps" in specDict:
                sampleDict = sampler.gibbs_sample(updateKeys=updateCoreKeys, sweepNum=specDict["sweeps"])
            else:
                raise ValueError("Bad parameter specification for Gibbs: {}".format(specDict))
            solutionDict = {key[:-8]: int(sampleDict[key]) for key in
                            sampleDict}  # Drop parameterCoreSuffix and ensure int output
        else:
            raise ValueError("Sampling Method {} not known!".format(specDict["method"]))
        return encoding.create_solution_expression(architectureDict, solutionDict)

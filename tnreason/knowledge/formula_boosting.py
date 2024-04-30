from tnreason import algorithms
from tnreason import encoding
from tnreason import engine
parameterCoreSuffix = "_parCore"


class FormulaBooster:
    def __init__(self, knowledgeBase, specDict):
        self.knowledgeBase = knowledgeBase
        self.specDict = specDict

    def find_candidate(self, architectureDict, sampleDf):
        networkCores = {**encoding.create_architecture(architectureDict)}
        importanceColors = encoding.find_atoms(architectureDict)
        importanceList = [({**encoding.create_data_cores(sampleDf, importanceColors)}, 1 / sampleDf.values.shape[0]),
                          ({**self.knowledgeBase.create_cores()}, -1 /
                           engine.contract(coreDict=self.knowledgeBase.create_cores(), openColors=[]).values)]

        colorDims = encoding.find_selection_dimDict(architectureDict)
        updateShapes = {key + parameterCoreSuffix: colorDims[key] for key in colorDims}
        updateColors = {key + parameterCoreSuffix: [key] for key in colorDims}
        updateCoreKeys = list(updateShapes.keys())
        if self.specDict["method"] == "als":
            sampler = algorithms.ALS(networkCores=networkCores, importanceColors=importanceColors,
                                     importanceList=importanceList, targetCores={})
            sampler.random_initialize(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            sampler.alternating_optimization(updateKeys=updateCoreKeys, sweepNum=self.specDict["sweeps"])
            solutionDict = sampler.get_color_argmax(updateKeys=updateCoreKeys)

        elif self.specDict["method"] == "gibbs":
            sampler = algorithms.Gibbs(networkCores=networkCores, importanceColors=importanceColors,
                                       importanceList=importanceList)
            sampler.ones_initialization(updateKeys=updateCoreKeys, shapesDict=updateShapes, colorsDict=updateColors)
            if "annealingPattern" in self.specDict:
                sampleDict = sampler.annealed_sample(updateKeys=updateCoreKeys,
                                                     annealingPattern=self.specDict["annealingPattern"])
            elif "sweeps" in self.specDict:
                sampleDict = sampler.gibbs_sample(updateKeys=updateCoreKeys, sweepNum=self.specDict["sweeps"])
            else:
                raise ValueError("Bad parameter specification for Gibbs: {}".format(self.specDict))
            solutionDict = {key[:-8]: int(sampleDict[key]) for key in
                            sampleDict}  # Drop parameterCoreSuffix and ensure int output
        else:
            raise ValueError("Sampling Method {} not known!".format(self.specDict["method"]))

        self.candidate = encoding.create_solution_expression(architectureDict, solutionDict)


    def test_candidate(self):
        if "acceptanceCriterion" not in self.specDict:
            return True


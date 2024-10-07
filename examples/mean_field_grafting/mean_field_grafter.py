from tnreason import encoding, engine

import numpy as np


class MeanField:
    def __init__(self, networkCores, importanceList, partitionColorDict, partitionShapeDict):
        self.networkCores = networkCores
        self.importanceList = importanceList

        self.partitionColorDict = partitionColorDict
        self.partitionShapeDict = partitionShapeDict

        ## Initialize MeanCores from uniform distribution
        self.meanCores = {parKey: encoding.create_trivial_core(parKey, partitionShapeDict[parKey],
                                                               partitionColorDict[parKey]).multiply(
            1 / np.prod(self.partitionShapeDict[parKey])) for parKey
                          in partitionColorDict}

    def update_meanCore(self, upKey, temperature=1):

        oldMean = self.meanCores[upKey].clone()

        contracted = engine.contract(
            {**{secKey: self.meanCores[secKey] for secKey in self.meanCores if secKey != upKey},
             **self.networkCores, **importanceList[0][0]
             }, openColors=self.partitionColorDict[upKey]).multiply(importanceList[0][1])
        for cores, weight in importanceList[1:]:
            contracted.sum_with(
                engine.contract({**{secKey: self.meanCores[secKey] for secKey in self.meanCores if secKey != upKey},
                                 **self.networkCores, **cores
                                 }, openColors=self.partitionColorDict[upKey]).multiply(weight)
            )

        contracted = contracted.multiply(1 / temperature)
        self.meanCores[upKey] = engine.get_core("NumpyTensorCore")(values=np.exp(contracted.values),
                                                                   colors=contracted.colors).normalize()

        angle = engine.contract({"old": oldMean, "new": self.meanCores[upKey]}, openColors=[])
        return angle.values

    def anneal(self, temperaturePattern):
        angles = np.empty(shape=(len(temperaturePattern), len(self.partitionColorDict)))
        for i, temperature in enumerate(temperaturePattern):
            for j, upKey in enumerate(self.partitionColorDict):
                angles[i, j] = self.update_meanCore(upKey, temperature=temperature)
        return angles


if __name__ == "__main__":
    from tnreason import knowledge

    neuronDict = {"neur1": [["imp"],
                            ["neur2"],
                            ["a3", "a2"]],
                  "neur2": [["id", "not"],
                            ["a3", "a2"]]
                  }

    #    engine.draw_factor_graph(encoding.create_architecture(neuronDict, ["neur1"]))

    trueKB = knowledge.HybridKnowledgeBase(
        weightedFormulas={"w1": ["imp", "a1", "a2", 1],
                          "w2": ["a3", 0]}
    )
    samples = knowledge.InferenceProvider(trueKB).draw_samples(10)
    empDist = knowledge.EmpiricalDistribution(samples, ["a1", "a2", "a3"])

    importanceList = [(empDist.create_cores(), 1 / empDist.get_partition_function()), ({}, -1 / 8)]

    meanField = MeanField(networkCores=encoding.create_architecture(neuronDict, headNeurons=["neur1"]),
                          importanceList=importanceList,
                          partitionColorDict={"co1": ["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar"],
                                              "co2": ["neur2_actVar", "neur2_p0_selVar"]},
                          partitionShapeDict={"co1": [1, 1, 2],
                                              "co2": [2, 2]})

    print(meanField.anneal([1 for i in range(10)]))
    print(meanField.anneal([0.1 for i in range(10)]))
    print(meanField.anneal([0.01 for i in range(10)]))
    print(meanField.meanCores["co1"].values)

    exit()

    ## Comparison
    posPhase = engine.contract(
        {**importanceList[0][0], **encoding.create_architecture(neuronDict, headNeurons=["neur1"])},
        openColors=["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar"]).multiply(importanceList[0][1])
    negPhase = engine.contract(
        {**importanceList[1][0], **encoding.create_architecture(neuronDict, headNeurons=["neur1"])},
        openColors=["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar"]).multiply(importanceList[1][1])
    gradient = posPhase.sum_with(negPhase)
    print(gradient.values)

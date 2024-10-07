from tnreason.algorithms import energy_based_algorithms as eba

from tnreason import knowledge, encoding

neuronDict = {"neur1": [["imp"],
                        ["neur2"],
                        ["a3", "a2"]],
              "neur2": [["not", "id"],
                        ["a3", "a2"]]
              }

#    engine.draw_factor_graph(encoding.create_architecture(neuronDict, ["neur1"]))

trueKB = knowledge.HybridKnowledgeBase(
    weightedFormulas={"w1": ["imp", "a1", "a2", 1],
                      "w2": ["a3", 0]}
)
samples = knowledge.InferenceProvider(trueKB).draw_samples(10)
empDist = knowledge.EmpiricalDistribution(samples, ["a1", "a2", "a3"])

statCores = encoding.create_architecture(neuronDict, headNeurons=["neur1"])

energyDict = {"pos": [{**empDist.create_cores(), **statCores}, 1 / empDist.get_partition_function()],
              "neg": [{**statCores}, -1 / 8]}
temperatureList = [1-0.02*i for i in range(50)]
gibbser = eba.EnergyGibbs(energyDict,
                          ["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar", "neur2_actVar", "neur2_p0_selVar"],
                          dimDict={
                              "neur1_actVar": 1, "neur1_p0_selVar": 1, "neur1_p1_selVar": 2, "neur2_actVar": 2,
                              "neur2_p0_selVar": 2
                          })

gibbser.annealed_sample(temperatureList)
print(gibbser.sample)

mfApproximator = eba.EnergyMeanField(energyDict, ["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar", "neur2_actVar",
                                                  "neur2_p0_selVar"],
                                     dimDict={
                                         "neur1_actVar": 1, "neur1_p0_selVar": 1, "neur1_p1_selVar": 2,
                                         "neur2_actVar": 2,
                                         "neur2_p0_selVar": 2
                                     },
                                     partitionColorDict={"co1": ["neur1_actVar", "neur1_p0_selVar", "neur1_p1_selVar"],
                                                         "co2": ["neur2_actVar", "neur2_p0_selVar"]}
                                     )
mfApproximator.anneal(temperatureList)
print(mfApproximator.draw_sample())

from tnreason import encoding
from tnreason import engine

import numpy as np

neuronCores = encoding.get_neuron_cores(
    connectiveList=["imp"],
    candidatesDict={"pos1": ["a1"],
                    "pos2": ["a2", "a3"]},
    name="cracyNeuron"
)
headCore = encoding.get_head_core("cracyNeuron", "truthEvaluation")

dataNum = 4
data = np.zeros(shape=(2, 2, 2, dataNum))
data[1, 1, 0, 0] = 1
data[1, 1, 0, 1] = 1
data[1, 1, 1, 2] = 1
data[1, 1, 0, 3] = 1
pos_phase = ({"dataTensor": engine.get_core()(values=data, colors=["a1", "a2", "a3", "dat"])}, 1)

from tnreason.network import gibbs

sampler = gibbs.Gibbs(networkCores={**neuronCores,
                                    **headCore
                                    },
                      importanceList=[pos_phase])

sampler.ones_initialization(["pos1core", "pos2core"], {"pos1core": 1, "pos2core": 2},
                            {"pos1core": ['cracyNeuron_pos1_selControl'], "pos2core": ['cracyNeuron_pos2_selControl']})
sampler.alternating_sampling(["pos1core", "pos2core"])

print(sampler.networkCores["pos2core"].values)

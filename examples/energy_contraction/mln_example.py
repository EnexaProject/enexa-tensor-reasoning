from examples.energy_contraction import energy_based_algorithms as eba

from tnreason import encoding


energyDict = {
    "w1": [{**encoding.create_raw_formula_cores(["imp", "a", "b"]),
            **encoding.create_head_core(["imp", "a", "b"], headType="truthEvaluation")}, -1],
    "w2": [{**encoding.create_raw_formula_cores(["xor", "b", "c"]),
            **encoding.create_head_core(["xor", "b", "c"], headType="truthEvaluation")}, 1]
}

sampler = eba.EnergyMeanField(energyDict, colors=["a", "b", "c"], dimDict={"a": 2, "b": 2, "c": 2},
                          partitionColorDict={"co1": ["a", "b"], "co2": ["c"]})
sampler.update_meanCore("co1")
sampler.anneal([0.1 * i + 0.1 for i in range(10)])
print(sampler.draw_sample())

gibbser = eba.EnergyGibbs(energyDict, colors=["a", "b", "c"], dimDict={"a": 2, "b": 2, "c": 2})
gibbser.annealed_sample([11 for i in range(5)])
print(gibbser.sample)
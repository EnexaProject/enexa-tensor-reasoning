
from tnreason.model import sampling

expressionsDict = {
    "e0": [["a1", "imp", "a2"], 2.123],
   # "e1": [["a4", "eq", ["not", "a1"]], 2],
   # "e2": [["a4", "xor", ["a5", "eq", "a1"]], 2],
   # "e3": [["a6", "or", ["not", "a1"]], 2],
   # "e4": ["a3", 100]
}

sampler = sampling.GibbsSampler(expressionsDict,
                                categoricalConstraintsDict={"c1": ["a1", "a2", "a8"]})

sampler.compute_marginalized_distributions()

print(sampler.simulated_annealing_gibbs(["a1","a2","a8"], annealingPattern=[(3,1)]))






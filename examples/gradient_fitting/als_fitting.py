from tnreason import knowledge, encoding
import numpy as np

from tnreason import engine
from tnreason.algorithms import alternating_least_squares as als


#### ! MISSING PARTITION FUNCTIONS ! ####

architecture = {
    "n1" : [["imp","or"],
            ["a","b"],
            ["a","b"]]
}

currentModel = knowledge.HybridKnowledgeBase(weightedFormulas={
    "w1" : ["or","a","b", 0.2]
})

generatingModel = knowledge.HybridKnowledgeBase(weightedFormulas={
    "w1" : ["or","a","b", 0.2],
    "w2" : ["imp","a","b", 2],
    "w3" : ["c", 0.3]
})
samples = knowledge.InferenceProvider(generatingModel).draw_samples(10)

negativeCores = {**encoding.create_formulas_cores(currentModel.weightedFormulas),
                 **encoding.create_architecture(architecture)}

positiveCores = {**encoding.create_data_cores(samples, ["a","b","c"]),
                 **encoding.create_architecture(architecture)}

#arDict = encoding.create_architecture(architecture)
#print([(arDict[name].colors, arDict[name].values.shape) for name in arDict])


networkCores = {
    "n_a": encoding.create_random_core("n_a", [2, 2], ['n1_actVar_tbo', 'n1_p0_selVar_tbo']),
    "n_b": encoding.create_random_core("n_b", [2, 2], ['n1_p0_selVar_tbo', 'n1_p1_selVar_tbo']),
    "n_c":  encoding.create_random_core("n_c", [2, 2, 2], ['n1_actVar_tbo', 'n1_p0_selVar_tbo', 'n1_p1_selVar_tbo']),
    "triv" : encoding.create_trivial_core("triv", shape=[2,2,2,2,2,2], colors=['n1_actVar_tbo', 'n1_p0_selVar_tbo', 'n1_p1_selVar_tbo', 'n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'])
}
fitter = als.ALS(networkCores=networkCores, importanceColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'],
        targetList=[(positiveCores,1),(negativeCores,-1)])
residua = fitter.alternating_optimization(["n_a","n_b"], computeResiduum=True)

print(residua)


gradient = engine.contract(negativeCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar']).multiply(-1).sum_with(
    engine.contract(positiveCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'])
)

fitted = engine.contract(fitter.networkCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'])

print(np.linalg.norm(
    gradient.values
))

print(np.linalg.norm(
    gradient.sum_with(fitted.multiply(-1)).values
))
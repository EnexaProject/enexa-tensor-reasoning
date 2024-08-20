from tnreason import knowledge, encoding, algorithms, engine
import max_calibrator as mc

architecture = {
    "n1" : [["imp","or"],
            ["a","b","c"],
            ["a","b","c"]]
}

currentModel = knowledge.HybridKnowledgeBase(weightedFormulas={
    "w1" : ["or","a","b", 0.2],
    "triv" : ["c",0] # Important to get right partition functions
})

generatingModel = knowledge.HybridKnowledgeBase(weightedFormulas={
    "w1" : ["or","a","b", 0.2],
    "w2" : ["imp","a","b", 12],
    "w3" : ["c", 0.3]
})
sampleNum = 100
samples = knowledge.InferenceProvider(generatingModel).draw_samples(sampleNum)

negativeCores = {**encoding.create_formulas_cores(currentModel.weightedFormulas),
                 **encoding.create_architecture(architecture, headNeurons=["n1"])}

positiveCores = {**encoding.create_data_cores(samples, ["a","b","c"]),
                 **encoding.create_architecture(architecture, headNeurons=["n1"])}

networkCores = {
    "n_a": encoding.create_random_core("n_a", [2, 3], ['n1_actVar_tbo', 'n1_p0_selVar_tbo']),
    "n_b": encoding.create_random_core("n_b", [2, 3], ['n1_actVar_tbo', 'n1_p1_selVar_tbo']),
    "n_c":  encoding.create_random_core("n_c", [3, 3], ['n1_p0_selVar_tbo', 'n1_p1_selVar_tbo']),
    "triv" : encoding.create_trivial_core("triv", shape=[2,3,3,2,3,3],
                                          colors=['n1_actVar_tbo', 'n1_p0_selVar_tbo', 'n1_p1_selVar_tbo', 'n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'])
}

fitter = algorithms.ALS(networkCores=networkCores, importanceColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar'],
        targetList=[(positiveCores,1/sampleNum),(negativeCores,-1/currentModel.get_partition_function())])
residua = fitter.alternating_optimization(["n_a","n_b","n_c"], computeResiduum=True)

calibrationClusters = {
    "c0" : {"n_a" : fitter.networkCores["n_a"]},
    "c1" : {"n_b" : fitter.networkCores["n_b"]},
    "c2" : {"n_c" : fitter.networkCores["n_c"]},
            }
maximizer = mc.MaxCalibrator(calibrationClusters)
maximizer.max_propagation([("c0","c1"),("c0","c2"),("c1","c0"),("c1","c2")])
maximizer.get_max_assignment(["c2","c1","c0"])

print(maximizer.max_assignment)
print(encoding.create_solution_expression(architecture,{key.split("_tbo")[0] : maximizer.max_assignment[key] for key in maximizer.max_assignment}))


gradient = engine.contract(negativeCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar']).multiply(-1/currentModel.get_partition_function()).sum_with(
    engine.contract(positiveCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar']).multiply(1/sampleNum)
)

#print(engine.contract(negativeCores, openColors=['n1_actVar', 'n1_p0_selVar', 'n1_p1_selVar']).values)
#engine.draw_factor_graph(negativeCores)
## CHECK

import numpy as np

maxAssignment = np.unravel_index(np.argmax(gradient.values), gradient.values.shape)
assignment = {color : maxAssignment[i] for i, color in enumerate(gradient.colors)}

print(gradient.values)
print(encoding.create_solution_expression(architecture,{key : assignment[key] for key in assignment}))

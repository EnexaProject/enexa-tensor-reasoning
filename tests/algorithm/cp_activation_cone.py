from tnreason import algorithms
from tnreason import knowledge

testKB = knowledge.load_kb_from_yaml("../knowledge/test_kb.yaml")
cores = testKB.create_cores()
print(cores.keys())

cp = algorithms.ConstraintPropagator(cores)
cp.initialize_domainCoresDict()
cp.propagate_cores()
print(cp.domainCoresDict)
print(cp.binaryCoresDict)
print(cp.find_variable_cone(variables=["a1"]))

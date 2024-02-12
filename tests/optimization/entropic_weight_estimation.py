from tnreason.model import generate_test_data as gtd
from tnreason.optimization import entropy_maximization as enm

formulaDict = {
    "f1": ["a1", 2],
    "f2": [["a2", "and", "a3"], 100],
    "f3": [["a1", "imp", "a2"], 2]
}

sampleDf = gtd.generate_sampleDf(formulaDict, 100)

satCounter = enm.EmpiricalCounter(sampleDf)
satisfactionDict = {
    key : satCounter.get_empirical_satisfaction(formulaDict[key][0]) for key in formulaDict
}

maximizer = enm.EntropyMaximizer(formulaDict=formulaDict, satisfactionDict=satisfactionDict, factDict={"c0": "a4"})
maximizer.independent_estimation()
maximizer.formula_optimization("f1")
maximizer.alternating_optimization(10)
maximizer.fact_identification()
maximizer.soften_facts(10.2)

print(maximizer.formulaDict)
print(maximizer.satisfactionDict)

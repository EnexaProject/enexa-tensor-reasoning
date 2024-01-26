from tnreason.model import generate_test_data as gtd
from tnreason.optimization import entropy_maximization as enm

formulaDict = {
    "f1": ["a1", 2],
    "f2": [["a2", "and", "a3"], 100],
    "f3": [["a1", "imp", "a2"], 2]
}

sampleDf = gtd.generate_sampleDf(formulaDict, 100)

maximizer = enm.EntropyMaximizer(formulaDict=formulaDict, sampleDf=sampleDf, factDict={"c0": "a4"})
maximizer.formula_optimization("f1")

maximizer.fact_identification()
maximizer.soften_facts(10.2)

print(maximizer.formulaDict)
print(maximizer.satisfactionDict)

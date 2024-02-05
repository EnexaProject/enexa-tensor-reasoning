from tnreason.learning import stub_clause_learner as cll


skeletonExpression = cll.create_skeletonExpression(["P1","P2"],["P3"])
print(skeletonExpression)

candidatesDict = {
    "P1" : ["a","b"],
    "P2" : ["b","c"],
    "P3" : ["a"]
}

learnedFormulaDict = {
    "f0": ["b", 10],
    "f1": [["not", ["a", "and", "b"]], 5],
    "f2": ["c", 2]
}
import tnreason.model.generate_test_data as gtd
sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)

learner = cll.ClauseLearnerMLE(skeletonExpression, candidatesDict)
learner.load_sampleDf(sampleDf)

learner.compute_clause_gradient("P1")
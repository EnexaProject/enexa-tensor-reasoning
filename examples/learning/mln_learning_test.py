import pandas as pd

from tnreason.learning import mln_learning as mlnl
from tnreason.model import generate_test_data as gtd
from tnreason.logic import expression_calculus as ec
from tnreason.logic import expression_generation as eg

## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 1.5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)", "verbuchtDurch(z,q)"],"Umsatzerloese(q)", 1.5]
}
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}

dataNum = 10
savePath = "./examples/learning/synthetic_test_data/synthetic_accounting/"
regenerate = True
if regenerate:
    sampleDf = gtd.generate_sampleDf(example_expression_dict, sampleNum=dataNum, chainSize=10)
    #sampleDf.to_csv(savePath + "generated_sampleDf.csv")
else:
    sampleDf = pd.read_csv(savePath + "generated_sampleDf.csv", index_col=0).astype("int64")

learner = mlnl.SampleBasedMLNLearner(sampleDf)

skeletonExpression = ["P1", "and", "P2"]  # ,"and","R2(x,z)"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)", "hatBelegzeile(x,z)"],
    "P2": ["hatLeistungserbringer(x,y)", "Bautischlerei(y)"],
    "P3": ["Moebel(z)"],
    "P4": ["Moebel(z)"],
}

learner.learn(saveMod="imp",skeletonExpression=skeletonExpression,positiveExpression="Ausgangsrechnung(x)",candidatesDict=candidatesDict,
              boostNum=2, refinementCriterion="weight>0.6", acceptanceCriterion="weight>0.6", refinementNum=2, balance=False)


learner.learn_implication("Ausgangsrechnung(x)",skeletonExpression,candidatesDict,acceptanceCriterion="weight>0.1,empRate>0.9")
learner.learn_equivalence("Ausgangsrechnung(x)",skeletonExpression,candidatesDict,acceptanceCriterion="weight>0.1,empRate>0.9")

skeletonExpression2 = ["not",[["not","P2"],"and","P1"]]
candidatesDict2 = {
    "P1": ["versandterBeleg(y,x)", "Unterschrank(z)"],
    "P2": ["Bautischlerei(y)","Moebel(z)"],
    "P3": ["Moebel(z)"],
    "P4": ["ja"]
}
learner.learn_tautology(skeletonExpression2,candidatesDict2)


learner.alternating_weight_optimization(10)
print(learner.weightedFormulas)
model = learner.generate_mln()
#model.visualize()
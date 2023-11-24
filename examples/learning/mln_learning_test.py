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

dataNum = 1000
savePath = "./examples/learning/synthetic_test_data/synthetic_accounting/"
regenerate = True
if regenerate:
    sampleDf = gtd.generate_sampleDf(example_expression_dict, sampleNum=dataNum, chainSize=10)
    #sampleDf.to_csv(savePath + "generated_sampleDf.csv")
else:
    sampleDf = pd.read_csv(savePath + "generated_sampleDf.csv", index_col=0).astype("int64")

learner = mlnl.AtomicMLNLearner()
learner.load_sampleDf(sampleDf)

skeletonExpression = ["P1", "and", "P2"]  # ,"and","R2(x,z)"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)", "hatBelegzeile(x,z)"],
    "P2": ["hatLeistungserbringer(x,y)", "Bautischlerei(y)"],
}
learner.learn_implication("Ausgangsrechnung(x)",skeletonExpression,candidatesDict,acceptanceCriterion="weight>0.1,empRate>0.9")

skeletonExpression2 = ["not",[["not","P2"],"and","P1"]]
candidatesDict2 = {
    "P1": ["versandterBeleg(y,x)", "Unterschrank(z)"],
    "P2": ["Bautischlerei(y)","Moebel(z)"],
}
learner.learn_tautology(skeletonExpression2,candidatesDict2)

model = learner.generate_mln()
#model.visualize()
import pandas as pd

from tnreason.learning import mln_learning as mlnl
from tnreason.logic import expression_calculus as ec
from tnreason.logic import expression_generation as eg

## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 1.5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)", "verbuchtDurch(z,q)"],"Umsatzerloese(q)", 1.5]
}


savePath = "./examples/learning/synthetic_test_data/synthetic_accounting/"
sampleDf = pd.read_csv(savePath + "generated_sampleDf.csv", index_col=0).astype("int64")

learner = mlnl.AtomicMLNLearner()
learner.load_sampleDf(sampleDf)

skeletonExpression = ["P1", "and", "P2"]  # ,"and","R2(x,z)"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)", "hatBelegzeile(x,z)"],
    "P2": ["hatLeistungserbringer(x,y)", "Bautischlerei(y)"],
}
learner.learn_equivalence("Ausgangsrechnung(x)",skeletonExpression,candidatesDict)

skeletonExpression2 = ["not",[["not","P2"],"and","P1"]]
candidatesDict2 = {
    "P1": ["versandterBeleg(y,x)", "hatBelegzeile(x,z)", "Unterschrank(z)"],
    "P2": ["hatLeistungserbringer(x,y)", "Bautischlerei(y)","Moebel(z)"],
}
learner.learn_tautology(skeletonExpression2,candidatesDict2)

model = learner.generate_mln()
#model.visualize()
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
sampleDf = pd.read_csv(savePath + "generated_sampleDf.csv", index_col=0)
#print(sampleDf)
learner = mlnl.MLNLearner()

skeletonExpression = ["P1", "and", "P2"]  # ,"and","R2(x,z)"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)", "hatBelegzeile(x,z)"],
    "P2": ["Ausgangsrechnung(x)", "Bautischlerei(y)"],
}
positiveCore = ec.evaluate_expression_on_sampleDf(sampleDf, ["versandterBeleg(y,x)","and","Ausgangsrechnung(x)"])
negativeCore = positiveCore.negate()
learner.learn_formula(skeletonExpression,candidatesDict,sampleDf,positiveCore,negativeCore)

learner.generate_mln()

learner.graphicalModel.vi
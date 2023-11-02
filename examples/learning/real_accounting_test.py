import pandas as pd
import numpy as np

from tensor_reasoning.representation import sampledf_to_factdf as stof, sampledf_to_pairdf as stop

from tensor_reasoning.learning import check_cores as checkcc, expression_learning as el

## DATA LOADING ##

sampleDf = pd.read_csv("S:/_MariaSF/extractedDataFrame/extracted_20230925_150557.csv", index_col=0)
print(sampleDf.var())

prefix = ""
factDf = stof.sampleDf_to_factDf(sampleDf, prefix=prefix)
pairDf = stop.sampleDf_to_pairDf(sampleDf, prefix=prefix)

individualsDict = {
    "beleg": np.unique(pairDf["beleg"].values),
    "branche": np.unique(pairDf["branche"].values),
    "kto": np.unique(pairDf["kto"].values)
}

## EXPRESSION LEARNING ##

# skeletonExpression=[["C3(branche)","and","C2(kto)"],"and","C1(branche)"]
skeletonExpression = ["random1(beleg)", "and", "C2(kto)"]
candidatesDict = {
    "random1(beleg)": ["tev:Ausgangsrechnung(beleg)", "tev:Eingangsrechnung(beleg)"],
    "C2(kto)": ['tev:Umsatzerlöse_gesamt(kto)',
                'tev:Gesamtkostenverfahren_-_Kosten(kto)', 'tev:Gemeinsamer_Teil(kto)'],
    "C1(branche)": ['tev:Branche_A(branche)', 'tev:Branche_B(branche)', 'tev:Branche_D(branche)',
                    'tev:Branche_C(branche)'],
    "C3(branche)": ['tev:Branche_A(branche)', 'tev:Branche_B(branche)', 'tev:Branche_D(branche)',
                    'tev:Branche_C(branche)']
}

learner = el.ExpressionLearner(skeletonExpression)
learner.generate_fixedCores_factDf(factDf, individualsDict, candidatesDict, prefix="")
learner.random_initialize_variableCoresDict()
learner.generate_targetCore_pairDf(pairDf, individualsDict)
learner.set_filterCore(learner.targetCore)

learner.als(1)
print("########### Variable Cores After ALS are:")
print(checkcc.review_coreDict(learner.variablesCoresDict))
learner.get_solution()
print("The solution expression is:")
print(learner.solutionExpression)

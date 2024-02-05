import pandas as pd
import numpy as np

from tnreason.learning import expression_learning as el
import tests.check_cores as chcc

import tnreason.logic.coordinate_calculus as cc

## Based on BBB generated data (sampleDf)

skeleton = ["R2(x,z)","and",["C1(z)","and","R1(y,x)"]]
learner = el.ExpressionLearner(skeleton)

sampleDf = pd.read_csv("./tests/assets/bbb_generated.csv",index_col=0).head(100)
learner.generate_fixedCores_sampledf(sampleDf)
learner.random_initialize_variableCoresDict()

datanum = 100
target = cc.CoordinateCore(np.random.binomial(n=1,p=0.4,size=(datanum,datanum,datanum)),["x","y","z"])
learner.set_targetCore(target,
                       targetIsFilter=True)
learner.als(sweepnum=2)
learner.get_solution()

## Based on BBB generated data (factDf)

skeleton = [["C1(a)", "and", "R1(a,b)"], "and", ["not", "C1(a)"]]
learner = el.VariableLearner(skeleton)

individualsDict = {
    "a": ["http://datev.de/ontology#ocr_item_5f3d5dc5-c55e-d6bc-50bf-7071e0f90d61_6"],
    "b": ["http://datev.de/ontology#ocr_item_5f3d5dc5-c55e-d6bc-50bf-7071e0f90d61_6"]
}

candidatesDict = {
    "C1(a)": ["http://datev.de/ontology#OcrItemsSubClass_65504"],
    "R1(a,b)": ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
}

learner.generate_fixedCores_turtlePath("./tests/assets/bbb_generated.ttl", 1e3, individualsDict, candidatesDict)
learner.random_initialize_variableCoresDict()

targetCore = cc.CoordinateCore(np.random.normal(size=(1, 1)), ["a", "b"])
learner.set_targetCore(targetCore,
                       targetIsFilter=True)
learner.als(sweepnum=1)
learner.get_solution()


## Based on random generated Data (sampleDf)

skeleton = ["R2(x,z)","and",["C1(x)","and","R1(y,x)"]]
learner = el.ExpressionLearner(skeleton)

datanum = 100
columns = ['tev:Ausgangsrechnung(x)', 'tev:Eingangsrechnung(x)', 'tev:versandterBeleg(y,x)', 'tev:hatBelegzeile(x,z)',
           "sledz(x)", "jaszczur(y,x)", "sokol(y,x)", "sikorka(x,z)"]
skeleton = ["R2(x,z)","and",["C1(x)","and","R1(y,x)"]]
randomDf = pd.DataFrame(np.random.binomial(1,0.8,size=(datanum,len(columns))),columns = columns)

learner = el.ExpressionLearner(skeleton)
learner.generate_fixedCores_sampledf(randomDf)
learner.random_initialize_variableCoresDict()

positiveExampleCore = cc.CoordinateCore(np.random.binomial(1,0.3, size =(datanum,datanum,datanum)),["x","y","z"])
negativeExampleCore = cc.CoordinateCore(np.random.binomial(1,0.3, size =(datanum,datanum,datanum)),["x","y","z"])
learner.generate_target_and_filterCore_from_exampleCores(positiveExampleCore,negativeExampleCore)

chcc.review_core(learner.filterCore)
chcc.review_core(learner.targetCore)

target = cc.CoordinateCore(np.random.normal(size=(datanum,datanum,datanum)),["x","y","z"])
learner.set_targetCore(target,
                       targetIsFilter=True)
learner.als(5)
learner.get_solution()

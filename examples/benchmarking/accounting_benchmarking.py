import numpy as np
from matplotlib import pyplot as plt

from tnreason.logic import expression_generation as eg
from examples.benchmarking import recovery_experiment as re

## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)", "verbuchtDurch(z,q)"],"Umsatzerloese(q)", 1.5],
}

## We transform the rules into propositional formulas containing negations and conjunctions only
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}

example_expression_dict["r3"]=[["not","Ausgangsrechnung(x)"], 2]

experimentDict = {
    "No Marker": [
        "Thing",
        ['not', ['H', 'and', [['not', 'P1'], 'and', ['not', 'P2']]]],
        ['not', ['Ausgangsrechnung(x)', 'and',[['not', 'versandterBeleg(y,x)'], 'and', ['not', 'hatLeistungserbringer(x,y)']]]],
    ],
    "Full Marker" : [
        ['not', ['Ausgangsrechnung(x)', 'and', [['not', 'versandterBeleg(y,x)'], 'and', ['not', 'hatLeistungserbringer(x,y)']]]],
        ['not', ['H', 'and', [['not', 'P1'], 'and', ['not', 'P2']]]],
        None
    ],
    "Head Marker" :  [
        "Ausgangsrechnung(x)",
        ["P1","and","P2"],
        ["versandterBeleg(y,x)","and","hatLeistungserbringer(x,y)"]
    ]
}

candidatesDict = {
    "H": ["Ausgangsrechnung(x)","Bautischlerei(y)"],
    "P1": ["versandterBeleg(y,x)","hatBelegzeile(x,z)"],
    "P2": ["hatLeistungserbringer(x,y)", "verbuchtDurch(z,q)"],
}

repetitions = 10
dataNums = range(500,550,50)
experimentKeys = experimentDict.keys()

successRates = np.empty((len(experimentKeys),len(dataNums)))
for exInd, exKey in enumerate(experimentKeys):
  for datInd, dataNum in enumerate(dataNums):
    rate = 0
    for rep in range(repetitions):
        positiveExpression, skeletonExpression, trueExpression = experimentDict[exKey]
        rate += re.sampleDf_experiment(example_expression_dict, dataNum, skeletonExpression, candidatesDict, positiveExpression, trueExpression)
    successRates[exInd,datInd] = rate/repetitions


plt.imshow(successRates, cmap = "coolwarm", vmin = 0, vmax=1)
plt.colorbar()
plt.yticks(range(len(experimentKeys)),list(experimentKeys))
plt.xticks(range(len(dataNums)),dataNums)
plt.xlabel("Number of Samples")
plt.savefig("./examples/benchmarking/diagrams/ausgangsrechnung_differentmodes.png")
plt.show()

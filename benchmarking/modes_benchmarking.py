import numpy as np
from matplotlib import pyplot as plt

from tnreason.logic import expression_generation as eg
from benchmarking import recovery_experiment as re

## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 1.5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)", "verbuchtDurch(z,q)"],"Umsatzerloese(q)", 1.5]
}

## We transform the rules into propositional formulas containing negations and conjunctions only
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}d

positiveExpression = ['not', ['Ausgangsrechnung(x)', 'and', [['not', 'versandterBeleg(y,x)'], 'and', ['not', 'hatLeistungserbringer(x,y)']]]]
skeletonExpression = ['not', ['H', 'and', [['not', 'P1'], 'and', ['not', 'P2']]]]
candidatesDict = {
    "H": ["Ausgangsrechnung(x)","Bautischlerei(y)"],
    "P1": ["versandterBeleg(y,x)","hatBelegzeile(x,z)"],
    "P2": ["hatLeistungserbringer(x,y)", "verbuchtDurch(z,q)"],
}

repetitions = 10
dataNums = range(5,20,5)

successRates = np.empty(len(dataNums))
for i, dataNum in enumerate(dataNums):
    rate = 0
    for rep in range(repetitions):
        rate += re.sampleDf_experiment(example_expression_dict, dataNum, skeletonExpression, candidatesDict, positiveExpression)
    successRates[i] = rate/repetitions

plt.scatter(dataNums,successRates, marker = "+")
plt.xlabel("Number of Data")
plt.ylabel("Success rate of {} repetitions".format(repetitions))
plt.ylim(0,1)
plt.savefig("./benchmarking/diagrams/ausgangsrechnung.png")
plt.show()


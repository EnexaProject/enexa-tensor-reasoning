import numpy as np

from tnreason.logic import expression_generation as eg
from benchmarking import recovery_experiment as re

import matplotlib.pyplot as plt

trueFormula = ['not',
               [['versandterBeleg(y,x)', 'and', 'hatLeistungserbringer(x,y)'], 'and', ['not', 'Ausgangsrechnung(x)']]]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)",
            "verbuchtDurch(z,q)"], "Umsatzerloese(q)", 1.5]
}
rawDict = {key: [eg.generate_list_from_rule(value[0], value[1]), value[2]] for (key, value) in
           example_rule_dict.items()}

positiveExpression = "Ausgangsrechnung(x)"

skeletonExpression = ["P1", "and", "P2"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)", "Unterschrank(z)", "Bautischlerei(y)"],
    "P2": ["hatLeistungserbringer(x,y)", "hatBelegzeile(x,z)", "Umsatzerloese(q)"]
}

weights = range(2, 7, 1)
sampleNums = range(20, 320, 20)
successes = np.empty(shape=(len(weights), len(sampleNums)))
precisions = np.empty(shape=(len(weights), len(sampleNums)))
for i, weight in enumerate(weights):
    for j, sampleNum in enumerate(sampleNums):
        expressionDict = rawDict.copy()
        expressionDict["groundtruth"] = [trueFormula, weight]
        suc, prec = re.weight_recovery(expressionDict, trueFormula, weight, sampleNum, skeletonExpression,
                                       candidatesDict, positiveExpression, testMod="imp")
        successes[i, j] = suc
        precisions[i, j] = prec

plt.imshow(successes, cmap="Greys", vmax=0, vmin=1)
plt.title("Success of the Formula Recovery")
plt.colorbar()
plt.ylabel("Weights")
plt.xlabel("Samples")
plt.yticks(range(len(weights)), weights)
plt.xticks(range(len(sampleNums)), sampleNums)
plt.savefig("./benchmarking/diagrams/success_rate.png")
plt.close()

plt.imshow(precisions, cmap="coolwarm", vmax=10, vmin=0)
plt.title("Absolute Error of the Weight Estimation")
plt.colorbar()
plt.ylabel("Weights")
plt.xlabel("Samples")
plt.yticks(range(len(weights)), weights)
plt.xticks(range(len(sampleNums)), sampleNums)

masked_array = np.ma.masked_where(precisions != -1, precisions)
plt.imshow(masked_array, cmap="Greys", vmax=-1, vmin=-2)

plt.savefig("./benchmarking/diagrams/precision.png")
plt.show()


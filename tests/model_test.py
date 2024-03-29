from tnreason.learning import mln_learning as lmln

from tnreason.model import tensor_network_mln as tmln

from tnreason.optimization import weight_estimation as wees

from tnreason.logic import expression_generation as eg
from tnreason.logic import expression_calculus as ec
from tnreason.logic import coordinate_calculus as cc

import numpy as np
import pandas as pd

example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 15],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 15],
}
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}
print(example_expression_dict)

sampleNum = 100
generator = tmln.TensorMLN(example_expression_dict)
sampleDf = generator.generate_sampleDf(sampleNum)

learner = lmln.SampleBasedMLNLearner()
learner.load_sampleDf(sampleDf)

atomDict = {
    str(col) : cc.CoordinateCore(sampleDf[col].values.flatten(), ["j"])
    for col in list(sampleDf.columns)
}

for key in example_expression_dict.keys():
    rule = example_expression_dict[key][0]
    calculated_rate = wees.calculate_empRate(rule, atomDict)

    print("Rule {}  is satisfied in {} cases.".format(rule, calculated_rate))
    assert np.sum(ec.evaluate_expression_on_sampleDf(sampleDf,rule).values)/sampleNum == calculated_rate
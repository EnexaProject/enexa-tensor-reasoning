import tnreason.contraction.expression_evaluation as ee

from tnreason.model import generate_test_data as gtd

from tnreason.representation import sampledf_to_cores as stoc

import numpy as np

learnedFormulaDict = {
    "f0": ["a2", 10],
    "f1": [["a1", "and", "a2"], 5],
    "f2": ["a3", 2]
}


sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 10)

expression = ["a1", "and", "a2"]
atoms = ["a1", "a2"]

## Check how many satisfactions
satisfaction = np.sum(ee.ExpressionEvaluator(expression).evaluate_on_sampleDf(sampleDf).values)
fixedCoresDict = {
    atom : stoc.create_fixedCore(sampleDf, atom) for atom in atoms
}
print(satisfaction)


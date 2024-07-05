from examples.sudoku import representation as rep
from examples.sudoku import visualization as vis

import numpy as np

from tnreason import algorithms
from tnreason import encoding

num = 2
structureCores = encoding.create_categorical_cores(rep.get_sudoku_constraints(num=num))
# preEvidence = {
#             "a_0_1_0_0_1": 1,
#             "a_0_0_0_1_0": 1,
#             "a_0_1_1_0_3": 1,
#             "a_1_0_0_0_2": 1,
#             "a_0_0_1_0_2": 1
#         }

sudoku_array = np.array([
    [3, 0, 0, 0],
    [4, 1, 3, 0],
    [2, 0, 0, 0],
    [1, 3, 0, 0],
])

catEvidence = rep.array_to_catEvidence(sudoku_array, num = 2)
preEvidence = rep.catEvidence_to_atomEvidence(catEvidence)

propagator = algorithms.ConstraintPropagator(
            {**structureCores,
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
propagator.propagate_cores()
assignmentDict = propagator.find_assignments()

array = rep.evidence_to_array(assignmentDict, 2)
vis.visualize_sudoku(array.astype(int))

print(array)

exit()


evidence_to_array(evidence, num=2)

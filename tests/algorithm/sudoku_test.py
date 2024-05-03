from tnreason.algorithms import constraint_propagation as cp
from tnreason.tensor import model_cores as crc, formula_tensors as ft

import numpy as np


def get_sudoku_constraints(num=3):
    return {**get_column_constraints(num),
            **get_row_constraints(num),
            **get_squares_constraints(num),
            **get_position_constraints(num)}


def get_column_constraints(num=3):
    categoricalConstraints = {}
    for c1 in range(num):
        for c2 in range(num):
            for n in range(num ** 2):
                catVarKey = "col_" + str(n) + "_" + str(c1) + "_" + str(c2)
                categoricalConstraints[catVarKey] = []
                for r1 in range(num):
                    for r2 in range(num):
                        categoricalConstraints[catVarKey].append(
                            "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n))

    return categoricalConstraints


def get_row_constraints(num=3):
    categoricalConstraints = {}
    for r1 in range(num):
        for r2 in range(num):
            for n in range(num ** 2):
                catVarKey = "row_" + str(n) + "_" + str(r1) + "_" + str(r2)
                categoricalConstraints[catVarKey] = []
                for c1 in range(num):
                    for c2 in range(num):
                        categoricalConstraints[catVarKey].append(
                            "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n))
    return categoricalConstraints


def get_squares_constraints(num=3):
    categoricalConstraints = {}
    for r1 in range(num):
        for c1 in range(num):
            for n in range(num ** 2):
                catVarKey = "square_" + str(n) + "_" + str(r1) + "_" + str(c1)
                categoricalConstraints[catVarKey] = []
                for r2 in range(num):
                    for c2 in range(num):
                        categoricalConstraints[catVarKey].append(
                            "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n))
    return categoricalConstraints


def get_position_constraints(num=3):
    categoricalConstraints = {}
    for r1 in range(num):
        for r2 in range(num):
            for c1 in range(num):
                for c2 in range(num):
                    catVarKey = "pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)
                    categoricalConstraints[catVarKey] = []
                    for n in range(num ** 2):
                        categoricalConstraints[catVarKey].append(
                            "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n))
    return categoricalConstraints


## Visualization
def evidence_to_array(evidenceDict, num, verbose=False):
    array = np.empty((num ** 2, num ** 2))
    for r1 in range(num):
        for r2 in range(num):
            for c1 in range(num):
                for c2 in range(num):
                    catVarKey = "pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)
                    if catVarKey in evidenceDict:
                        if verbose:
                            print(
                                "Position {} known to be {}".format(
                                    str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2),
                                    evidenceDict[catVarKey] + 1))
                        array[r1 * num + r2, c1 * num + c2] = evidenceDict[catVarKey] + 1
                    else:
                        array[r1 * num + r2, c1 * num + c2] = 0
    return array


num = 2
constraints = get_sudoku_constraints(num)

cores = {}
for key in constraints:
    cores = {**cores,
             **ft.CategoricalConstraint(constraints[key], key).get_cores()}

print("Created {} cores for sudoku with side number {}.".format(len(cores), num))

preEvidence = {
    "a_0_1_0_0_1": 1,
    "a_0_0_0_1_0": 1,

    "a_0_1_1_0_3": 1,

    "a_1_0_0_0_2": 1,

    "a_0_0_1_0_2": 1
}

propagator = cp.ConstraintPropagator(
    {**cores,
     **crc.create_evidenceCoresDict(preEvidence)},
    verbose=False
)
propagator.propagate_cores()

evidenceDict, multipleAssignmentColors, redundantCores, remainingCores = propagator.find_evidence_and_redundant_cores()

## Sanity Check: This follows already from evidence {"a_0_1_0_0_1": 1}
print(evidenceDict.keys())
## The 1 in the first position is in place 2
print(evidenceDict["square_1_0_0"] == num)

## The 1 is not in place 2
print(not evidenceDict["a_0_0_0_0_1"])

# print(evidenceDict["pos_0_0_0_0"])
resArr = evidence_to_array(evidenceDict, num)
print(resArr)

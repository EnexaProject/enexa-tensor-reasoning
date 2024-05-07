import unittest

from tnreason import algorithms
from tnreason import encoding

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


class SudokuTest(unittest.TestCase):
    def test_num2(self):
        num = 2
        structureCores = encoding.create_categorical_cores(get_sudoku_constraints(num=num))
        preEvidence = {
            "a_0_1_0_0_1": 1,
            "a_0_0_0_1_0": 1,
            "a_0_1_1_0_3": 1,
            "a_1_0_0_0_2": 1,
            "a_0_0_1_0_2": 1
        }
        propagator = algorithms.ConstraintPropagator(
            {**structureCores,
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["square_1_0_0"] == num)
        self.assertTrue(not assignmentDict["a_0_0_0_0_1"])

    def test_num3(self):
        num = 3
        structureCores = encoding.create_categorical_cores(get_sudoku_constraints(num=num))
        preEvidence = {
            "a_0_1_0_0_1": 1,
            "a_0_0_0_1_0": 1,
            "a_0_1_1_0_3": 1,
            "a_1_0_0_0_2": 1,
            "a_0_0_1_0_2": 1
        }
        propagator = algorithms.ConstraintPropagator(
            {**structureCores,
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["square_1_0_0"] == num)
        self.assertTrue(not assignmentDict["a_0_0_0_0_1"])

if __name__ == "__main__":
    unittest.main()

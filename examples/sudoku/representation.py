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

def array_to_catEvidence(array, num, verbose=False):
    evidenceDict = {}
    rearranged = array.reshape((num,num,num,num))
    for r1 in range(num):
        for r2 in range(num):
            for c1 in range(num):
                for c2 in range(num):
                    if rearranged[r1,r2,c1,c2] != 0:
                        evidenceDict["pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)] = rearranged[r1,r2,c1,c2] - 1
    return evidenceDict

def catEvidence_to_atomEvidence(catEvidence):
    return {"a_"+"_".join(key.split("_")[1:]) + "_" + str(catEvidence[key]) : 1  for key in catEvidence}
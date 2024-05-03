from tnreason import engine

import numpy as np


def create_transition_matrix(position, length, tendency_x=0.5):
    values = np.zeros(shape=(length, length, length + 1, length + 1))
    for x_pos in range(length):
        for y_pos in range(length):
            values[x_pos, y_pos, x_pos + 1, y_pos] = tendency_x
            values[x_pos, y_pos, x_pos, y_pos + 1] = 1 - tendency_x
    return engine.get_core()(values, ["x" + str(position), "y" + str(position), "x" + str(position + 1),
                                      "y" + str(position + 1)])


def create_markov_cain(positionNumber, tendency_x=0.5):
    return {"core" + str(position): create_transition_matrix(position+1, position+1, tendency_x=tendency_x) for position in
            range(positionNumber)}


def create_evidence(variable, dimension, index):
    values = np.zeros(shape=(dimension))
    values[index] = 1
    return engine.get_core()(values, [str(variable)])


if __name__ == "__main__":
    mc = create_markov_cain(3)
    res = engine.contract(method="PgmpyVariableEliminator", coreDict={**mc, "evidence": create_evidence("x3", 3, 1)}, openColors=["y3"])

    print("y2", res.values / res.values.sum())

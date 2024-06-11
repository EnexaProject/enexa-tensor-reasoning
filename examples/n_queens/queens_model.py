from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

import numpy as np
from matplotlib import pyplot as plt

import time

def get_queens_constraints(n=3):
    ## Rows
    return {**{"row_"+str(i): ["q_"+str(i)+"_"+str(j) for j in range(n)] for i in range(n)},
        **{"col_"+str(j): ["q_"+str(i)+"_"+str(j) for i in range(n)] for j in range(n)}}

def get_queens_propagator(n=3):
    return algorithms.ConstraintPropagator(encoding.create_categorical_cores(get_queens_constraints(n=n)))

def to_random_basis(binaryCore):
    ones = np.where(binaryCore.values==1)[0]
    assert len(ones)>0, ValueError("No possibilities detected.")
    return encoding.create_basis_core(binaryCore.name, binaryCore.values.shape, binaryCore.colors, (np.random.choice(ones)))


def get_random_assignment(n=3):
    propagator = get_queens_propagator(n=n)
    times = []
    currentTime = time.time()
    for colPos in range(n):
        propagator.domainCoresDict["col_"+str(colPos)+"_domainCore"] = to_random_basis(propagator.domainCoresDict["col_"+str(colPos)+"_domainCore"])
        propagator.propagate_cores(coreOrder=["col_"+str(colPos)+"_q_"+str(i)+"_"+str(colPos)+"_catCore" for i in range(n)])
        times.append(time.time()-currentTime)
        currentTime = time.time()
    return propagator.find_assignments(), times

def get_queen_positions(assignment, n=3):
    queensPositions = np.zeros(shape=(n,n))
    for colPos in range(n):
        for rowPos in range(n):
            if "q_"+str(rowPos)+"_"+str(colPos) in assignment:
                if assignment["q_"+str(rowPos)+"_"+str(colPos)] == 1:
                    queensPositions[colPos,rowPos] = 1
    return queensPositions

def draw_positions(queensPositions):
    size = queensPositions.shape[0]
    plt.imshow(queensPositions, cmap= "gray", vmin=0, vmax=1)
    plt.xticks(range(size), range(size))
    plt.yticks(range(size), range(size))
    plt.title("Random Assignment to the {} queens problem".format(size), fontsize=15)
    plt.show()


def calculate_possibilities(n):
    return engine.contract(coreDict=encoding.create_categorical_cores(get_queens_constraints(n=n)),
                    openColors=[]).values


if __name__ == "__main__":
    n = 10
    assignment, times = get_random_assignment(n=n)
    draw_positions(get_queen_positions(assignment, n=n))

    plt.scatter(range(n),times, marker="+")
    plt.title("Execution times for the steps", fontsize=15)
    plt.ylim(0)
    plt.xlim(0)
    plt.show()

    print(calculate_possibilities(6))
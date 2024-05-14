import numpy as np
import time

from tnreason import engine


def create_random_edges(nodeNum, edgeNum, edgeLength):
    nodes = range(nodeNum)
    hyperedgeList = []
    for edge in range(edgeNum):
        hyperedgeList.append(np.random.choice(nodes, edgeLength, replace=False))
    return hyperedgeList


def create_random_tensor_network(nodeNum, edgeNum, edgeLength, nodeDim):
    edges = create_random_edges(nodeNum, edgeNum, edgeLength)
    return {
        "core" + str(i): engine.get_core()(values=np.random.random([nodeDim for pos in range(len(edge))]),
                                           colors=[str(node) for node in edge]) for i, edge in enumerate(edges)
    }


def measure_contraction(method, tensorNetwork, openColors):
    try:
        startTime = time.time()
        engine.contract(method=method, coreDict=tensorNetwork, openColors=openColors)
        endTime = time.time()
    except:
        return -1
    return (endTime - startTime)


if __name__ == "__main__":
    random_network = create_random_tensor_network(18, 20, 3, 2)

    nodeNum = 10
    edgeLength = 3
    nodeDim = 2

    edgeNums = range(10,210,10)
    openNums = [5]

    import tensorflow as tf
    import torch
    import pgmpy
    
    methods = ["NumpyEinsum", "TensorFlowEinsum", "TorchEinsum", "PgmpyVariableEliminator"]
    contractionTimes = np.empty((len(edgeNums), len(openNums), len(methods)))

    for i, edgeNum in enumerate(edgeNums):
        for j, openNum in enumerate(openNums):
            network = create_random_tensor_network(nodeNum=nodeNum, edgeNum=edgeNum, edgeLength=edgeLength,
                                                   nodeDim=nodeDim)
            for k, method in enumerate(methods):
                contractionTimes[i, j, k] = measure_contraction(
                    method=method,
                    tensorNetwork=network,
                    openColors=[str(n) for n in range(openNum)])
                print(method, contractionTimes[i, j, k])

    from matplotlib import pyplot as plt

    plt.scatter(edgeNums, contractionTimes[:,0,0], marker="+", label="NumpyEinsum")
    plt.scatter(edgeNums, contractionTimes[:,0,1], marker="+", label="TensorFlowEinsum")
    plt.scatter(edgeNums, contractionTimes[:,0,2], marker="+", label="TorchEinsum")
    plt.scatter(edgeNums, contractionTimes[:,0,3], marker="+", label="PgmpyVariableEliminator")


    plt.title("Execution Time of Tensor Contractors")
    plt.ylabel("Time [s]")
    plt.xlabel("Number of Tensors")
    plt.yscale("log")
    plt.legend()

    #plt.savefig("./conTimes_{}_{}_{}.png".format(nodeNum, edgeLength, openNums[0]))
    plt.show()
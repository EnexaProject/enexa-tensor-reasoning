import networkx as nx
import matplotlib.pyplot as plt


def draw_contractionDiagram(coreDict, fontsize=10, title="Contraction Diagram"):
    ##

    coreNodes = list(coreDict.keys())

    colorNodes = []
    ccEdges = []
    for coreKey in coreDict:
        for color in coreDict[coreKey].colors:
            if color not in colorNodes:
                colorNodes.append(color)
            ccEdges.append([color, coreKey])

    graph = nx.DiGraph()
    graph.add_nodes_from(coreNodes)
    graph.add_nodes_from(colorNodes)
    graph.add_edges_from(ccEdges)

    pos = nx.spring_layout(graph)

    labels = {atom: atom for atom in coreNodes + colorNodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=fontsize)

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=colorNodes,
                           node_color='r',
                           node_size=100,
                           alpha=0.2)

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=coreNodes,
                           node_color='b',
                           node_size=300,
                           alpha=0.2)

    nx.draw_networkx_edges(graph, pos,
                           edgelist=ccEdges,
                           arrowstyle="<-",
                           width=2, alpha=1, edge_color='tab:grey')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    import numpy as np

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    draw_contractionDiagram(atomDict)

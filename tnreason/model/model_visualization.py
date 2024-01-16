import networkx as nx
from matplotlib import pyplot as plt

from tnreason.logic import expression_utils as eu


def visualize_model(expressionsDict,
                    strengthMultiplier=4,
                    strengthCutoff=10,
                    fontsize=10,
                    showFormula=False,
                    evidenceDict={},
                    pos=None):
    expressionsList = [expressionsDict[key][0] for key in expressionsDict]
    atomsList = eu.get_all_variables(expressionsList)

    ## Collect edges for position optimization
    edges = []
    for expressionKey in expressionsDict:
        for atom in eu.get_variables(expressionsDict[expressionKey][0]):
            edges.append([atom, expressionKey])
    graph = nx.Graph()
    graph.add_nodes_from(atomsList)
    graph.add_nodes_from(expressionsDict.keys())
    graph.add_edges_from(edges)
    if pos is None:
        pos = nx.spring_layout(graph)

    ## Draw Nodes
    trueColor = "blue"
    falseColor = "red"

    # Known Trues
    atomFontSize = fontsize * 100
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if bool(evidenceDict[atomKey])],
                           node_color=trueColor,
                           node_size=atomFontSize,
                           alpha=0.6)
    # Known False
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if not bool(evidenceDict[atomKey])],
                           node_color=falseColor,
                           node_size=atomFontSize,
                           alpha=0.6)

    # Unknown Atoms
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in atomsList if atomKey not in evidenceDict.keys()],
                           node_color="grey",
                           node_size=atomFontSize,
                           alpha=0.2)
    nx.draw_networkx_labels(graph, pos, {atomKey: atomKey for atomKey in atomsList}, font_size=fontsize)

    if showFormula:
        expressionLabels = {expressionKey: expressionsDict[expressionKey][0] for expressionKey in expressionsDict}
    else:
        expressionLabels = {expressionKey: expressionKey for expressionKey in expressionsDict}
    nx.draw_networkx_labels(graph, pos, expressionLabels, font_size=fontsize)

    ## Draw Edges
    colorList = ["r", "g", "b", "r", "g", "b", "r", "g", "b"]
    for i, expressionKey in enumerate(expressionsDict.keys()):
        strength = min(expressionsDict[expressionKey][1], strengthCutoff)
        drawEdges = []
        for atom in eu.get_variables(expressionsDict[expressionKey][0]):
            drawEdges.append([atom, expressionKey])
        nx.draw_networkx_edges(graph, pos,
                               edgelist=drawEdges,
                               width=strengthMultiplier * strength,
                               alpha=0.2,
                               edge_color=colorList[i])

    plt.show()
    return pos


if __name__ == "__main__":
    exDict = {
        "e0": ["a2", 1],
        "e1": [[["a2", "and", ["not", "a3"]], "and", ["a6", "and", ["not", "a7"]]], 10],
        "e2": [["a4", "and", ["not", "a2"]], 2],
        "e4": ["a5", 1],
        "e5": ["a5", 2]
    }

    visualize_model(exDict, evidenceDict={"a2": 1, "a3": 0})

    exit()

    G = nx.Graph()
    G.add_edges_from([(12, 2)])  # , (2, 3), (3, 4), (4, 1)])

    pos = {12: [10, 0], 2: [1, 0]}
    nx.draw(G, pos=pos)
    plt.show()

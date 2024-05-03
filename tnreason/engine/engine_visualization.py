import networkx as nx
import matplotlib.pyplot as plt

def draw_contractionDiagram(coreDict, fontsize=10, title="Contraction Diagram", pos=None):
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

    if pos is None:
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

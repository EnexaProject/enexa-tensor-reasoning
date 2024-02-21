import networkx as nx
from matplotlib import pyplot as plt

from tnreason.logic import expression_utils as eu


def get_edges_and_positions(expression):
    expressionString = eu.get_expression_string(expression)

    if isinstance(expression, str):
        return [], {expressionString: 0}, [expressionString]

    elif len(expression) == 2:
        preExpressionString = eu.get_expression_string(expression[1])
        edges, positions, subexpressions = get_edges_and_positions(expression[1])

        edges.append([preExpressionString, expressionString])
        positions[expressionString] = positions[preExpressionString] + 1
        subexpressions.append(expressionString)
        return edges, positions, subexpressions

    elif len(expression) == 3:
        leftExpressionString = eu.get_expression_string(expression[0])
        rightExpressionString = eu.get_expression_string(expression[2])

        leftEdges, leftPositions, leftSubexpressions = get_edges_and_positions(expression[0])
        rightEdges, rightPositions, rightSubexpressions = get_edges_and_positions(expression[2])

        leftEdges.extend(rightEdges)
        leftPositions = {**leftPositions,
                         **rightPositions}
        leftSubexpressions.extend(rightSubexpressions)

        leftSubexpressions.append(expressionString)
        leftPositions[expressionString] = max(leftPositions[leftExpressionString],
                                              leftPositions[rightExpressionString]) + 1
        leftEdges.append([leftExpressionString, expressionString])
        leftEdges.append([rightExpressionString, expressionString])

        return leftEdges, leftPositions, leftSubexpressions


def visualize_knowledge(expressionsDict={},
                        factsDict={},
                        categoricalConstraints={},
                        evidenceDict={}):
    edges = []
    horPositions = {}
    nodes = []

    allExpressions = [expressionsDict[key][0] for key in expressionsDict]
    allExpressions.extend(list(factsDict.values()))


    for expression in allExpressions:
        expressionEdges, expressionPositions, subexpressions = get_edges_and_positions(expression)
        edges.extend(expressionEdges)
        horPositions = {
            **horPositions,
            **expressionPositions}
        nodes.extend(subexpressions)

    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes)

    pos = nx.spring_layout(graph, k=0.8)
    for nodeKey in pos:
        pos[nodeKey][0] = horPositions[nodeKey]

    atoms = eu.get_all_variables(allExpressions)

    trueColor = "blue"
    falseColor = "red"
    neutralColor = "gray"

    hidden_node_alpha = 0.3
    visible_node_alpha = 0.8

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if
                                     evidenceDict[atomKey] and atomKey not in atoms],
                           node_size=1000,
                           node_color=trueColor,
                           alpha=hidden_node_alpha)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if
                                     not evidenceDict[atomKey] and atomKey not in atoms],
                           node_size=1000,
                           node_color=falseColor,
                           alpha=hidden_node_alpha)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in nodes if
                                     atomKey not in evidenceDict and atomKey not in atoms],
                           node_size=1000,
                           node_color=neutralColor,
                           alpha=hidden_node_alpha)

    ## Visible
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if evidenceDict[atomKey] and atomKey in atoms],
                           node_size=1000,
                           node_color=trueColor,
                           alpha=visible_node_alpha)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if
                                     not evidenceDict[atomKey] and atomKey in atoms],
                           node_size=1000,
                           node_color=falseColor,
                           alpha=visible_node_alpha)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in atoms if atomKey not in evidenceDict and atomKey in atoms],
                           node_size=1000,
                           node_color=neutralColor,
                           alpha=visible_node_alpha)

    # nx.draw_networkx_nodes(graph, pos, nodes)

    nx.draw_networkx_labels(graph, pos, {nodeKey: nodeKey for nodeKey in nodes})
    nx.draw_networkx_edges(graph, pos, edges)

    plt.show()


if __name__ == "__main__":
    visualize_knowledge(expressionsDict={"a": [["not", ["b", "and", "c"]], 2]},
                        factsDict={"f1" : ["not", "c"],
                                   "f3" : ["b", "and", ["not","c"]]},
                        evidenceDict={"(b_and_c)": 1,
                                      "c": 0})

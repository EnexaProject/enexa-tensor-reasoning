import networkx as nx
from matplotlib import pyplot as plt

from tnreason import encoding


def visualize(hybridKB, title="Visualization of the Knowledge Base",
              savePath=None):
    expressionsDict = hybridKB.weightedFormulas
    factsDict = hybridKB.facts
    categoricalConstraints = hybridKB.categoricalConstraints
    evidenceDict = hybridKB.evidence

    graph, pos, atoms = visualize_subexpressions(expressionsDict=expressionsDict, factsDict=factsDict,
                                                 categoricalConstraints=categoricalConstraints,
                                                 evidenceDict=evidenceDict)
    draw_with_evidence(graph, pos, evidenceDict, atoms, title=title, savePath=savePath)


def visualize_with_differing_evidence(expressionsDict={},
                                      factsDict={},
                                      categoricalConstraints={},
                                      evidenceList=[],
                                      title="Visualization of the Knowledge Base",
                                      saveBasePath="../../tests/visualization"):
    graph, pos, atoms = visualize_subexpressions(expressionsDict=expressionsDict, factsDict=factsDict,
                                                 categoricalConstraints=categoricalConstraints,
                                                 evidenceDict={})
    for i, evidenceDict in enumerate(evidenceList):
        draw_with_evidence(graph, pos, evidenceDict, atoms, title=title, savePath=saveBasePath + str(i) + ".png")


def get_edges_and_positions(expression):
    expressionString = encoding.get_formula_color(expression)

    if isinstance(expression, str):
        return [], {expressionString: 0}, [expressionString]
    else:
        edges, positions, subExpressions = [], {}, [expressionString]
        for subExpression in expression:
            if is_subexpression(subExpression):
                subEdges, subPositions, subSubExpressions = get_edges_and_positions(subExpression)
                edges = edges + subEdges
                positions = {**positions, **subPositions}
                subExpressions = subExpressions + subSubExpressions

                edges.append([encoding.get_formula_color(expression), encoding.get_formula_color(subExpression)])

        positions[encoding.get_formula_color(expression)] = 1 + max(
            [positions[encoding.get_formula_color(subExpression)] for subExpression in expression if
             is_subexpression(subExpression)])

        return edges, positions, subExpressions


def visualize_subexpressions(expressionsDict={},
                             factsDict={},
                             categoricalConstraints={},
                             evidenceDict={}):
    edges = []
    horPositions = {}
    nodes = []

    allExpressions = list(expressionsDict.values()) + list(factsDict.values()) + [[key] for key in evidenceDict]
    for key in categoricalConstraints:
        allExpressions = allExpressions + categoricalConstraints[key]

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

    atoms = encoding.get_all_atoms({**{key: expressionsDict[key][:-1] for key in expressionsDict}, **factsDict})
    return graph, pos, atoms


def draw_with_evidence(graph, pos, evidenceDict, atoms, title, savePath=None):
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
                           nodelist=[atomKey for atomKey in graph.nodes if
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

    nx.draw_networkx_labels(graph, pos, {nodeKey: get_symbol(nodeKey) for nodeKey in graph.nodes})
    edges = [edge for edge in graph.edges if edge[0] != edge[1]]
    nx.draw_networkx_edges(graph, pos, edges)

    plt.title(title, fontsize=15)
    if savePath is not None:
        plt.savefig(savePath)
    plt.show()

def is_subexpression(expression):
    if type(expression) == list:
        return True
    elif type(expression) == str:
        return not expression in ["id", "not", "and", "or", "xor", "imp", "eq"]
    else:
        return False


def get_symbol(expressionString):
    connectiveDict = {
        "and": "and",
        "not": "not"
    }
    if not "_" in expressionString:
        return expressionString
    else:
        connective = expressionString.split("_")[0][1:]
        if connective in connectiveDict:
            return connectiveDict[connective]
        return connective


if __name__ == "__main__":
    visualize_with_differing_evidence(
        expressionsDict={"e1": ["not", ["and", "b", "c"], 2]},
        factsDict={"f1": ["c"]},
        categoricalConstraints={"c1": ["e", "c"]},
        evidenceList=[{"c": 0}, {"c": 1, "e": 0}])

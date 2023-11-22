import networkx as nx
import matplotlib.pyplot as plt

def generate_expression_graph(expression):
    if type(expression) == str:
        atomNodes = [expression]
        expressionNodes = []
        edgesAnd = []
        edgesNot = []
    elif expression[0] == "not":
        atomNodes, expressionNodes, edgesAnd, edgesNot = generate_expression_graph(expression[1])
        edgesNot.append([expression[1], expression])
    elif expression[1] == "and":
        atomNodes0, expressionNodes0, edgesAnd0, edgesNot0 = generate_expression_graph(expression[0])
        atomNodes2, expressionNodes2, edgesAnd2, edgesNot2 = generate_expression_graph(expression[2])
        atomNodes = atomNodes0 + atomNodes2
        expressionNodes = expressionNodes0 + expressionNodes2
        edgesAnd = edgesAnd0 + edgesAnd2
        edgesNot = edgesNot0 + edgesNot2
        edgesAnd.append([expression[0], expression])
        edgesAnd.append([expression[2], expression])

    expressionNodes.append(expression)
    return atomNodes, expressionNodes, edgesAnd, edgesNot

def visualize_expression_graph(expression,fontsize=16):
    atomNodes, expressionNodes, edgesAnd, edgesNot = generate_expression_graph(expression)

    graph = nx.DiGraph()
    graph.add_nodes_from([str(node) for node in expressionNodes])
    graph.add_edges_from([[str(edge[0]), str(edge[1])] for edge in edgesAnd + edgesNot])

    pos = nx.spring_layout(graph)

    labels = { atom : atom for atom in atomNodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=fontsize)

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=atomNodes,
                           node_color='b',
                           node_size=3000,
                           alpha=0.2)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[str(node) for node in expressionNodes if node not in atomNodes],
                           node_color='b',
                           node_size=1000,
                           alpha=0.1)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[str(expression)],
                           node_color="black",
                           node_size=1000,
                           alpha=0.4)

    nx.draw_networkx_edges(graph, pos,
                           edgelist=[[str(edge[0]), str(edge[1])] for edge in edgesAnd],
                           arrowstyle="-|>",
                           width=2, alpha=1, edge_color='b')
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[[str(edge[0]), str(edge[1])] for edge in edgesNot],
                           width=2, alpha=1, edge_color='r')
    plt.show()


if __name__ == "__main__":
   visualize_expression_graph(["R2(x,z)", "and", ["C1(x)", "and", ["not", "R1(y,x)"]]],fontsize=10)


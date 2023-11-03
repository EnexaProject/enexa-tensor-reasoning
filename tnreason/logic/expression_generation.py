def generate_list_from_rule(premises, head):
    expression = ["not", premises[0]]
    for premise in premises[1:]:
        expression = [["not", premise], "and", expression]
    expression = ["not", [head, "and", expression]]
    return expression


def generate_conjunctions(atoms):
    expression = atoms[0]
    for atom in atoms[1:]:
        expression = [atom, "and", expression]
    return expression


def generate_negated_conjunctions(positive_atoms, negated_atoms):
    return generate_conjunctions(positive_atoms + [["not", atom] for atom in negated_atoms])


def replace_atoms(expression, atomDict):
    if type(expression) == str:
        return atomDict[expression]
    else:
        if expression[0] == "not":
            return ["not", replace_atoms(expression[1], atomDict)]
        elif expression[1] == "and":
            return [replace_atoms(expression[0], atomDict), "and", replace_atoms(expression[2], atomDict)]


def generate_graph(expression):
    if type(expression) == str:
        nodes = []
        edgesAnd = []
        edgesNot = []
    elif expression[0] == "not":
        nodes, headNode, edgesAnd, edgesNot = generate_graph(expression[1])
        edgesNot.append([headNode, expression])
    elif expression[1] == "and":
        nodes0, headNode0, edgesAnd0, edgesNot0 = generate_graph(expression[0])
        nodes2, headNode2, edgesAnd2, edgesNot2 = generate_graph(expression[2])
        nodes = nodes0 + nodes2
        edgesAnd = edgesAnd0 + edgesAnd2
        edgesNot = edgesNot0 + edgesNot2
        edgesAnd.append([headNode0, expression])
        edgesAnd.append([headNode2, expression])

    nodes.append(expression)
    headNode = expression
    return nodes, headNode, edgesAnd, edgesNot


def generate_pracmln_string(expression, weight):
    return str(weight) + " " + generate_pracmln_formulastring(expression)

def generate_pracmln_formulastring(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        return "!(" + generate_pracmln_string(expression[1]) + ")"
        # raise TypeError("pracmln string model does not yet support {}.".format(expression))
    elif expression[1] == "and":
        return generate_pracmln_string(expression[0]) + " ^ " + generate_pracmln_string(expression[2])


if __name__ == "__main__":
    nodes, headNode, edgesAnd, edgesNot = generate_graph(["R2(x,z)", "and", ["C1(x)", "and", ["not", "R1(y,x)"]]])

    import networkx as nx
    import matplotlib.pyplot as plt

    graph = nx.DiGraph()

    graph.add_nodes_from([str(node) for node in nodes])
    graph.add_edges_from([[str(edge[0]), str(edge[1])] for edge in edgesAnd])
    graph.add_edges_from([[str(edge[0]), str(edge[1])] for edge in edgesNot])

    nx.draw(graph)
    plt.show()

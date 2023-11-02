from tensor_reasoning.logic import expression_calculus as ec, basis_calculus as bc

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor

import numpy as np

def create_markov_logic_network(expressions_dict):
    mn = MarkovNetwork()
    factors = []
    edges = []
    for exKey in expressions_dict:
        expression = expressions_dict[exKey][0]
        weight = expressions_dict[exKey][1]
        ex_variables = np.unique(ec.get_variables(expression))

        mn.add_nodes_from(ex_variables)

        for startnode in ex_variables:
            for endnode in ex_variables:
                if (startnode,endnode) not in edges and (endnode,startnode) not in edges:
                    if startnode!=endnode:
                        edges.append((startnode,endnode))

        core = calculate_dangling_basis(expression).calculate_truth()
        variables = core.colors

        factors.append(DiscreteFactor(variables, [2 for node in variables], np.exp(weight * core.values)))

    mn.add_edges_from(edges)
    mn.add_factors(*factors)

    return mn

def calculate_dangling_basis(expression):

    variables = np.unique(ec.get_variables(expression))

    atom_dict = {}
    for variable in variables:
        atom_dict[variable] = bc.BasisCore(np.eye(2),[variable,"head"],headcolor="head",name = variable)

    return ec.calculate_core(atom_dict,expression)

if __name__ == "__main__":

    example_expression_dict = {
        "e1": [["not", "Ausgangsrechnung(x)"], 12],
        "e2": [[["not", "Ausgangsrechnung(x)"], "and", ["not", "Rechnung(x)"]], 14]
    }

    mn = create_markov_logic_network(example_expression_dict)
    print(mn.get_partition_function())
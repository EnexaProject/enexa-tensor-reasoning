from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling

# from tnreason.logic import expression_calculus as ec
from tnreason.logic import expression_utils as eu

from tnreason.contraction import expression_evaluation as ee

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class PgmpyMLN:
    def __init__(self, expressionsDict=None):
        self.model = MarkovNetwork()

        if expressionsDict is not None:
            self.extend_from_expressionsDict(expressionsDict)

    def extend_from_expressionsDict(self, expressionsDict):
        factors = []
        edges = []
        for exKey in expressionsDict:
            expression = expressionsDict[exKey][0]
            weight = expressionsDict[exKey][1]
            ex_variables = np.unique(eu.get_variables(expression))

            self.model.add_nodes_from(ex_variables)

            for startnode in ex_variables:
                for endnode in ex_variables:
                    if (startnode, endnode) not in edges and (endnode, startnode) not in edges:
                        if startnode != endnode:
                            edges.append((startnode, endnode))

            core = ee.ExpressionEvaluator(expression, initializeBasisCores=True).create_formula_factor()
            variables = core.colors

            factors.append(DiscreteFactor(variables, [2 for node in variables], np.exp(weight * core.values)))
        self.model.add_edges_from(edges)
        self.model.add_factors(*factors)

    def map_query_given_evidenceDict(self, evidenceDict, variables):
        inference_algorithm = VariableElimination(self.model)
        query_result = inference_algorithm.map_query(evidence=evidenceDict, variables=variables)
        return query_result

    def cond_query_given_evidenceDict(self, evidenceDict, variables):
        inference_algorithm = VariableElimination(self.model)
        query_result = inference_algorithm.query(evidence=evidenceDict, variables=variables)
        return query_result

    def generate_sampleDf(self, sampleNum=1, chainSize=10, method="Gibbs"):
        if method == "Gibbs":
            sampler = GibbsSampling(self.model)
        else:
            return ValueError("Method {} not supported!".format(method))

        df = pd.DataFrame(columns=self.model.nodes)
        for ind in range(sampleNum):
            row_df = pd.DataFrame(sampler.sample(size=chainSize).iloc[-1].to_dict(), index=[ind])
            df = pd.concat([df, row_df])
        return df.astype("int64")

    def visualize(self, regenerate_graph=True, truthDict={}, fontsize=10):
        if regenerate_graph:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(self.model.nodes)
            self.graph.add_edges_from(self.model.edges)
            self.graph_pos = nx.spring_layout(self.graph)
        labels = {atom: atom for atom in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, self.graph_pos, labels, font_size=fontsize)

        colorList = ["r", "g", "b", "r", "g", "b", "r", "g", "b"]
        for i, factor in enumerate(self.model.get_factors()):
            variables = factor.variables
            color = colorList[i]
            edges = [(var1, var2) for var1 in variables for var2 in variables]
            nx.draw_networkx_edges(self.graph, self.graph_pos,
                                   edgelist=edges, width=10, alpha=0.2, edge_color=color)

        trueNodes = [atom for atom in truthDict.keys() if truthDict[atom] == True]
        falseNodes = [atom for atom in truthDict.keys() if truthDict[atom] == False]
        otherNodes = {atom: atom for atom in self.graph.nodes if atom not in truthDict.keys()}

        nx.draw_networkx_nodes(self.graph, self.graph_pos,
                               nodelist=otherNodes,
                               node_color="grey",
                               node_size=3000,
                               alpha=0.2)
        nx.draw_networkx_nodes(self.graph, self.graph_pos,
                               nodelist=trueNodes,
                               node_color="b",
                               node_size=3000,
                               alpha=0.6)
        nx.draw_networkx_nodes(self.graph, self.graph_pos,
                               nodelist=falseNodes,
                               node_color="r",
                               node_size=3000,
                               alpha=0.6)
        plt.show()


if __name__ == "__main__":
    example_expression_dict = {
        "e0": [["not", ["Unterschrank(z)", "and", ["not", "Moebel(z)"]]], 20],
        "e0.5": ["Moebel(z)", 4],
        "e0.625": ["Sledz", 4],
        "e0.626": [["not", ["Moebel(z)", "and", "Sledz"]], 5],
        "e0.75": [["not", [["Unterschrank(z)", "and", "Sledz"], "and", ["not", "Ausgangsrechnung(x)"]]], 2],
        "e1": [["not", "Ausgangsrechnung(x)"], 12],
        "e2": [[["not", "Ausgangsrechnung(x)"], "and", ["not", "Rechnung(x)"]], 14]
    }
    ## 1 = True, 0 = False
    example_evidence_dict = {
        "Unterschrank(z)": 1,
        "Ausgangsrechnung(x)": 1
    }

    test_mln = PgmpyMLN(example_expression_dict)
    cond_query_result = test_mln.cond_query_given_evidenceDict(example_evidence_dict, ["Moebel(z)"])
    cond_query_result.normalize()

    test_mln.visualize(truthDict={"Unterschrank(z)": True, "Ausgangsrechnung(x)": False})

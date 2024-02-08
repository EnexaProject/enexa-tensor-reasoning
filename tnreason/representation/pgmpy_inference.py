from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


def from_hybridKB(hybridKB):
    return PgmpyInferer({
        **hybridKB.formulaTensors.all_cores(),
        **hybridKB.facts.all_cores()
    })


class PgmpyInferer:
    def __init__(self, coresDict={}):
        self.model = MarkovNetwork()
        self.add_factors_from_coresDict(coresDict)

    def add_factors_from_coresDict(self, coresDict):
        for coreKey in coresDict:
            for col1 in coresDict[coreKey].colors:
                self.model.add_node(col1)
                for col2 in coresDict[coreKey].colors:
                    if col2 != col1:
                        self.model.add_edge(col1, col2)
            self.model.add_factors(
                DiscreteFactor(coresDict[coreKey].colors, coresDict[coreKey].values.shape, coresDict[coreKey].values)
            )

    def map_query(self, variables, evidenceDict={}):
        inference_algorithm = VariableElimination(self.model)
        query_result = inference_algorithm.map_query(evidence=evidenceDict, variables=variables)
        return query_result

    def query(self, variables, evidenceDict={}):
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

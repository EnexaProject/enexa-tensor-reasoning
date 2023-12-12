from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling

from tnreason.logic import basis_calculus as bc
from tnreason.logic import expression_calculus as ec
from tnreason.logic import expression_generation as eg
from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import core_contractor as ccon

from tnreason.model import infer_mln as imln

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class TensorMLN:
    def __init__(self, expressionsDict, formulaCoreDict=None):
        self.expressionsDict = expressionsDict
        self.atomKeys = np.unique(
            ec.get_all_variables([self.expressionsDict[formulaKey][0] for formulaKey in self.expressionsDict]))
        self.formulaCoreDict = formulaCoreDict

    def infer_on_evidenceDict(self, evidenceDict={}):
        inferedExpressionsDict = {}
        for key in self.expressionsDict:
            inferedFormula = imln.infer_expression(self.expressionsDict[key][0], evidenceDict)
            if inferedFormula not in ["Thing", "Nothing"]:
                inferedFormula = eg.remove_double_not(inferedFormula)
                inferedExpressionsDict[key] = [inferedFormula, self.expressionsDict[key][1]]
        return TensorMLN(inferedExpressionsDict)

    def reduce_double_formulas(self):
        checkedKeys = []
        reducedExpressionDict = {}
        for key in self.expressionsDict:
            if key not in checkedKeys:
                checkedKeys.append(key)
                keyFormula, keyWeight = self.expressionsDict[key]
                for otherKey in self.expressionsDict:
                    if otherKey not in checkedKeys and eg.equality_check(keyFormula, self.expressionsDict[otherKey][0]):
                        checkedKeys.append(otherKey)
                        keyWeight = keyWeight + self.expressionsDict[otherKey][1]
                reducedExpressionDict[key] = [keyFormula, keyWeight]
        self.expressionsDict = reducedExpressionDict

    def initialize_formulaCoreDict(self):
        self.formulaCoreDict = create_formulaCoreDict(self.expressionsDict)

    def compute_marginalized(self, marginalKeys, optimizationMethod="GreedyHeuristic"):
        if self.formulaCoreDict is None:
            self.initialize_formulaCoreDict()
        contractionDict = self.formulaCoreDict.copy()
        for atomKey in self.atomKeys:
            if atomKey not in marginalKeys:
                contractionDict[atomKey] = cc.CoordinateCore(np.ones(2), [atomKey], atomKey)
        contractor = ccon.CoreContractor(contractionDict, openColors=marginalKeys)
        if optimizationMethod == "GreedyHeuristic":
            contractor.optimize_coreList()  ## Using Greedy, Alternative
        else:
            raise ValueError("Optimization Method {} not supported!".format(optimizationMethod))
        return contractor.contract().normalize()

    ## To be implemented: Here we need Tensor Network contractions
    def create_independent_atom_sample(self, atomSampleKey):
        marginalProbCore = self.compute_marginalized([atomSampleKey])
        return np.random.multinomial(1, marginalProbCore.values)[0] == 0

    def create_independent_sample(self):
        return {atomKey: self.create_independent_atom_sample(atomKey) for atomKey in self.atomKeys}

    def gibbs(self, repetitionNum=10, verbose=True):
        sampleDict = self.create_independent_sample()
        for repetitionPos in range(repetitionNum):
            if verbose:
                print("## Gibbs Iteration {} ##".format(repetitionPos))
            for refinementAtomKey in sampleDict:
                samplerMLN = self.infer_on_evidenceDict(
                    {key: sampleDict[key] for key in sampleDict if key != refinementAtomKey})
                samplerMLN.reduce_double_formulas()
                if refinementAtomKey not in samplerMLN.atomKeys:
                    print("Warning: Infered MLN is empty on key {}".format(refinementAtomKey))
                    samplerMLN = TensorMLN({refinementAtomKey: [str(refinementAtomKey), 0]})
                sampleDict[refinementAtomKey] = samplerMLN.create_independent_atom_sample(refinementAtomKey)
            if verbose:
                print("SampleDict is {}".format(sampleDict))
        return sampleDict

    def generate_sampleDf(self, sampleNum, method="Gibbs10"):
        df = pd.DataFrame(columns=self.atomKeys)
        for ind in range(sampleNum):
            if method.startswith("Gibbs"):
                repetitionNum = int(method.split("Gibbs")[1])
                row_df = pd.DataFrame(self.gibbs(repetitionNum=repetitionNum, verbose=False), index=[ind])
                df = pd.concat([df, row_df])
        return df.astype("int64")


class MarkovLogicNetwork:
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
            ex_variables = np.unique(ec.get_variables(expression))

            self.model.add_nodes_from(ex_variables)

            for startnode in ex_variables:
                for endnode in ex_variables:
                    if (startnode, endnode) not in edges and (endnode, startnode) not in edges:
                        if startnode != endnode:
                            edges.append((startnode, endnode))

            core = calculate_dangling_basis(expression)
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


def calculate_dangling_basis(expression):
    variables = np.unique(ec.get_variables(expression))
    atom_dict = {}
    for variable in variables:
        atom_dict[variable] = bc.BasisCore(np.eye(2), [variable, "head"], headcolor="head", name=variable)
    return ec.calculate_core(atom_dict, expression).calculate_truth().reduce_identical_colors()


def create_formulaCoreDict(expressionsDict):
    return {formulaKey: calculate_dangling_basis(
        expressionsDict[formulaKey][0]).to_coordinate().weighted_exponentiation(
        expressionsDict[formulaKey][1])
        for formulaKey in expressionsDict}


if __name__ == "__main__":
    example_expression_dict = {
        "e0": [["not", ["Unterschrank(z)", "and", ["not", "Moebel(z)"]]], 20],
        "e0.5": ["Moebel(z)", 10],
        "e0.625": ["Sledz", 10],
        "e0.626": [["not", ["Moebel(z)", "and", "Sledz"]], 4],
        "e0.75": [["not", [["Unterschrank(z)", "and", "Sledz"], "and", ["not", "Ausgangsrechnung(x)"]]], 2],
        "e1": [["not", "Ausgangsrechnung(x)"], 12],
        "e2": [[["not", "Ausgangsrechnung(x)"], "and", ["not", "Rechnung(x)"]], 14]
    }
    ## 1 = True, 0 = False
    example_evidence_dict = {
        "Unterschrank(z)": 1,
        "Ausgangsrechnung(x)": 1
    }

    # print(create_formulaCoreDict(example_expression_dict))

    tn_mln = TensorMLN(example_expression_dict)
    # tn_mln.independent_atom_sample("Unterschrank(z)")
    # print(tn_mln.gibbs(10))
    print(tn_mln.generate_sampleDf(int(1e4)))
    exit()

    infered_mln = tn_mln.infer_on_evidenceDict(example_evidence_dict)
    #    print(infered_mln.expressionsDict)
    #    infered_mln.reduce_double_formulas()
    #    print(infered_mln.expressionsDict)

    test_mln = MarkovLogicNetwork(example_expression_dict)
    cond_query_result = test_mln.cond_query_given_evidenceDict(example_evidence_dict, ["Moebel(z)"])
    cond_query_result.normalize()
    # print(cond_query_result.values)

    # test_mln.visualize(truthDict={"Unterschrank(z)": True, "Ausgangsrechnung(x)": False})

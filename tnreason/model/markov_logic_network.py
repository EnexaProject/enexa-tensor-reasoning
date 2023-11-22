from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling

from tnreason.logic import basis_calculus as bc
from tnreason.logic import expression_calculus as ec

import numpy as np
import pandas as pd

class MarkovLogicNetwork:
    def __init__(self, expressionsDict = None):
        self.model = MarkovNetwork()

        if expressionsDict is not None:
            self.extend_from_expressionsDict(expressionsDict)

    def extend_from_expressionsDict(self,expressionsDict):
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

            core = calculate_dangling_basis(expression).calculate_truth()
            variables = core.colors

            factors.append(DiscreteFactor(variables, [2 for node in variables], np.exp(weight * core.values)))
        self.model.add_edges_from(edges)
        self.model.add_factors(*factors)

    def map_query_given_evidenceDict(self,evidenceDict,variables):
        inference_algorithm = VariableElimination(self.model)
        query_result = inference_algorithm.map_query(evidence = evidenceDict, variables = variables)
        return query_result

    def cond_query_given_evidenceDict(self,evidenceDict,variables):
        inference_algorithm = VariableElimination(self.model)
        query_result = inference_algorithm.query(evidence = evidenceDict, variables = variables)
        return query_result

    def generate_sampleDf(self, sampleNum=1, chainSize=10, method="Gibbs"):
        if method == "Gibbs":
            sampler = GibbsSampling(self.model)
        else:
            return ValueError("Method {} not supported!".format(method))

        df = pd.DataFrame(columns = self.model.nodes)
        for ind in range(sampleNum):
            row_df = pd.DataFrame(sampler.sample(size=chainSize).iloc[-1].to_dict(), index = [ind])
            df = pd.concat([df, row_df])
        return df.astype("bool")

def calculate_dangling_basis(expression):
    variables = np.unique(ec.get_variables(expression))
    atom_dict = {}
    for variable in variables:
        atom_dict[variable] = bc.BasisCore(np.eye(2),[variable,"head"],headcolor="head",name = variable)
    return ec.calculate_core(atom_dict,expression)

if __name__ == "__main__":
    test_mln = MarkovLogicNetwork()

    example_expression_dict = {
        "e0": [["not",["Unterschrank(z)","and",["not","Moebel(z)"]]], 20],
        "e0.5": ["Moebel(z)", -2],
        "e1": [["not", "Ausgangsrechnung(x)"], 12],
        "e2": [[["not", "Ausgangsrechnung(x)"], "and", ["not", "Rechnung(x)"]], 14]
    }
    test_mln.create_from_expressionsDict(example_expression_dict)

    ## 1 = True, 0 = False
    example_evidence_dict = {
        "Unterschrank(z)": 1,
        "Ausgangsrechnung(x)": 1
    }
    print(test_mln.map_query_given_evidenceDict(example_evidence_dict,["Moebel(z)"])["Moebel(z)"])
    cond_query_result = test_mln.cond_query_given_evidenceDict(example_evidence_dict,["Moebel(z)"])
    cond_query_result.normalize()
    print(cond_query_result.values)
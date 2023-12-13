import numpy as np
# import tnreason.logic.coordinate_calculus as cc
# import tnreason.logic.basis_calculus as bc

## SHIFTED TO THESE MODULES
import tnreason.logic.expression_utils as eu
import tnreason.contraction.expression_evaluation as ee


## Replaced by tnreason.contraction.expression_evaluation -> ExpressionEvaluator(expression, atomDict).evaluate()
def calculate_core(atom_dict, expression):
    return ee.ExpressionEvaluator(expression, atomDict=atom_dict).evaluate()
    # if type(expression) == str:
    #     return atom_dict[expression]
    # elif expression[0] == "not":
    #     return calculate_core(atom_dict, expression[1]).negate()
    # elif expression[1] == "and":
    #     return calculate_core(atom_dict, expression[0]).compute_and(
    #         calculate_core(atom_dict, expression[2]))
    # else:
    #     raise ValueError("Expression {} not understood.".format(expression))


## To be replaced by tnreason.contraction.expression_evaluation -> ExpressionEvaluator(expression).evaluate_on_sampleDf()
def evaluate_expression_on_sampleDf(sampleDf, expression):
    return ee.ExpressionEvaluator(expression).evaluate_on_sampleDf(sampleDf)
    # variables = eu.get_variables(expression)
    # atomDict = {}
    # for variable in variables:
    #     if variable == "Thing":
    #         values = np.ones(sampleDf.shape[0])
    #     elif variable == "Nothing":
    #         values = np.zeros(sampleDf.shape[0])
    #     else:
    #         values = sampleDf[variable].astype("int64").values
    #     atomDict[variable] = cc.CoordinateCore(values, ["j"], variable)
    # return calculate_core(atomDict, expression)


###

def evaluate_expression_on_factDf(factDf, individualsDict, expression):
    variables = eu.get_variables(expression)
    atomDict = generate_atomDict(factDf, individualsDict, variables)
    return calculate_core(atomDict, expression)


def generate_atomDict(factDf, individualsDict, atoms):
    atomDict = {}
    for atomKey in atoms:
        if "," in atomKey:
            relationKey = atomKey.split("(")[0]
            indKey1, indKey2 = atomKey.split("(")[1][:-1].split(",")
            relValues = generate_relation_values(factDf, individualsDict[indKey1], individualsDict[indKey2],
                                                 relationKey)
            atomDict[atomKey] = cc.CoordinateCore(relValues, [indKey1, indKey2], atomKey)
        else:
            classKey = atomKey.split("(")[0]
            indKey = atomKey.split("(")[1][:-1]
            classValues = generate_class_values(factDf, individualsDict[indKey], classKey)
            atomDict[atomKey] = cc.CoordinateCore(classValues, [indKey], atomKey)
    return atomDict


def generate_class_values(factDf, individuals, classKey):
    relevantInds = factDf[factDf["predicate"].isin(["http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "rdf:type"])]
    relevantInds = relevantInds[relevantInds["object"] == classKey]
    outvalues = np.zeros((len(individuals)))
    for i, row in relevantInds.iterrows():
        subPos = np.argwhere(individuals == row["subject"])
        outvalues[subPos] = 1
    return outvalues


def generate_relation_values(factDf, individuals1, individuals2, relationKey):
    relevantPairs = factDf[factDf["predicate"] == relationKey]
    outValues = np.zeros((len(individuals1), len(individuals2)))
    for i, row in relevantPairs.iterrows():
        subPos = np.argwhere(individuals1 == row["subject"])
        obPos = np.argwhere(individuals2 == row["object"])
        outValues[subPos, obPos] = 1
    return outValues


def calculate_expressionCore(expression):
    variables = np.unique(eu.get_variables(expression))
    atom_dict = {}
    for variable in variables:
        atom_dict[variable] = bc.BasisCore(np.eye(2), [variable, "head"], headcolor="head", name=variable)
    return calculate_core(atom_dict, expression).calculate_truth().reduce_identical_colors().to_coordinate()


if __name__ == "__main__":
    from tnreason.logic import basis_calculus as bc, coordinate_calculus as cc

    core0_values = np.random.normal(size=(100, 10, 5))
    core0_colors = ["a", "b", "c"]
    core1_values = np.random.normal(size=(10, 100, 3))
    core1_colors = ["b", "a", "d"]

    core0 = cc.CoordinateCore(core0_values, core0_colors, "Sledz")
    core1 = cc.CoordinateCore(core1_values, core1_colors, "Jaszczur")
    core2 = cc.CoordinateCore(core1_values, core1_colors, "Sokol")

    atom_dict = {"a": core0,
                 "b": core1,
                 "c": core2}
    # example_expression = ["a","and",["not",["b","and","b"]]]
    example_expression = [[["not", "a"], "and", ["not", "b"]], "and", "b"]
    core = calculate_core(atom_dict, example_expression)
    print(core.name)
    print(core.values.shape)
    # example_expression = [["a", "and", "b"], "and", ["not", "c"]]

    core0 = bc.BasisCore(np.ones(2), ["Sledz"])
    core1 = bc.BasisCore(np.ones(2), ["Jaszczur"])
    core2 = bc.BasisCore(np.ones(2), ["Sokol"])

    atom_dict = {"a": core0,
                 "b": core1,
                 "c": core2}
    # example_expression = ["a","and",["not",["b","and","b"]]]
    example_expression = [[["not", "a"], "and", ["not", "b"]], "and", "b"]
    core = calculate_core(atom_dict, example_expression)
    print(core.name)
    print(core.values.shape)

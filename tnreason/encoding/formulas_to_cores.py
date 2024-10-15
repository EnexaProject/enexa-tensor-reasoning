from tnreason import engine

from tnreason.encoding import connectives as con

import math

connectiveFixCoreSuffix = "_conCore"
headCoreSuffix = "_headCore"


def create_formulas_cores(expressionsDict, alreadyCreated=[], coreType=None):
    """
    Creates a tensor network of connective and head cores
        * expressionsDict: Dictionary of nested listed representing expressions
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    knowledgeCores = {}
    for formulaName in expressionsDict.keys():
        if isinstance(expressionsDict[formulaName][-1], float) or isinstance(expressionsDict[formulaName][-1], int):
            knowledgeCores = {**knowledgeCores,
                              **create_head_core(get_formula_color(expressionsDict[formulaName][:-1]), "expFactor",
                                                 weight=
                                                 expressionsDict[formulaName][-1], coreType=coreType),
                              **create_raw_formula_cores(expressionsDict[formulaName][:-1],
                                                         alreadyCreated=
                                                         list(knowledgeCores.keys()) + alreadyCreated,
                                                         coreType=coreType)}
        else:
            knowledgeCores = {**knowledgeCores,
                              **create_head_core(get_formula_color(expressionsDict[formulaName]), "truthEvaluation",
                                                 coreType=coreType),
                              **create_raw_formula_cores(expressionsDict[formulaName],
                                                         alreadyCreated=list(knowledgeCores.keys()) + alreadyCreated,
                                                         coreType=coreType)}
    return knowledgeCores


def create_raw_formula_cores(expression, alreadyCreated=[], coreType=None):
    """
    Creates the connective cores to an expression, omitting the elsewhere created cores
        * expression: Nested list specifying a formula
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    if get_formula_color(expression) + connectiveFixCoreSuffix in alreadyCreated:
        return {}
    if isinstance(expression, str):
        return {}
    elif len(expression) == 1:
        assert isinstance(expression[0], str)
        return {}

    elif len(expression) == 2:
        return {**create_connective_core(expression, coreType=coreType),
                **create_raw_formula_cores(expression[1], alreadyCreated=alreadyCreated, coreType=coreType)}
    elif len(expression) == 3:
        return {**create_connective_core(expression, coreType=coreType),
                **create_raw_formula_cores(expression[1], alreadyCreated=alreadyCreated, coreType=coreType),
                **create_raw_formula_cores(expression[2], alreadyCreated=alreadyCreated, coreType=coreType)
                }
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_connective_core(expression, coreType=None):
    """
    Creates the connective core at the head of the expression by loading the truth table
    """
    expressionString = get_formula_color(expression)
    if isinstance(expression, str):
        return {}

    elif len(expression) == 2:
        preExpressionString = get_formula_color(expression[1])
        return {expressionString + connectiveFixCoreSuffix:
                    engine.create_relational_encoding(inshape=[2], outshape=[2], incolors=[preExpressionString],
                                                      outcolors=[expressionString],
                                                      function=con.get_connectives(expression[0]),
                                                      coreType=coreType,
                                                      name=expressionString + connectiveFixCoreSuffix)}

    elif len(expression) == 3:
        leftExpressionString = get_formula_color(expression[1])
        rightExpressionString = get_formula_color(expression[2])
        return {expressionString + connectiveFixCoreSuffix:
                    engine.create_relational_encoding(inshape=[2, 2], outshape=[2],
                                                      incolors=[leftExpressionString, rightExpressionString],
                                                      outcolors=[expressionString],
                                                      function=con.get_connectives(expression[0]),
                                                      coreType=coreType,
                                                      name=expressionString + connectiveFixCoreSuffix)}
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_head_core(expression, headType, weight=None, name=None, coreType=None):
    """
    Created the head core to an expression activating it
    """
    if headType == "truthEvaluation":
        headFunction = lambda x: x
    elif headType == "falseEvaluation":
        headFunction = lambda x: 1 - x
    elif headType == "expFactor":
        headFunction = lambda x: math.exp(weight * x)
    # elif headType == "weightedTruthEvaluation":
    #    headFunction = lambda x: weight * x
    # elif headType == "diffExpFactor":
    #   function = lambda x: x*math.exp(weight*x)
    else:
        raise ValueError("Headtype {} not understood!".format(headType))

    color = get_formula_color(expression)
    if name is None:
        name = color + headCoreSuffix
    return {name: engine.create_tensor_encoding([2], [color], headFunction, coreType=coreType, name=name)}


def create_evidence_cores(evidenceDict, coreType=None):
    """
    Turns positive and negative evidence into literal formulas and encodes them
    """
    return create_formulas_cores({**{key: [key] for key in evidenceDict if evidenceDict[key]},
                                  **{key: ["not", key] for key in evidenceDict if not evidenceDict[key]}
                                  }, coreType=coreType)


def get_formula_color(expression):
    """
    Identifies a color with an expression
    """
    if isinstance(expression, str):  ## Expression is atomic
        return expression
    elif len(expression) == 1:  ## Expression is atomic, but provided in nested form
        assert isinstance(expression[0], str)
        return expression[0]
    else:
        if not isinstance(expression[0], str):
            raise ValueError("Connective {} has wrong type!".format(expression[0]))
        return "(" + expression[0] + "_" + "_".join(
            [get_formula_color(entry) for entry in expression[1:]]) + ")"


def get_all_atoms(expressionsDict):
    """
    Identifies the leafs of the expressions in the expressionsDict as atoms
    """
    atoms = set()
    for key in expressionsDict:
        atoms = atoms | get_atoms(expressionsDict[key])
    return list(atoms)


def get_atoms(expression):
    if isinstance(expression, str):  ## Then an atom
        return {expression}
    elif len(expression) == 1:  ## Then an atomic formula
        return {expression[0]}
    else:  ## Then a formula with connective in first position
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_atoms(subExpression)
        return atoms

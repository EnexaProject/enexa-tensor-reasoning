from tnreason.knowledge import knowledge_visualization as knv

class LogicRepresentation:
    def __init__(self, expressionsDict, factsDict={}):
        self.expressionsDict = expressionsDict
        self.factsDict = factsDict

    def evaluate_evidence(self, evidenceDict):
        entailedExpressions = []
        contradictedExpressions = []
        contingentExpressions = []
        for key in self.expressionsDict:
            evidenceReplaced = infer_expression(self.expressionsDict[key][0], evidenceDict)
            if evidenceReplaced == "Thing":
                entailedExpressions.append(key)
            elif evidenceReplaced == "Nothing":
                contradictedExpressions.append(key)
            else:
                contingentExpressions.append(key)
        for key in self.factsDict:
            evidenceReplaced = infer_expression(self.factsDict[key], evidenceDict)
            if evidenceReplaced == "Thing":
                entailedExpressions.append(key)
            elif evidenceReplaced == "Nothing":
                contradictedExpressions.append(key)
            else:
                contingentExpressions.append(key)
        return entailedExpressions, contradictedExpressions, contingentExpressions

    def infer(self, evidenceDict, simplify=True):
        self.expressionsDict = {
            key: [replace_evidence_variables(self.expressionsDict[key][0], evidenceDict), self.expressionsDict[key][1]]
            for key in
            self.expressionsDict
        }
        self.factsDict = {
            key: replace_evidence_variables(self.factsDict[key], evidenceDict) for key in self.factsDict
        }
        if simplify:
            self.simplify()

    def forward_chaining(self, hardLogicLimit=100):
        ## Synthetic algorithm: Formulas with weights crossing the hardLogicLimit are taking for hard logical inference
        newEvidenceDict = find_new_evidence(self.expressionsDict, hardLogicLimit)
        evidenceDict = newEvidenceDict.copy()
        while not len(newEvidenceDict) == 0:
            self.infer({atomKey: bool(newEvidenceDict[atomKey] / abs(newEvidenceDict[atomKey])) for atomKey in
                        newEvidenceDict})
            newEvidenceDict = find_new_evidence(self.expressionsDict, hardLogicLimit)
            evidenceDict = {**evidenceDict, **newEvidenceDict}
        self.expressionsDict = {**self.expressionsDict,
                                **{atomKey + "_evidence": [atomKey, evidenceDict[atomKey]] for atomKey in evidenceDict}}

    def simplify(self):
        self.remove_thing_nothing()
        self.remove_double_nots()
        self.remove_doubles()  # Not working on constraints, this can be done using entailment checks
        self.beautify_weights()

    def remove_thing_nothing(self):
        newExpressionsDict = {}
        for key in self.expressionsDict:
            newExpression = reduce_thing_nothing(self.expressionsDict[key][0])
            if newExpression not in ["Thing", "Nothing"]:
                newExpressionsDict[key] = [newExpression, self.expressionsDict[key][1]]
        self.expressionsDict = newExpressionsDict
        newFactsDict = {}
        for key in self.factsDict:
            newExpression = reduce_thing_nothing(self.factsDict[key])
            if newExpression not in ["Thing", "Nothing"]:
                newFactsDict[key] = newExpression
        self.factsDict = newFactsDict

    def remove_double_nots(self):
        self.expressionsDict = {key: [reduce_double_not(self.expressionsDict[key][0]), self.expressionsDict[key][1]]
                                for
                                key in self.expressionsDict}
        self.factsDict = {key: reduce_double_not(self.factsDict[key]) for
                          key in self.factsDict}

    def remove_doubles(self):
        # Removes Expressions which are the same or negations of each other
        checkedKeys = []
        reducedExpressionDict = {}
        for key in self.expressionsDict:
            if key not in checkedKeys and self.expressionsDict[key][0]:
                checkedKeys.append(key)
                keyFormula, keyWeight = self.expressionsDict[key]
                for otherKey in self.expressionsDict:
                    if otherKey not in checkedKeys:
                        result = equality_contradiction_check(keyFormula, self.expressionsDict[otherKey][0])
                        if result == "equal":
                            checkedKeys.append(otherKey)
                            keyWeight += self.expressionsDict[otherKey][1]
                        elif result == "negequal":
                            checkedKeys.append(otherKey)
                            keyWeight += - self.expressionsDict[otherKey][1]
                if keyWeight != 0:
                    reducedExpressionDict[key] = [keyFormula, keyWeight]
        self.expressionsDict = reducedExpressionDict

    def beautify_weights(self):
        self.expressionsDict = {key: posify_weight(self.expressionsDict[key][0], self.expressionsDict[key][1]) for key
                                in self.expressionsDict
                                if self.expressionsDict[key][1] != 0}

    def visualize(self):
        return knv.visualize_knowledge(self.expressionsDict)

    def get_expressionsDict(self):
        return self.expressionsDict

    def get_formulas_and_facts(self):
        return self.expressionsDict, self.factsDict


def infer_expression(expression, evidenceDict):
    return reduce_thing_nothing(replace_evidence_variables(expression, evidenceDict))


def replace_evidence_variables(expression, evidenceDict):
    if isinstance(expression, str):
        if expression in evidenceDict.keys():
            if bool(evidenceDict[expression]) == True:
                return "Thing"
            else:
                return "Nothing"
        else:
            return expression
    elif len(expression) == 2:
        return [expression[0], replace_evidence_variables(expression[1], evidenceDict)]
    elif len(expression) == 3:
        return [replace_evidence_variables(expression[0], evidenceDict), expression[1],
                replace_evidence_variables(expression[2], evidenceDict)]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def equality_contradiction_check(expression1, expression2):
    if equality_check(expression1, expression2):
        return "equal"
    if len(expression1) > 1:
        if expression1[0] == "not":
            if equality_check(expression1[1], expression2):
                return "negequal"
    if len(expression2) > 1:
        if expression2[0] == "not":
            if equality_check(expression2[1], expression1):
                return "negequal"
    return "neither"


## Copied from logic/expression_generation
def equality_check(expression1, expression2):
    if type(expression1) == str:
        if type(expression2) == str:
            return expression1 == expression2
        else:
            return False
    elif expression1[0] == "not":
        if expression2[0] == "not":
            return equality_check(expression1[1], expression2[1])
        else:
            return False
    elif expression1[1] == "and":
        if expression2[1] == "and":
            return (equality_check(expression1[0], expression2[0]) and equality_check(expression1[2],
                                                                                      expression2[2])) or (
                    equality_check(expression1[0], expression2[2]) and equality_check(expression1[2],
                                                                                      expression2[0]))
    else:
        return expression1 == expression2


def find_new_evidence(expressionsDict, hardLogicLimit=100):
    newEvidenceDict = {}
    for key in expressionsDict:
        if abs(expressionsDict[key][1]) > hardLogicLimit:
            if type(expressionsDict[key][0]):
                if expressionsDict[key][0] in newEvidenceDict:
                    newEvidenceDict[expressionsDict[key][0]] += expressionsDict[key][1]
                else:
                    newEvidenceDict[expressionsDict[key][0]] = expressionsDict[key][1]
            elif len(expressionsDict[key][0]) == 2:
                if type(expressionsDict[key][0][1]) == str:
                    if expressionsDict[key][0][1] in newEvidenceDict:
                        newEvidenceDict[expressionsDict[key][0][1]] += -expressionsDict[key][1]
                    else:
                        newEvidenceDict[expressionsDict[key][0][1]] = -expressionsDict[key][1]
    return newEvidenceDict


def posify_weight(expression, weight):
    if weight < 0:
        if type(expression) == str:
            return [["not", expression], -weight]
        elif expression[0] == "not":
            return [expression, -weight]
        else:
            return [["not", expression], -weight]
    return [expression, weight]


def reduce_thing_nothing(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        subExpression = reduce_thing_nothing(expression[1])
        if subExpression == "Thing":
            return "Nothing"
        elif subExpression == "Nothing":
            return "Thing"
        return ["not", subExpression]

    elif len(expression) == 3:
        leftExpression = reduce_thing_nothing(expression[0])
        rightExpression = reduce_thing_nothing(expression[2])

        if expression[1] == "and":
            if leftExpression == "Thing":
                return rightExpression
            elif rightExpression == "Thing":
                return leftExpression
            elif leftExpression == "Nothing" or rightExpression == "Nothing":
                return "Nothing"
            return [leftExpression, "and", rightExpression]

        elif expression[1] == "or":
            if leftExpression == "Thing" or rightExpression == "Thing":
                return "Thing"
            elif rightExpression == "Nothing":
                return leftExpression
            elif leftExpression == "Nothing":
                return rightExpression
            return [leftExpression, "or", rightExpression]

        elif expression[1] == "xor":
            ## If both are in Thing Nothing
            if leftExpression in ["Thing", "Nothing"] and rightExpression in ["Thing", "Nothing"]:
                if leftExpression == "Thing" and rightExpression == "Nothing":
                    return "Thing"
                elif leftExpression == "Nothing" and rightExpression == "Thing":
                    return "Thing"
                else:
                    return "Nothing"
            ## If one is in Thing Nothing
            elif leftExpression == "Thing":
                return ["not", rightExpression]
            elif leftExpression == "Nothing":
                return rightExpression
            elif rightExpression == "Thing":
                return ["not", leftExpression]
            elif rightExpression == "Nothing":
                return leftExpression
            ## If no one is in Thing Nothing
            else:
                return [leftExpression, "xor", rightExpression]

        elif expression[1] == "imp":
            if leftExpression == "Nothing" or rightExpression == "Thing":
                return "Thing"
            elif rightExpression == "Nothing" and leftExpression != "Thing":
                return ["not", leftExpression]
            elif leftExpression == "Thing":
                return rightExpression
            return [leftExpression, "imp", rightExpression]

        elif expression[1] == "eq":
            ## If both are in Thing Nothing
            if leftExpression in ["Thing", "Nothing"] and rightExpression in ["Thing", "Nothing"]:
                if leftExpression == rightExpression:
                    return "Thing"
                else:
                    return "Nothing"
            # If one in Thing Nothing
            elif leftExpression == "Thing":
                return rightExpression
            elif rightExpression == "Thing":
                return leftExpression
            elif leftExpression == "Nothing":
                return ["not", rightExpression]
            elif rightExpression == "Nothing":
                return ["not", leftExpression]
            # If no one in Thing Nothing
            else:
                return [leftExpression, "eq", rightExpression]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def reduce_double_not(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        if len(expression[1][0]) == 2:
            return reduce_double_not(expression[1][1])
        else:
            return ["not", reduce_double_not(expression[1])]
    elif len(expression) == 3:
        return [reduce_double_not(expression[0]), expression[1], reduce_double_not(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


if __name__ == "__main__":
    redundantDict = {
        "f0": ["A1", 101.2312312],
        "f1": [["not", ["A2", "and", "A3"]], 1],
        "f2": ["A2", 90],
        "f3": [["not", "A2"], 90],
        "f4": ["Thing", 1],
        "f5": ["A3", -10]
    }

    # print(infer_expression(["not", ["A2", "and", "A3"]], {"A2":1}))

    lRep = LogicRepresentation(redundantDict)
    lRep.simplify()

    lRep.visualize()

    print(lRep.expressionsDict)

    lRep.forward_chaining()
    print(lRep.expressionsDict)

    lRep.infer({"A2": 1}, simplify=True)
    print(lRep.expressionsDict)

    lRep.remove_doubles()
    print(lRep.expressionsDict)

    and_expression = ['not', [['not', 'jaszczur'], 'and',
                              ['not', ['not', [['not', 'sikorka'], 'and', ['not', ['not', 'sledz']]]]]]]
    assert reduce_double_not(and_expression) == ['not',
                                                 [['not', 'jaszczur'], 'and', [['not', 'sikorka'], 'and', 'sledz']]], \
        "Generate from disjunctions or double not does not work"
    assert reduce_double_not(["not", ["not", "sledz"]]) == "sledz", "Removing double not does not work"

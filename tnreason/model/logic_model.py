from tnreason.model import model_visualization as mv


class LogicRepresentation:
    def __init__(self, expressionsDict):
        self.expressionsDict = expressionsDict

    def infer(self, evidenceDict, simplify=True):
        self.expressionsDict = {
            key: [replace_evidence_variables(self.expressionsDict[key][0], evidenceDict), self.expressionsDict[key][1]]
            for key in
            self.expressionsDict
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
        self.remove_doubles()
        self.beautify_weights()

    def remove_thing_nothing(self):
        newExpressionsDict = {}
        for key in self.expressionsDict:
            newExpression = reduce_thing_nothing(self.expressionsDict[key][0])
            if newExpression not in ["Thing", "Nothing"]:
                newExpressionsDict[key] = [newExpression, self.expressionsDict[key][1]]
        self.expressionsDict = newExpressionsDict

    def remove_double_nots(self):
        self.expressionsDict = {key: [reduce_double_not(self.expressionsDict[key][0]), self.expressionsDict[key][1]] for
                                key in self.expressionsDict}

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

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None):
        return mv.visualize_model(self.expressionsDict,
                                  strengthMultiplier=strengthMultiplier,
                                  strengthCutoff=strengthCutoff,
                                  fontsize=fontsize,
                                  showFormula=showFormula,
                                  evidenceDict=evidenceDict,
                                  pos=pos)


def infer_expression(expression, evidenceDict):
    return reduce_thing_nothing(replace_evidence_variables(expression, evidenceDict))


def replace_evidence_variables(expression, evidenceDict):
    # print(expression, type(expression)==str, expression in list(evidenceDict.keys()))
    if type(expression) == str:
        if expression in evidenceDict.keys():
            if bool(evidenceDict[expression]) == True:
                return "Thing"
            else:
                return "Nothing"
        else:
            return expression
    elif expression[0] == "not":
        return ["not", replace_evidence_variables(expression[1], evidenceDict)]
    elif expression[1] == "and":
        return [replace_evidence_variables(expression[0], evidenceDict), "and",
                replace_evidence_variables(expression[2], evidenceDict)]


def reduce_thing_nothing(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        rightExpression = reduce_thing_nothing(expression[1])
        if rightExpression == "Thing":
            return "Nothing"
        elif rightExpression == "Nothing":
            return "Thing"
        return ["not", rightExpression]
    elif expression[1] == "and":
        leftExpression = reduce_thing_nothing(expression[0])
        rightExpression = reduce_thing_nothing(expression[2])
        if leftExpression == "Thing":
            return rightExpression
        elif rightExpression == "Thing":
            return leftExpression
        elif leftExpression == "Nothing" or rightExpression == "Nothing":
            return "Nothing"
        return [leftExpression, "and", rightExpression]


def reduce_double_not(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        if expression[1][0] == "not":
            return reduce_double_not(expression[1][1])
        else:
            return ["not", reduce_double_not(expression[1])]
    elif expression[1] == "and":
        return [reduce_double_not(expression[0]), "and", reduce_double_not(expression[2])]
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

    lRep.visualize(strengthCutoff=3)

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

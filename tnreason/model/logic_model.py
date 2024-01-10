from tnreason.logic import expression_generation as eg


class LogicRepresentation:
    def __init__(self, expressionsDict):
        self.expressionsDict = expressionsDict

    def infer(self, evidenceDict, removeRedundancy=True):
        self.expressionsDict = {
            key: [infer_expression(self.expressionsDict[key][0], evidenceDict), self.expressionsDict[key][1]] for key in
            self.expressionsDict
        }
        if removeRedundancy:
            self.remove_redundancy()

    def remove_redundancy(self):
        # Removes Expressions which are Thing or Nothing
        checkedKeys = []
        reducedExpressionDict = {}
        for key in self.expressionsDict:
            if key not in checkedKeys and self.expressionsDict[key][0] not in ["Thing", "Nothing"]:
                checkedKeys.append(key)
                keyFormula, keyWeight = self.expressionsDict[key]
                for otherKey in self.expressionsDict:
                    if otherKey not in checkedKeys and eg.equality_check(keyFormula, self.expressionsDict[otherKey][0]):
                        checkedKeys.append(otherKey)
                        keyWeight = keyWeight + self.expressionsDict[otherKey][1]
                if keyWeight != 0:
                    reducedExpressionDict[key] = [keyFormula, keyWeight]
        self.expressionsDict = reducedExpressionDict


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


if __name__ == "__main__":
    redundantDict = {
        "f0": ["A1", 1],
        "f1": [["not", ["A2", "and", "A3"]], 1],
        "f2": ["A2", -1],
        "f3": ["A2", 1.1],
        "f4": ["Thing", 1]
    }

    # print(infer_expression(["not", ["A2", "and", "A3"]], {"A2":1}))

    lRep = LogicRepresentation(redundantDict)
    lRep.infer({"A2": 1}, removeRedundancy=False)
    print(lRep.expressionsDict)

    lRep.remove_redundancy()
    print(lRep.expressionsDict)

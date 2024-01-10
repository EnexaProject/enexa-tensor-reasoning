def infer_expression(expression, evidenceDict):
    return reduce_thing_nothing(replace_evidence_variables(expression, evidenceDict))


def replace_evidence_variables(expression, evidenceDict):
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
    replaced = replace_evidence_variables(["a", "and", ["not", "b"]], {"a": 0})
    print(replaced)
    reduced = reduce_thing_nothing(replaced)
    print(reduced)

    print(reduce_thing_nothing(['Nothing', 'and', ['not', 'Rechnung(x)']]))

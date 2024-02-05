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
            if leftExpression in ["Thing","Nothing"] and rightExpression in ["Thing", "Nothing"]:
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
                return ["not",rightExpression]
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
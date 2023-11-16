from tnreason.logic import expression_calculus as ec

def count_satisfaction(expression):
    conjunctionList = compute_conjustionList(expression)
    count = 0
    for entry in conjunctionList:
        count += 2**len(entry[0])*entry[1]
    return count

def compute_conjustionList(expression):
    if type(expression) == str:
        return [[[],1]]
    if expression[0] == "not":
        variables = ec.get_variables(expression)
        conjunctionList1 = compute_conjustionList(expression[1])
        conjunctionList = [[variables,1]]
        for entry in conjunctionList1:
            conjunctionList.append([entry[0],-1*entry[1]])
        return conjunctionList
    elif expression[1] == "and":
        conList0 =  compute_conjustionList(expression[0])
        conList2 =  compute_conjustionList(expression[2])

        conjunctionList = []
        for entry0 in conList0:
            for entry2 in conList2:
                conjunctionList.append([entry0[0]+entry2[0], entry0[1]*entry2[1]])
        return conjunctionList
    else:
        raise ValueError("Expression {} not understood.".format(expression))

if __name__ == "__main__":
    print(count_satisfaction("test"))
    print(count_satisfaction(["not",["test","and","sledz"]]))
    print(count_satisfaction(["not",["losos","and",["sledz","and","sikorka"]]]))

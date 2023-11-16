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
        variables = combine_unique(ec.get_variables(expression),[])
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
                conjunctionList.append([combine_unique(entry0[0],entry2[0]), entry0[1]*entry2[1]])
        return conjunctionList
    else:
        raise ValueError("Expression {} not understood.".format(expression))

def combine_unique(list1,list2):
    list = []
    for element in list1+list2:
        if element not in list:
            list.append(element)
    return list

if __name__ == "__main__":
    assert count_satisfaction("sledz")==1, "Satisfaction Counter wrong"
    assert count_satisfaction(["sikorka","and",["not",["jaszczur","and","sledz"]]])==3, "Satisfaction Counter wrong"
    assert count_satisfaction(["not",["test","and","sledz"]])==3, "Satisfaction Counter wrong"
    assert count_satisfaction(["not",["losos","and",["sledz","and","sikorka"]]])==7, "Satisfaction Counter wrong"
    assert count_satisfaction(["not",["losos","and",["losos","and","sikorka"]]])==3, "Satisfaction Counter wrong"

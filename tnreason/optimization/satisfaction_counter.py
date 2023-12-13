from tnreason.logic import expression_utils as eu

def count_satisfaction(expression):
    conjunctionList = compute_conjunctionList(expression)
    count = 0
    for entry in conjunctionList:
        count += 2**len(entry[0])*entry[2]
    return count

def compute_conjunctionList(expression):
    if type(expression) == str:
        return [[[],[expression],1]]
    if expression[0] == "not":
        variables = combine_unique(eu.get_variables(expression),[])
        conjunctionList1 = compute_conjunctionList(expression[1])
        conjunctionList = [[variables,[],1]]
        for entry in conjunctionList1:
            conjunctionList.append([entry[0],entry[1],-1*entry[2]])
        return conjunctionList
    elif expression[1] == "and":
        conList0 =  compute_conjunctionList(expression[0])
        conList2 =  compute_conjunctionList(expression[2])

        conjunctionList = []
        for entry0 in conList0:
            for entry2 in conList2:
                conjunctionList.append(merge_entries(entry0,entry2))
        return conjunctionList
    else:
        raise ValueError("Expression {} not understood.".format(expression))

def merge_entries(entry0, entry2):
    touchedVariables = entry0[1]+entry2[1]
    preuntouchedVariables = combine_unique(entry0[0],entry2[0])
    untouchedVariables = []
    for variable in preuntouchedVariables:
        if variable not in touchedVariables:
            untouchedVariables.append(variable)
    return [untouchedVariables, touchedVariables, entry0[2]*entry2[2]]

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

    #print(compute_conjunctionList(["jaszczur","and",["not",["jaszczur","and","sledz"]]]))
    print(count_satisfaction(["sledz","and",["not","sledz"]]))
    assert count_satisfaction(["jaszczur","and",["not",["jaszczur","and","sledz"]]])==1, "Satisfaction Counter wrong"
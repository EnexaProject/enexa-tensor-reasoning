from tentris import Variable

def sparql_evaluation_to_entryPositionList(querySolution, interpretationDict={}):
    """
    :querySolution: Instance of tentris.tentris.SPARQLSolutionGenerator
    :interpretationDict: Dictionary storing to each variable a list of string IRI identifiers
    :return: List of Dictionaries assigning an index to each variable, Modified interpretationDict
    """
    projectedVariables = [str(variable)[1:] for variable in querySolution.projected_variables]
    for variable in projectedVariables:
        if variable not in interpretationDict:
            interpretationDict[variable] = []

    entryPositionList = []
    for entry in iter(querySolution):
        positionDict = {}
        for variable in projectedVariables:
            variableString = str(entry[0][Variable(variable)])
            if variableString not in interpretationDict[variable]:
                interpretationDict[variable].append(variableString)
            positionDict[variable] = interpretationDict[variable].index(variableString)
        entryPositionList.append(positionDict)

    return entryPositionList, interpretationDict
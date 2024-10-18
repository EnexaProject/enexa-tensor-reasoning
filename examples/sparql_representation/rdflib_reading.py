import rdflib

def rdflib_sparql_evaluation_to_entryPositionList(querySolution, interpretationDict=dict()):
    """
    Handling of evaluated queries of rdflib
    :querySolution: Solution iterator of rdflib
    :interpretationDict: Dictionary of lists to each projection variable defining the position of individuals
    :return: List of Dictionaries assigning an index to each variable, Modified interpretationDict
    """
    projectedVariables = [str(variable) for variable in querySolution.vars]
    for variable in projectedVariables:
        if variable not in interpretationDict:
            interpretationDict[variable] = []

    entryPositionList = []
    for entry in iter(querySolution):
        positionDict = {}
        for variable in projectedVariables:
            variableString = str(entry[variable])
            if variableString not in interpretationDict[variable]:
                interpretationDict[variable].append(variableString)
            positionDict[variable] = interpretationDict[variable].index(variableString)
        entryPositionList.append(positionDict)

    return entryPositionList, interpretationDict

def rdflib_evaluate_query(rdfFilePath, queryString):
    g = rdflib.Graph()
    g.parse(rdfFilePath)
    return g.query(queryString)


if __name__ == "__main__":
    queryString = """
        SELECT ?x ?z ?y
        WHERE {
            ?x ?z ?y .
        }
    """
    result = rdflib_evaluate_query(rdfFilePath="./example_kg/THWS_demo.ttl", queryString=queryString)

    entryPositionList, interpretationDict = rdflib_sparql_evaluation_to_entryPositionList(result)
    print(entryPositionList)
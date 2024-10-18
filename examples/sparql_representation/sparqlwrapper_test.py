from SPARQLWrapper import SPARQLWrapper, JSON

def wrapper_json_sparql_evaluation_to_entryPositionList(querySolution, interpretationDict=dict(),
                                                        identifierCutoffLength=30):
    projectedVariables = [str(variable) for variable in querySolution["head"]["vars"]]
    for variable in projectedVariables:
        if variable not in interpretationDict:
            interpretationDict[variable] = []

    entryPositionList = []
    for entry in querySolution["results"]["bindings"]:
        positionDict = {}
        for variable in projectedVariables:
            variableString = str(entry[variable]["value"][:identifierCutoffLength])
            if variableString not in interpretationDict[variable]:
                interpretationDict[variable].append(variableString)
            positionDict[variable] = interpretationDict[variable].index(variableString)
        entryPositionList.append(positionDict)

    return entryPositionList, interpretationDict


def wrapper_json_evaluate_query(endpointString, queryString):
    sparql = SPARQLWrapper(endpointString)
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


if __name__ == "__main__":

    results = wrapper_json_evaluate_query(endpointString="https://dbpedia.org/sparql", queryString="""
        SELECT DISTINCT ?x ?y ?z
        WHERE { 
            ?x ?y ?z.
        }
        LIMIT 10
    """)

    entryPositionList, imDict = wrapper_json_sparql_evaluation_to_entryPositionList(results, dict())
    print(entryPositionList)

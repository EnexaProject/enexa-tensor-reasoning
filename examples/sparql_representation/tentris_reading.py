from tentris import tentris, Variable


def tentris_sparql_evaluation_to_entryPositionList(querySolution, interpretationDict=dict()):
    """
    Handling of evaluated queries of Tentris
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


def tentris_evaluate_query(rdfFilePath, queryString):
    tripler = tentris.TripleStore()
    tripler.load_rdf_data(rdfFilePath)
    return tripler.eval_sparql_query(queryString)


if __name__ == "__main__":
    queryString = """
        SELECT ?x ?z ?y
        WHERE {
            ?x ?z ?y .
        }
    """
    querySolution = tentris_evaluate_query(rdfFilePath="/home/examples/sparql_representation/example_kg/THWS_demo.ttl",
                                           queryString=queryString)

    entryPositionList, interpretationsDict = tentris_sparql_evaluation_to_entryPositionList(querySolution)

    from examples.sparql_representation import extract_datacores as ed

    core = ed.positionList_to_polynomialCore(entryPositionList, variables=[str(variable)[1:] for variable in
                                                                           querySolution.projected_variables])
    print(core.values)

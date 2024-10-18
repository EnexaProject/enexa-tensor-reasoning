from tentris import Variable

def sparql_evaluation_to_entryPositionList(querySolution, interpretationDict={}):
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


if __name__ == "__main__":
    from tentris import tentris, Hypertrie, Variable

    from examples.sparql_representation.tentris_reading import sparql_evaluation_to_entryPositionList

    tripler = tentris.TripleStore()
    tripler.load_rdf_data("/home/examples/hypertrie_cores/THWS_demo.ttl")
    #tripler.hypertrie()

    queryString = """
        SELECT ?x ?z ?y
        WHERE {
            ?x ?z ?y .
        }
    """

    querySolution = tripler.eval_sparql_query(queryString)
    projectionVariables = [str(variable)[1:] for variable in querySolution.projected_variables]

    from tnreason.engine.polynomial_contractor import PolynomialCore, SliceValues

    entryPositionList, interpretationsDict = sparql_evaluation_to_entryPositionList(querySolution)
    core = PolynomialCore(
        values=[(1, posDict) for posDict in entryPositionList],
        colors=projectionVariables
    )
    print(entryPositionList)
    print(interpretationsDict)


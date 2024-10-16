from tnreason import engine


def clause_to_pcore(variablesDict):
    return engine.get_core("PolynomialCore")(
        values=engine.SliceValues([(1, dict()), (-1, {vKey: 1 - variablesDict[vKey] for vKey in variablesDict})],
                                  shape=[2 for vKey in variablesDict]),
        colors=list(variablesDict.keys())
    )


def find_atoms(clauseList):
    return list(set.union(*[set(clauses.keys()) for clauses in clauseList]))


def clauseList_to_pcore(clauseList):
    return engine.contract(
        coreDict={"c" + str(i): clause_to_pcore(variablesDict) for i, variablesDict in enumerate(clauseList)},
        openColors=find_atoms(clauseList),
        method="PolynomialContractor"
        )


if __name__ == "__main__":
    testClause = [{"c": 0, "b": 1}, {"a": 0}, {"d":0, "a":1}]

    print(clauseList_to_pcore(testClause))

from tnreason import engine
from tnreason.engine.creation_handling import core_to_relational_encoding


def get_dataCores(importanceQueryCore, atomQueryCoreDict, dataColor="j", coreType=None):
    importanceQueryCore.enumerate_slices(enumerationColor=dataColor)
    return {atomKey + "_dataCore": core_to_relational_encoding(
        core=engine.contract({"imCore": importanceQueryCore, atomKey: atomQueryCoreDict[atomKey]},
                             openColors=[dataColor], method="PolynomialContractor"), headColor=atomKey,
        outCoreType=coreType)[0] for atomKey in atomQueryCoreDict}


if __name__ == "__main__":
    from tnreason.engine.polynomial_contractor import PolynomialCore, SliceValues
    from examples.sparql_representation import rdflib_reading as rr
    import rdflib

    g = rdflib.Graph()
    g.parse("../tests/engine/hypertrie_cores/THWS_demo.ttl")

    queryString = """
        SELECT DISTINCT ?x ?z ?y
        WHERE {
            ?x ?z ?y .
        }
    """

    result = g.query(queryString)
    importancePositionList, interpretationDict = rr.rdflib_sparql_evaluation_to_entryPositionList(result)



    importanceCore = PolynomialCore(
        values=[(1, posDict) for posDict in importancePositionList],
        shape=[10, 10, 10],
        colors=["x", "z", "y"]
    )

    atomAString = """
            SELECT DISTINCT ?x 
            WHERE {
                ?x ?z ?y .
            }
        """
    result = g.query(atomAString)
    atomAPositionList, interpretationDict = rr.rdflib_sparql_evaluation_to_entryPositionList(result, interpretationDict)
    atomACore = PolynomialCore(
        values=[(1, posDict) for posDict in atomAPositionList],
        shape=[10, 10, 10],
        colors=["x"]
    )

    dataCores = get_dataCores(importanceCore, atomQueryCoreDict={"aCore": atomACore})
    print(dataCores["aCore_dataCore"].values)
from tnreason import engine
from tnreason.engine.creation_handling import core_to_relational_encoding


def get_dataCores(importanceQueryCore, atomQueryCoreDict, dataColor="j", coreType=None,
                  contractionMethod="PolynomialContractor"):
    """
    :importanceQueryCore: Tensor Core representing the evaluation of the importance query (before slice enumeration!)
    :atomQueryCoreDict: Dictionary of Tensor Cores representing the evaluation of the atom extraction queries
    :dataColor: Color of the entry enumeration in the importanceQueryCore, which will be interpreted as the data color
    :coreType: Type of the resulting data cores
    """
    importanceQueryCore.enumerate_slices(enumerationColor=dataColor)
    return {atomKey + "_dataCore": core_to_relational_encoding(
        core=engine.contract({"imCore": importanceQueryCore, atomKey: atomQueryCoreDict[atomKey]},
                             openColors=[dataColor], method=contractionMethod), headColor=atomKey,
        outCoreType=coreType)[0] for atomKey in atomQueryCoreDict}


def positionList_to_polynomialCore(positionList, variables=[], shape=[]):
    """
    Turns outputs of query evaluations (rdflib or tentris) into polynomial cores for further contraction
    """
    return engine.get_core("PolynomialCore")(
        values=[(1, posDict) for posDict in positionList],
        shape=shape,
        colors=variables
    )


if __name__ == "__main__":
    from examples.sparql_representation import rdflib_reading as rr
    import rdflib

    g = rdflib.Graph()
    g.parse("./example_kg/THWS_demo.ttl")

    queryString = """
        SELECT DISTINCT ?x ?z ?y
        WHERE {
            ?x ?z ?y .
        }
    """

    result = g.query(queryString)
    importancePositionList, interpretationDict = rr.rdflib_sparql_evaluation_to_entryPositionList(result)
    importanceCore = positionList_to_polynomialCore(importancePositionList, variables=["x", "z", "y"],
                                                    shape=[10, 10, 10])

    atomAString = """
            SELECT DISTINCT ?x 
            WHERE {
                ?x ?z ?y .
            }
        """
    result = g.query(atomAString)
    atomAPositionList, interpretationDict = rr.rdflib_sparql_evaluation_to_entryPositionList(result, interpretationDict)
    atomACore = positionList_to_polynomialCore(atomAPositionList, variables=["x"], shape=[10])

    dataCores = get_dataCores(importanceCore, atomQueryCoreDict={"aCore": atomACore})
    print(dataCores["aCore_dataCore"].values)

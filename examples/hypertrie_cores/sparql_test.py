from tentris import tentris, Hypertrie, Variable

from examples.sparql_representation.tentris_reading import sparql_evaluation_to_entryPositionList

tripler = tentris.TripleStore()
tripler.load_rdf_data("/home/examples/hypertrie_cores/THWS_demo.ttl")
tripler.hypertrie()

queryString = """
    SELECT ?x ?z ?y
    WHERE {
        ?x ?z ?y .
    }
"""

querySolution = tripler.eval_sparql_query(queryString)
projectionVariables = [str(variable)[1:] for variable in querySolution.projected_variables]


from tnreason.engine.polynomial_contractor import PolynomialCore, SliceValues

entryPositionList , _ = sparql_evaluation_to_entryPositionList(querySolution)
core = PolynomialCore(
    values=SliceValues([(1, posDict) for posDict in entryPositionList]),
    colors=projectionVariables
)

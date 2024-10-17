from tnreason import engine
from examples.kg_representation import find_individuals as fi

import rdflib

def query_as_polynomial(kg, query, individuals, variables=[]):
    query = kg.query(query)
    slices = []
    for row in query:
        slices.append([1,{variable : individuals.index(str(row[i])) for i,variable in enumerate(variables)}])
    return engine.get_core("PolynomialCore")(values=slices, colors=variables)



if __name__ == "__main__":
    kg_example_string = """
    @prefix loc: <http://www.locationdemo.de/location/ontology#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix xml: <http://www.w3.org/XML/1998/namespace> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    loc:Alice rdf:type loc:Student ;
        loc:studiesSubject loc:TechnoMathematik .

    loc:Bob rdf:type loc:Student ;
        loc:studiesSubject loc:ArtificialIntelligence .

    loc:TechnoMathematik rdf:type loc:AppliedScienceSubject .

    loc:ArtificialIntelligence rdf:type loc:ComputerScienceSubject .

    loc:THWS rdf:type loc:University ;
        loc:hasCampusIn loc:Schweinfurt .

    loc:Schweinfurt rdf:type loc:City ;
        loc:locatedIn loc:Germany .

    loc:Germany rdf:type loc:Country .

    """

    schweinfurt_query = """
    SELECT ?student ?x ?y
    WHERE {
      ?student ?x ?y.
    }
    """

    rdf_kg = rdflib.Graph()
    rdf_kg.parse(data=kg_example_string)
    print(query_as_polynomial(rdf_kg,
                              query=schweinfurt_query,
                              individuals=fi.get_individuals_list(rdf_kg),
                              variables=["student", "fun", "furtherfun"]))
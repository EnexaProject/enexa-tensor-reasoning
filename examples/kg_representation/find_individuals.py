import rdflib

subject_query = """
SELECT ?subject
WHERE {
    ?subject ?predicate ?object
}
"""

predicate_query = """
SELECT ?predicate
WHERE {
    ?subject ?predicate ?object
}
"""

object_query = """
SELECT ?object
WHERE {
    ?subject ?predicate ?object
}
"""

def get_individuals_list(kg):
    subjects = [str(row.subject) for row in kg.query(subject_query)]
    predicates = [str(row.predicate) for row in kg.query(predicate_query)]
    objects = [str(row.object) for row in kg.query(object_query)]
    return list(set(subjects).union(set(predicates)).union(set(objects)))

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

    rdf_kg = rdflib.Graph()
    rdf_kg.parse(data=kg_example_string)
    print(get_individuals_list(rdf_kg))
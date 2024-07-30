from rdflib import Graph, URIRef, RDF, Namespace, OWL

def sampleDf_to_naiveKG(sampleDf, domain="http://example.org"):
    g = Graph()
    EX = Namespace(domain)
    g.bind("ex", EX)
    ## TBox
    for atom in sampleDf.columns:
        g.add((URIRef(EX[atom]), RDF.type, OWL.Class))
    ## ABox
    for i, row in sampleDf.iterrows():
        sampleEntity = URIRef(EX["sample"+str(i)])
        for atom in row.keys():
            if row[atom] == 1:
                classEntity = URIRef(EX[atom])
                g.add((sampleEntity, RDF.type, classEntity))
    return g

def sampleDf_to_factoredKG(sampleDf, domain,
                           projectionVariables,
                           importanceQuery, atomTriplesDict):
    g = Graph()
    EX = Namespace(domain)
    g.bind("ex", EX)


    for i, row in sampleDf.iterrows():
        ## Abstract reproduction of the extraction query:
        # sampleEntity = URIRef(EX["sample"+str(i)])
        # g.add((sampleEntity, RDF.type, EX.Sample))
        # for variable in projectionVariables:
        #     variableClass = URIRef(EX[variable])
        #     sampleVariableEntitiy = URIRef(EX[variable+"_"+str(i)])
        #     g.add((sampleVariableEntitiy, EX.toSample, sampleEntity))
        #     g.add((sampleVariableEntitiy, RDF.type, variableClass))
        ## Customized Reproduction of the extraction query
        for importanceKey in importanceQuery.keys():
            s, p, o = importanceQuery[importanceKey]
            if s in projectionVariables:
                subEntity = URIRef(EX[s +"_"+str(i)])
            else:
                subEntity = URIRef(EX[s])

            if p in projectionVariables:
                predEntity = URIRef(EX[p +"_"+str(i)])
            elif p == "rdf:type":
                predEntity = RDF.type
            else:
                predEntity = URIRef(EX[p])

            if o in projectionVariables:
                obEntity = URIRef(EX[o+"_"+str(i)])
            else:
                obEntity = URIRef(EX[o])
            g.add((subEntity,predEntity,obEntity))

        # Reproducing the atom queries
        for atom in row.keys():
            if row[atom] == 1:
                s, p, o = atomTriplesDict[atom]
                if s in projectionVariables:
                    subEntity = URIRef(EX[s +"_"+str(i)])
                else:
                    subEntity = URIRef(EX[s])

                if p in projectionVariables:
                    predEntity = URIRef(EX[p +"_"+str(i)])
                elif p == "rdf:type":
                    predEntity = RDF.type
                else:
                    predEntity = URIRef(EX[p])

                if o in projectionVariables:
                    obEntity = URIRef(EX[o+"_"+str(i)])
                else:
                    obEntity = URIRef(EX[o])

                g.add((subEntity, predEntity,obEntity))
    return g




if __name__ == "__main__":
    from tnreason import knowledge
    dist = knowledge.HybridKnowledgeBase(weightedFormulas={
        "f1" : ["or","alpha","beta", 0.4],
        "f2" : ["gamma",0.3]
    })

    sampleDf_to_naiveKG(knowledge.InferenceProvider(dist).draw_samples(10), "http://fun.org").serialize(
        destination="./generated/naiveKG.ttl",format="turtle")

    factoredKG = sampleDf_to_factoredKG(knowledge.InferenceProvider(dist).draw_samples(10), "http://fun.org",
                                        projectionVariables=["v1","v2","v3"], # v1 -> Beleg, v2 -> Buchung
                                        importanceQuery={
                                            "con1" : ["v1", "extractionRelation", "v2"],
                                            "con2" : ["v1", "secExtractionRelation", "v3"]
                                        },
                                        atomTriplesDict={
                                            "alpha" : ["v1", "rdf:type", "v1Property"],
                                            "beta" : ["v1", "v1v2Relation", "v2"],
                                            "gamma" : ["v3", "rdf:type", "v3Property"]
                                        })
    factoredKG.serialize(
        destination="./generated/factoredKG.ttl",format="turtle")
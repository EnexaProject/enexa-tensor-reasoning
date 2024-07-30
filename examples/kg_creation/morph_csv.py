import morph_kgc

basePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/examples/kg_creation/"

g_rdflib = morph_kgc.materialize(basePath+"specfiles/morph_config.ini")
g_rdflib.serialize(basePath+"generated/generated.ttl")
import rdflib
import pandas as pd
import time

def generate_ttl(csvPath, limit=None):
    dataFrame = pd.read_csv(csvPath)
    g = dataframe_to_kg(dataFrame)
    return g

def dataframe_to_kg(dataFrame):
    g = rdflib.Graph()
    for i, row in dataFrame.iterrows():
        subject = rdflib.URIRef(row["subject"])
        predicate = rdflib.URIRef(row["predicate"])
        object = rdflib.URIRef(row["object"])
        g.add((subject,predicate,object))
    return g
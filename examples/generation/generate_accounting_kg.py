from tnreason.model import generate_test_data as gtd
from tnreason.representation import csv_to_ttl as ctt

example_expression_dict = {
    "e0": ["Rechnung(x)", 1.5],
    "e1": [["not", "Ausgangsrechnung(x)"], 1.5],
    "e2": [["Ausgangsrechnung(x)", "and", "zuMandant(x,y)"], 2],
    "e3": [[["not", "Ausgangsrechnung(y)"], "and", ["not", "Rechnung(x)"]], 2],
    "e4": [["Ausgangsrechnung(x)", "and", "Rechnung(y)"], 2],
    "e5": ["Spezialrechnung(x)", 1.5],
    "e6": [["versandt(y,x)", "and", "Ausgangsrechnung(x)"], 2],
    "e7": [["bearbeitet(y,x)", "and", "enthaelt(x,z)"], 1.2]
}

dataNum = 10
factDf, pairDf = gtd.generate_factDf_and_pairDf(example_expression_dict, sampleNum=dataNum, prefix="tev:")

savePath = "./examples/model/synthetic_test_data/"
factDf.to_csv(savePath+"generated_factDf.csv")

g = ctt.dataframe_to_kg(factDf)
g.serialize(savePath+"generated_kg.ttl")
from tnreason.model import generate_test_data as gtd
from tnreason.representation import csv_to_ttl as ctt
from tnreason.logic import expression_generation as eg

## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r0": [["Unterschrank(z)"], "Moebel(z)", 1.5],
    "r1": [["hatLeistungserbringer(x,y)", "versandterBeleg(y,x)"], "Ausgangsrechnung(x)", 1.5],
    "r2": [["Ausgangsrechnung(x)", "versandterBeleg(y,x)", "Bautischlerei(y)", "hatBelegzeile(x,z)", "Moebel(z)", "verbuchtDurch(z,q)"],"Umsatzerloese(q)", 1.5]
}

## We transform the rules into propositional formulas containing negations and conjunctions only
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}

## A list of facts is generated using Gibbs Sampling of the associated Markov Logic Network
dataNum = 1000
factDf, pairDf = gtd.generate_factDf_and_pairDf(example_expression_dict, sampleNum=dataNum, prefix="tev:")
sampleDf = gtd.generate_sampleDf(example_expression_dict, sampleNum=dataNum, chainSize=10)

savePath = "./examples/generation/synthetic_test_data/"
factDf.to_csv(savePath+"generated_factDf.csv")
sampleDf.to_csv(savePath+"generated_sampleDf.csv")

g = ctt.dataframe_to_kg(factDf)
g.serialize(savePath+"generated_kg.ttl")
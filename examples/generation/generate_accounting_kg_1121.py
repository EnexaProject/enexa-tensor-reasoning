from tnreason.model import generate_test_data as gtd
from tnreason.model import markov_logic_network as mln
from tnreason.representation import csv_to_ttl as ctt
from tnreason.logic import expression_generation as eg

# 21.11.2023: Inference-Bsp für Tischlerei Musterholz - Ausgangsrechung Unterschrank

# root_class_dict = {
#     # FEATURES
#     "beleg" : "tev:Beleg",
#     "mandant" : "tev:Unternehmen",
#     "pklasse" :"tev:Produktklasse", #'tev:Produktklasse_C3102 --> Küchenmöbel
#     "usatz": "tev:Umsatzsteuer",
#     "buchung" : "tev:Buchung",
#     "branche" : "tev:Branche", # 'tev:Branche_43.32.0(branche)' --> "43.32.0_Bautischlerei und -schlosserei",

#       # LABEL
#      "ggkto": "tev:Buchungskonto",
#      "bschluessel": "tev:Buchungsschluessel",
#      "sid" : "tev:SteuerinfoID",
#      "tatb" : "tev:Tatbestand",
# }

# Bausteine
# 'tev:Ausgangsrechnung(beleg)', 'tev:Branche_43.32.0(branche)' , 'tev:Produktklasse_C3102(pklasse)', 'tev:Umsatzerlöse(ggkto)' # ODER 'tev:RefKonto_401000_0(ggkto)'
# 'tev:Umsatzsteuer_19(usatz)', 'tev:Buchungsschluessel_0(bschluessel)', 'tev:steuerinfoID_27601900.0(sid)','tev:tatbestandID_20001(tatb)'
# tev:Inlandsbeleg(beleg) # Anmerkung: Behelfsklasse; das ist so noch nicht im KG drin, muss ggf nochmal geändert werden



## Each rule is stored as value in the dictionary, and has format [list of premises, head, MLN weight]
example_rule_dict = {
    "r1": [['Ausgangsrechnung(beleg)', 'Branche_43.32.0(branche)', 'Produktklasse_C3102(pklasse)' ],'RefKonto_401000_0(ggkto)', 1.5],
    "r2": [['Ausgangsrechnung(beleg)', 'Branche_43.32.0(branche)', "Inlandsbeleg(beleg)", 'Umsatzsteuer_19(usatz)'],
           'Buchungsschluessel_0(bschluessel)', 1.5],
    "r3": [['Ausgangsrechnung(beleg)', 'Branche_43.32.0(branche)', "Inlandsbeleg(beleg)", 'Umsatzsteuer_19(usatz)'], 'steuerinfoID_27601900.0(sid)', 1.5],
    "r4": [['Ausgangsrechnung(beleg)', 'Branche_43.32.0(branche)', "Inlandsbeleg(beleg)", 'Umsatzsteuer_19(usatz)'],'tatbestandID_20001(tatb)', 1.5]

}

## We transform the rules into propositional formulas containing negations and conjunctions only
example_expression_dict = {key:[eg.generate_list_from_rule(value[0],value[1]), value[2]] for (key,value) in example_rule_dict.items()}

## A list of facts is generated using Gibbs Sampling of the associated Markov Logic Network
dataNum = 1000
factDf, pairDf = gtd.generate_factDf_and_pairDf(example_expression_dict, sampleNum=dataNum, prefix="tev:")

generator = mln.MarkovLogicNetwork(example_expression_dict)
sampleDf = generator.generate_sampleDf(sampleNum = dataNum, chainSize = 10)

savePath = "./examples/generation/internal_real_data/"
factDf.to_csv(savePath+"generated_factDf.csv")
sampleDf.to_csv(savePath+"generated_sampleDf.csv")

g = ctt.dataframe_to_kg(factDf)
g.serialize(savePath+"generated_kg.ttl")
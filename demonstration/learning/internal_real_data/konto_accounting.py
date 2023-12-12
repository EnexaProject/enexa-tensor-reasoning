import pandas as pd
from tnreason.learning import expression_learning as el
from tnreason.logic import expression_calculus as ec
from tnreason.logic.expression_generation import generate_from_or_expression

## DATA LOADING ##

# path = "S:/_MariaSF/extractedDataFrame/extracted_20230925_150557.csv" # Sep
path = "S:/_MariaSF/extractedDataFrame/extracted_20231120_150937.csv" # Okt

sampleDf = pd.read_csv(path, index_col=0)
print(sampleDf.var())
sampleDf.var().to_csv("demonstration/learning/internal_real_data/variance/"+path.split("/")[-1])

## EXPRESSION LEARNING ##

skeletonExpression = [["b1(beleg)","and", "C1(branche)"],
                      "and",
                      ["P1(pklasse)","and", "C2(rform)"],]
                       #"and",["C2(rform)","and", "b4(bsatz)"]]]
# skeletonExpression = "b1(beleg)"
# skeletonExpression = ["b1(beleg)","and", "C1(branche)"]
# skeletonExpression = "b4(bsatz)"
# skeletonExpression = ["b4(bsatz)", "and", "b3(wjb)"]

#

candidatesDict = {
    "b1(beleg)": ["tev:Ausgangsrechnung(beleg)", "tev:Eingangsrechnung(beleg)"],
   # "b2(usatz)" :['tev:Umsatzsteuer_0(usatz)', 'tev:Umsatzsteuer_19(usatz)',
   #    'tev:Umsatzsteuer_5(usatz)', 'tev:Umsatzsteuer_7(usatz)'],
   # "b3(wjb)" : ['tev:WJ_20190101(wjb)',
   #    'tev:WJ_20200101(wjb)', 'tev:WJ_20210101(wjb)', 'tev:WJ_20220101(wjb)',
   #    'tev:WJ_20230101(wjb)'],
   #  "b4(bsatz)" : ['tev:Buchungssteuersatz_0.00(bsatz)',
   #     'tev:Buchungssteuersatz_16.00(bsatz)',
   #     'tev:Buchungssteuersatz_19.00(bsatz)',
   #     'tev:Buchungssteuersatz_7.00(bsatz)'],

    "C2(rform)": ['tev:Einzelunternehmen(rform)', # 'tev:GbR(rform)',
           # 'tev:Genossenschaft(rform)',
                      'tev:GmbHCoKG(rform)',
                      # 'tev:KG(rform)',
           # 'tev:Kap(rform)',
                      'tev:KdoeR(rform)',
                      # 'tev:KoerpGemHVKommHV(rform)',
           'tev:Koerperschaften(rform)',
                      # 'tev:OHG(rform)',
           'tev:Personengesellschaft(rform)',
                      # 'tev:Rechtsform(rform)',
          # 'tev:Stiftung(rform)', 'tev:Verein(rform)', 'tev:gKap(rform)',
          # 'tev:sonstKdoeRHGB(rform)'
                  ],

#  'tev:Stiftung(rform)', 'tev:Verein(rform)', 'tev:gKap(rform)', 'tev:sonstKdoeRHGB(rform)']


    "C2(kto)": ['tev:GP_Konto(kto)',
                'tev:Umsatzerlöse(kto)',
                      'tev:Erlösschmälerungen(kto)',
                      'tev:Materialaufwand(kto)',
                      'tev:Personalaufwand_GKV(kto)',
                      'tev:Sonstige_betriebliche_Erträge(kto)',
                      'tev:Sonstige_betriebliche_Aufwendungen(kto)',
                      'tev:Erträge_aus_Beteiligungen(kto)',
                      'tev:Erträge_aus_anderen_Wertpapieren_und_Ausleihungen_des_Finanzanlagevermögens(kto)',
                      'tev:Sonstige_Zinsen_und_ähnliche_Erträge(kto)',
                      'tev:Zinsen_und_ähnliche_Aufwendungen(kto)',
                      'tev:Außerordentliche_Erträge(kto)',
                      'tev:Außerordentliche_Aufwendungen(kto)',
                      'tev:Steuern_vom_Einkommen_und_Ertrag(kto)',
                      'tev:Sonstige_Steuern(kto)',
                ],
    "C3(ggkto)": ['tev:GP_Konto(ggkto)',
                    'tev:Umsatzerlöse(ggkto)',
                      'tev:Erlösschmälerungen(ggkto)',
                      'tev:Materialaufwand(ggkto)',
                      'tev:Personalaufwand_GKV(ggkto)',
                      'tev:Sonstige_betriebliche_Erträge(ggkto)',
                      'tev:Sonstige_betriebliche_Aufwendungen(ggkto)',
                      'tev:Erträge_aus_Beteiligungen(ggkto)',
                      'tev:Erträge_aus_anderen_Wertpapieren_und_Ausleihungen_des_Finanzanlagevermögens(ggkto)',
                      'tev:Sonstige_Zinsen_und_ähnliche_Erträge(ggkto)',
                      'tev:Zinsen_und_ähnliche_Aufwendungen(ggkto)',
                      'tev:Außerordentliche_Erträge(ggkto)',
                      'tev:Außerordentliche_Aufwendungen(ggkto)',
                      'tev:Steuern_vom_Einkommen_und_Ertrag(ggkto)',
                      'tev:Sonstige_Steuern(ggkto)'
    ],

    "P1(pklasse)": ['tev:Produktklasse_C1083(pklasse)',
                      'tev:Produktklasse_C108917(pklasse)',
                      'tev:Produktklasse_C109(pklasse)',
                      'tev:Produktklasse_C11(pklasse)',
                      'tev:Produktklasse_C172(pklasse)',
                      'tev:Produktklasse_C19202(pklasse)',
                      'tev:Produktklasse_C212(pklasse)',
                      'tev:Produktklasse_C2221(pklasse)',
                      'tev:Produktklasse_C2511(pklasse)',
                      'tev:Produktklasse_C252(pklasse)',
                      'tev:Produktklasse_C2594(pklasse)',
                      'tev:Produktklasse_C259912(pklasse)',
                      'tev:Produktklasse_C2620(pklasse)',
                      'tev:Produktklasse_C264(pklasse)',
                      'tev:Produktklasse_C293(pklasse)',
                      'tev:Produktklasse_E381(pklasse)',
                      'tev:Produktklasse_G47(pklasse)',
                      'tev:Produktklasse_H51(pklasse)',
                      'tev:Produktklasse_H53(pklasse)',
                      'tev:Produktklasse_J6110(pklasse)',
                      'tev:Produktklasse_J620(pklasse)',
                      'tev:Produktklasse_J631(pklasse)',
                      'tev:Produktklasse_J6312 (pklasse)',
                      'tev:Produktklasse_L682(pklasse)',
                      'tev:Produktklasse_M6920(pklasse)',
                      'tev:Produktklasse_M712(pklasse)',
                      'tev:Produktklasse_M7120(pklasse)',
                      'tev:Produktklasse_M731(pklasse)',
                      'tev:Produktklasse_N8129(pklasse)',
                      'tev:Produktklasse_N82191(pklasse)',
                      'tev:Produktklasse_R93(pklasse)'],

    "C1(branche)": ['tev:Branche_X(branche)', 'tev:Branche_A(branche)',
                    'tev:Branche_B(branche)', 'tev:Branche_C(branche)',
       'tev:Branche_D(branche)', 'tev:Branche_E(branche)',
       'tev:Branche_F(branche)', 'tev:Branche_G(branche)',
       'tev:Branche_H(branche)', 'tev:Branche_I(branche)',
       'tev:Branche_J(branche)', 'tev:Branche_K(branche)',
       'tev:Branche_L(branche)', 'tev:Branche_M(branche)',
       'tev:Branche_N(branche)', 'tev:Branche_O(branche)',
       'tev:Branche_P(branche)', 'tev:Branche_Q(branche)',
       'tev:Branche_R(branche)', 'tev:Branche_S(branche)',
       'tev:Branche_T(branche)', 'tev:Branche_U(branche)',],

}

learner = el.AtomicLearner(skeletonExpression)
learner.generate_fixedCores_sampleDf(sampleDf,candidatesDict)

learner.random_initialize_variableCoresDict()
learner.set_targetCore(length=sampleDf.values.shape[0])

from tnreason.logic import coordinate_calculus as cc


# label = 'tev:GP_Konto(ggkto)'
# positiveValues = sampleDf[label].values
# positiveCore = cc.CoordinateCore(positiveValues,["j"])


positiveExpression = generate_from_or_expression([
                        'tev:GP_Konto(ggkto)',"or",
                    'tev:Umsatzerlöse(ggkto)',"or",
                      'tev:Erlösschmälerungen(ggkto)',"or",
                      'tev:Materialaufwand(ggkto)',"or",
                      'tev:Personalaufwand_GKV(ggkto)',"or",
                      'tev:Sonstige_betriebliche_Erträge(ggkto)',"or",
                      'tev:Sonstige_betriebliche_Aufwendungen(ggkto)',"or",
                      'tev:Erträge_aus_Beteiligungen(ggkto)',"or",
                      'tev:Erträge_aus_anderen_Wertpapieren_und_Ausleihungen_des_Finanzanlagevermögens(ggkto)',"or",
                      'tev:Sonstige_Zinsen_und_ähnliche_Erträge(ggkto)',"or",
                      'tev:Zinsen_und_ähnliche_Aufwendungen(ggkto)',"or",
                      'tev:Außerordentliche_Erträge(ggkto)',"or",
                      'tev:Außerordentliche_Aufwendungen(ggkto)',"or",
                      'tev:Steuern_vom_Einkommen_und_Ertrag(ggkto)',"or",
                      'tev:Sonstige_Steuern(ggkto)'])

    #["not",[["not","versandt(y,x)"],"and",["not","Rechnung(x)"]]] # Versand oder Rechnung
positiveCore = ec.evaluate_expression_on_sampleDf(sampleDf, positiveExpression)

negativeCore = positiveCore.negate()
learner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)

learner.als(10)
learner.get_solution()
#print(learner.variablesCoresDict[skeletonExpression].values)
print("The solution expression is:")
print(learner.solutionExpression)


# start_index = label.find('(')
# end_index = label.find(')')
# before_parentheses = label[:start_index]
# inside_parentheses = label[start_index + 1:end_index]
#
# print("Nice text of the rule: ")
# print("If ", skeletonExpression, " is ", learner.solutionExpression, "then the ", \
#     inside_parentheses, "is ", before_parentheses )
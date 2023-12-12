import pandas as pd
from tnreason.learning import expression_learning as el

## DATA LOADING ##

# path = "S:/_MariaSF/extractedDataFrame/extracted_20230925_150557.csv" # Sep
path = "S:/_MariaSF/extractedDataFrame/extracted_20231027_102609.csv" # Okt

sampleDf = pd.read_csv(path, index_col=0)
print(sampleDf.var())

## EXPRESSION LEARNING ##

skeletonExpression = [["b1(beleg)","and", "C1(branche)"],
                      "and",
                      [["b2(usatz)","and", "b3(wjb)"],"and",["C2(rform)","and", "b4(bsatz)"]]]

candidatesDict = {
    "b1(beleg)": ["tev:Ausgangsrechnung(beleg)", "tev:Eingangsrechnung(beleg)"],
    "b2(usatz)" :['tev:Umsatzsteuer_0(usatz)', 'tev:Umsatzsteuer_19(usatz)',
       'tev:Umsatzsteuer_5(usatz)', 'tev:Umsatzsteuer_7(usatz)'],
    "b3(wjb)" : ['tev:WJ_20190101(wjb)',
       'tev:WJ_20200101(wjb)', 'tev:WJ_20210101(wjb)', 'tev:WJ_20220101(wjb)',
       'tev:WJ_20230101(wjb)'],
    "b4(bsatz)" : ['tev:Buchungssteuersatz_0.00(bsatz)',
       'tev:Buchungssteuersatz_16.00(bsatz)',
       'tev:Buchungssteuersatz_19.00(bsatz)',
       'tev:Buchungssteuersatz_7.00(bsatz)'],
    "C2(rform)": ['tev:Einzelunternehmen(rform)', 'tev:GbR(rform)',
       'tev:Genossenschaft(rform)', 'tev:GmbHCoKG(rform)', 'tev:KG(rform)',
       'tev:Kap(rform)', 'tev:KdoeR(rform)', 'tev:KoerpGemHVKommHV(rform)',
       'tev:Koerperschaften(rform)', 'tev:OHG(rform)',
       'tev:Personengesellschaft(rform)', 'tev:Rechtsform(rform)',
       'tev:Stiftung(rform)', 'tev:Verein(rform)', 'tev:gKap(rform)',
       'tev:sonstKdoeRHGB(rform)'],

    #"C2(kto)": ['tev:Umsatzerlöse_gesamt(kto)',
    #            'tev:Gesamtkostenverfahren_-_Kosten(kto)', 'tev:Gemeinsamer_Teil(kto)'],
    "C1(branche)": ['tev:Branche_B(branche)', 'tev:Branche_C(branche)',
       'tev:Branche_D(branche)', 'tev:Branche_E(branche)',
       'tev:Branche_F(branche)', 'tev:Branche_G(branche)',
       'tev:Branche_H(branche)', 'tev:Branche_I(branche)',
       'tev:Branche_J(branche)', 'tev:Branche_K(branche)',
       'tev:Branche_L(branche)', 'tev:Branche_M(branche)',
       'tev:Branche_N(branche)', 'tev:Branche_O(branche)',
       'tev:Branche_P(branche)', 'tev:Branche_Q(branche)',
       'tev:Branche_R(branche)', 'tev:Branche_S(branche)',
       'tev:Branche_T(branche)', 'tev:Branche_U(branche)',],
  #  "C3(branche)": ['tev:Branche_A(branche)', 'tev:Branche_B(branche)', 'tev:Branche_D(branche)',
  #                  'tev:Branche_C(branche)']
}

learner = el.AtomicLearner(skeletonExpression)
learner.generate_fixedCores_sampleDf(sampleDf,candidatesDict)

learner.random_initialize_variableCoresDict()
learner.set_targetCore(length=sampleDf.values.shape[0])

learner.als(10)
learner.get_solution()
print("The solution expression is:")
print(learner.solutionExpression)

from tnreason import knowledge
from tnreason import encoding

generatingKB = knowledge.HybridInferer(weightedFormulas=
{
    "f1": ["imp","a","b", 2.567],
    "f2": ["imp","a","c", 2.222],
    "f3": ["a", 1.78]
})

#checkCore = encoding.create_formulas_cores({"f1":["imp","a","b", 2.567]})['(imp_a_b)_conCore']


sampleNum = 200
sampleDf = generatingKB.create_sampleDf(sampleNum)
print(sampleDf.columns)
#print(sampleDf)

# from tnreason.knowledge import weight_estimation as wees
#
# empDistribution = wees.EmpiricalDistribution(sampleDf)
# print(empDistribution.get_satisfactionDict({
#     "f1": ["imp", "a", "b"]}
# ))

expressionsDict = {"f1": ["imp","a","b"], "f2": ["imp","a","c"], "f3": ["a"]}
satisfactionDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(expressionsDict)


entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict, backCores={})
values = entropyMaximizer.alternating_optimization(sweepNum=1)

print(values)


formuladict = {
    "f1" : [
        "eq",
         ["or","a","b"],
         "c",
        1.1
         ],
    "f2": [
        "imp",
        "a",
        ["or","c","d"],
        0.237461234
    ]
}

#print(encoding.create_formulas_cores(formuladict).keys())



knowledge.HybridInferer(weightedFormulas=formuladict)
generatingKB = knowledge.HybridInferer(weightedFormulas=
{
    "f1": ["imp","a","b", 2.567],
    "f2": ["imp","a","c", 2.222],
    "f3": ["a", 1.78]
})
print(generatingKB.create_cores().keys())
print(generatingKB.create_cores()["a_headCore"].colors)
sampleNum = 200
sampleDf = generatingKB.create_sampleDf(sampleNum)




architecture = {
    "neur1" : [
        ["imp","eq"],
        ["a1","a2"],
        ["a3"]
    ],
    "neur2" : [
        ["not"],
        ["neur1", "a1"]
    ]
}

print(encoding.create_architecture(architecture).keys())
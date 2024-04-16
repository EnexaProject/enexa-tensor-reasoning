import unittest

from tnreason import encoding
from tnreason import knowledge

generatingKB = knowledge.HybridKnowledgeBase(weightedFormulasDict=
{
    "f1": [["a", "imp", "b"], 2.567],
    "f2": [["a", "imp", "c"], 2.222],
    "f3": ["a", 1.78]
})

sampleNum = 200
sampleDf = generatingKB.create_sampleDf(sampleNum)


class HybridKBTest(unittest.TestCase):
    def test_convergence(self):
        expressionsDict = {"f1": ["a", "imp", "b"], "f2": ["a", "imp", "c"], "f3": "a"}
        satisfactionDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(expressionsDict)

        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict, backCores={})
        values = entropyMaximizer.alternating_optimization(sweepNum=10)
        for key in values:
            self.assertGreaterEqual(0.1, abs(values[key][-1] - values[key][-2]))

    def test_backCores(self):
        expressionsDict = {"f1": ["a", "imp", "b"], "f2": ["a", "imp", "c"], "f3": "a"}
        satisfactionDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(expressionsDict)
        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict,
                                                      backCores=encoding.create_formulas_cores({
                                                          "fact1" : ["a", "imp", "b"]
                                                      }))
        values = entropyMaximizer.alternating_optimization(sweepNum=2)

        self.assertEquals(0, values["f1"][0])
        self.assertEquals(0, values["f1"][1])